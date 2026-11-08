# SO-101 DNN-IK: GPU training hand-off (Modal)

This document is written for an AI agent (or a person) with a Modal account who
will generate the dataset, train the models, run the benchmarks and send the
artefacts back. Follow the commands in order. Every path and flag below matches
the package contract in `so101_ik/README.md`; do not rename anything.

## 1. What this is

We train a small seed-conditioned residual MLP that predicts the joint angles of
the 5-DOF SO-101 arm for a target tool pose, and use that prediction to
initialise a damped Newton-Raphson polish, which converges far more often and in
far fewer iterations than a randomly initialised solver. The method is a port of
a UR5e study whose methodology write-up is not available to you, so the recipe is
restated here in full; the code in `so101_ik/` already implements it and you only
need to run it. The one sanctioned deviation from the UR5e study is that the
SO-101 has 5 kinematic joints (the gripper jaw is excluded), so the network input
is 14D and the output 10D.

The recipe (already implemented, listed so you can sanity-check logs):

| Item | Value |
|---|---|
| Robot | SO-101 (TheRobotStudio SO-ARM100, `so101_new_calib.urdf`), joints `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll` |
| Data | `q* ~ Uniform(joint limits)`, `T = FK(q*)` (product of exponentials), `pose9d = [x, y, z] + first two columns of R` (6D continuous rotation), `seed = clip(q* + N(0, 0.5 rad), limits)`, target = sin/cos of `q*` |
| Input / output | 14D = 9D pose + 5D seed (two independent `StandardScaler`s, fit on the train split) -> 10D = 5 (sin, cos) pairs |
| Architecture | `14 -> 256 -> 512 -> 512 -> 256 -> 10`, BatchNorm + Mish after every linear, residual skip 512->512 in block 3, skip from block-1 output (256) into block-4 output; ~0.53M params |
| Loss | `NormalizedSinCosLoss`: L2-normalise each predicted (sin, cos) pair, then MSE against the targets |
| Optimiser | Adam lr 1e-3, `ReduceLROnPlateau(factor 0.5, patience 5, min_lr 1e-6)`, batch 2048, grad-norm clip 1.0, up to 150 epochs, early stop patience 20, 80/10/10 split |
| Decode | normalise pairs, `atan2(sin, cos)` |
| Inference | DNN guess -> damped Newton-Raphson (`dq = solve(J^T J + lam I, J^T e)`, `lam = 0.01*||e||`, 0.5 rad step clamp, tolerance 1 mm and 0.01 rad, max 200 iterations) |

The experiments to reproduce are (a) one production model on 20M samples and
(b) the data-scaling sweep 10k, 50k, 100k, 200k, 500k, 1M, 5M, 10M, 20M, each
trained with `--max_samples` on the same 20M file, each benchmarked against a
random-initialised Newton baseline on 1,000 poses.

## 2. Prerequisites

```bash
git clone https://github.com/AengusMcGuinness/neural-reflex-fpga.git
cd neural-reflex-fpga
git checkout feat/so101-dnn-ik

python3 --version          # 3.11 or newer
python3 -m pip install --upgrade modal
modal setup                # opens a browser; or: modal token set --token-id ... --token-secret ...
modal profile current      # confirm the workspace that has GPU credit
```

Nothing else needs installing locally: all Python dependencies (torch CUDA wheel,
numpy, pandas, scikit-learn, matplotlib, tqdm) are installed into the Modal image
on first run. If you want to run the local CLIs too, `pip install torch numpy
pandas scikit-learn matplotlib tqdm` in a venv.

Always run `modal run` from the repository root (the directory containing
`so101_ik/`); the app mounts the package with `add_local_python_source("so101_ik")`
relative to the cwd. Print the command list at any time:

```bash
modal run so101_ik/modal_train.py --command help
```

What the app does: app name `so101-dnn-ik`, a persistent volume `so101-ik-data`
mounted at `/vol` (created automatically on first use), one CPU function per
`generate` / `benchmark`, one GPU function `train` (A10G by default; set
`SO101_GPU=A100` or `SO101_GPU=H100` in the environment of the `modal run`
command to change it), and a `download` helper that copies `/vol/results/<name>`
into `./so101_ik/results/<name>` locally. The volume is committed every 10
minutes during training and at the end, so a dropped client never loses a run.

## 3. Commands, in order

### 3.1 Generate the 20M dataset (CPU, ~5-15 minutes end to end)

```bash
modal run so101_ik/modal_train.py --command generate --samples 20000000 --seed-noise 0.5 --rng-seed 42
```

* The first invocation also builds the image (torch download, ~3-6 min). The
  generation itself is vectorised numpy FK in 100k-sample batches (~1-3 min) plus
  writing ~1.5 GB (`20M x 19 floats x 4 B`).
* Output: `/vol/so101_seed_20000000.npz` with float32 arrays `pose9d (N,9)`,
  `seed (N,5)`, `q (N,5)` and a JSON `meta` string. The log ends with
  `[generate] wrote /vol/so101_seed_20000000.npz (1.5x GB) and committed volume`.
* The dataset is fully deterministic from `--rng_seed 42`, so it is regenerated
  on demand and is **never committed to git**. Re-running the command is a no-op
  if the file exists (`--overwrite` forces regeneration).
* Check: `modal volume ls so101-ik-data` should list `so101_seed_20000000.npz`.

### 3.2 Train the production 20M model (GPU)

```bash
modal run --detach so101_ik/modal_train.py --command train \
    --data-name so101_seed_20000000.npz --out-name so101_seed20m \
    --epochs 150 --batch-size 2048
```

`--detach` keeps the job running if your terminal disconnects. Follow it with
`modal app logs so101-dnn-ik` (or the URL that `modal run` prints). The job runs
`python -m so101_ik.train --data /vol/so101_seed_20000000.npz --out_dir
/vol/results/so101_seed20m --epochs 150 --batch_size 2048 --lr 1e-3 --patience 20
--grad_clip 1.0 --seed 42` and then, unless you pass `--no-bench`, immediately
runs the 1,000-pose benchmark on CPU and writes
`/vol/results/so101_seed20m/benchmark_summary.json`.

Expected duration on an A10G: **roughly 2-7 hours, budget 8**. Reasoning: 16M
training rows / 2048 = ~7,800 optimiser steps per epoch plus ~1,000 validation
steps. The network is ~0.5M parameters, so the GPU is never compute-bound; a step
costs ~1 ms of kernels and the rest is host-side batching. If `train.py` feeds
batches through a plain `DataLoader(TensorDataset)` as the UR5e script did, that
is ~10-20 ms per step, i.e. ~2-3 minutes per epoch and ~5-7 hours for 150 epochs;
if it slices GPU-resident tensors it is ~30 s per epoch and ~1.5 hours. The UR5e
10M run used all 150 epochs without early stopping, so expect the same here. The
function timeout is 24 hours. An A100/H100 does not help much for this model
size; use it only if A10G capacity is unavailable (`SO101_GPU=A100 modal run ...`).
The container requests 48 GiB RAM because the 20M arrays plus scaler copies peak
at several GB; GPU memory is not a concern (batch 2048 x 14 floats).

Output directory `/vol/results/so101_seed20m/`:
`model_best.pt`, `pose_scaler.pkl`, `seed_scaler.pkl`, `training_history.csv`,
`training_curves.png`, `test_indices.npy`, `results.json`, `benchmark_summary.json`.

Then copy it locally:

```bash
modal run so101_ik/modal_train.py --command download --name so101_seed20m
ls so101_ik/results/so101_seed20m/
```

### 3.3 Benchmark only (if you need to re-run it)

```bash
modal run so101_ik/modal_train.py --command benchmark --out-name so101_seed20m --n 1000 --task se3
modal run so101_ik/modal_train.py --command benchmark --out-name baseline --n 1000 --no-model   # random-init only
```

CPU job, ~10-20 minutes for 1,000 poses (each pose runs Newton from the DNN start
and from 5 random starts, up to 200 iterations each).

### 3.4 Data-scaling sweep (GPU, the main experiment)

```bash
# sequential: one GPU container after another, ~1.5-1.8x the duration of the 20M run (4-12 h)
modal run --detach so101_ik/modal_train.py --command sweep --epochs 150 --batch-size 2048 --n 1000

# or all nine sizes at once (nine GPU containers; wall time ~ the 20M run alone)
modal run --detach so101_ik/modal_train.py --command sweep --parallel
```

The sweep reuses `/vol/so101_seed_20000000.npz` (it calls `generate` first, which
is a no-op when the file exists) and trains one run per size with
`--max_samples {10000, 50000, 100000, 200000, 500000, 1000000, 5000000, 10000000, 20000000}`,
written to `/vol/results/so101_sweep_10k`, `so101_sweep_50k`, ..., `so101_sweep_20m`.
Each run is benchmarked on 1,000 poses. Small sizes finish in minutes (and
early-stop before 150 epochs, as in the UR5e study); 10M and 20M dominate the
total. `--sizes 10k,100k,1m` restricts the sweep if you need to test the pipeline
first. At the end the app writes `/vol/results/sweep_summary.csv`, copies it to
`./so101_ik/results/sweep_summary.csv` on your machine and prints the table
(columns include `run, max_samples, epochs_run, final_train_loss, final_val_loss,
best_val_mae_deg, bench.dnn.convergence_percent, bench.random.convergence_percent,
bench.dnn.median_iterations, bench.speedup_x`, plus every scalar in each
`results.json` prefixed `results.`).

Then download every run:

```bash
for r in so101_sweep_10k so101_sweep_50k so101_sweep_100k so101_sweep_200k so101_sweep_500k \
         so101_sweep_1m so101_sweep_5m so101_sweep_10m so101_sweep_20m; do
  modal run so101_ik/modal_train.py --command download --name $r
done
modal run so101_ik/modal_train.py --command download --name sweep_summary.csv
```

(`so101_sweep_20m` and `so101_seed20m` are the same configuration trained twice;
keep both, the duplicate is a useful seed-to-seed consistency check.)

## 4. How to verify success

1. `results.json` exists in every run directory and reports the test-split
   metrics (best validation loss, test MAE in radians and degrees, per-joint MAE,
   percentage of joint errors below 1/5/10 degrees, number of epochs run, device,
   number of samples used). `training_history.csv` has columns
   `epoch, train_loss, val_loss, val_mae_rad, val_mae_deg, lr` and `val_loss`
   should decrease monotonically-ish with the LR halving a few times.
2. Accuracy targets, based on the UR5e study (val MAE 14.4 deg at 10k, 7.4 deg at
   100k, 4.4 deg at 1M, 3.06 deg at 10M; every 10x data roughly halves the MAE):
   expect **val MAE around 3 degrees or better at 10M and 20M**. The SO-101 has
   fewer joints and narrower limits than the UR5e, so somewhat better numbers are
   plausible; a 20M run worse than ~5 degrees means something is wrong (check the
   log for `cuda=True`, the loss going below ~0.01, and that `--max_samples` was
   not accidentally set).
3. `benchmark_summary.json`: `summary.dnn.convergence_percent` must be clearly
   **higher than `summary.random.convergence_percent`**, with a lower
   `dnn.median_iterations` and `speedup_x > 1`. For reference the UR5e 10M model
   gave 77% vs 44% convergence, median 7 iterations and ~4.9x speedup. Also look
   at the DNN-only error fields (position in mm, rotation in degrees before the
   Newton polish); they should be on the order of millimetres-to-centimetres and
   a few degrees for the big runs.
4. `sweep_summary.csv`: MAE should fall and DNN convergence should rise
   monotonically with sample count, while the random baseline stays flat.
5. `training_curves.png` opens and shows train/val loss, val MAE in degrees and
   the LR schedule.

Also run the local checks if you have a venv with torch (optional, CPU, minutes):

```bash
python -m so101_ik.benchmark --model_dir so101_ik/results/so101_seed20m --n 50
```

## 5. What to send back

Create a branch from `feat/so101-dnn-ik` and commit **only** these files
(each `model_best.pt` is ~2 MB, so the whole set is a few tens of MB):

```
so101_ik/results/so101_seed20m/{model_best.pt,pose_scaler.pkl,seed_scaler.pkl,training_history.csv,training_curves.png,results.json,benchmark_summary.json}
so101_ik/results/so101_sweep_{10k,50k,100k,200k,500k,1m,5m,10m,20m}/{same seven files}
so101_ik/results/sweep_summary.csv
```

Do **not** commit `test_indices.npy` (large, reproducible from `--seed 42`) and
do **not** commit any `.npz` or `.csv` dataset: the 20M dataset is ~1.5 GB and is
regenerated bit-for-bit with `--rng_seed 42`. If `git status` shows an `.npz`,
remove it from the commit.

```bash
git checkout -b feat/so101-dnn-ik-results
git add so101_ik/results/*/model_best.pt so101_ik/results/*/pose_scaler.pkl so101_ik/results/*/seed_scaler.pkl \
        so101_ik/results/*/training_history.csv so101_ik/results/*/training_curves.png \
        so101_ik/results/*/results.json so101_ik/results/*/benchmark_summary.json so101_ik/results/sweep_summary.csv
git commit -m "SO-101 DNN-IK: trained models, data-scaling sweep and benchmarks (Modal A10G)"
git push -u origin feat/so101-dnn-ik-results
```

Open a pull request **into `feat/so101-dnn-ik`** (not `main`) with the title

```
SO-101 DNN-IK: trained models, data-scaling sweep and benchmarks (Modal GPU runs)
```

and paste the `sweep_summary.csv` table plus the 20M `benchmark_summary.json`
summary block and the GPU type / wall-clock time of the 20M run into the
description.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'so101_ik'` in the container or when `modal run` builds the image | You are not in the repo root. `cd` to the directory that contains `so101_ik/` and re-run. |
| `modal_train needs the modal package` | `python3 -m pip install modal`; make sure the `modal` CLI you run comes from the same interpreter (`python3 -m modal run ...` works too). |
| `Volume ... not found` / `NotFoundError` | The volume is created with `create_if_missing=True` on first use, so this means a different Modal profile/workspace is active. `modal profile list`, `modal profile activate <name>`, then `modal volume list`. |
| `FileNotFoundError: /vol/so101_seed_20000000.npz not found` | Run step 3.1 first; check with `modal volume ls so101-ik-data`. |
| `cuda=False` in the train log | The image did not get a CUDA torch wheel. Edit `so101_ik/modal_train.py` and add `extra_index_url="https://download.pytorch.org/whl/cu124"` to `pip_install(...)` (Linux PyPI wheels normally already include CUDA). Also confirm `gpu=` is set on `train` (it is, `A10G` by default). |
| `CUDA out of memory` | Essentially impossible with this model at batch 2048; if it happens, something else is on the GPU. Re-run; as a last resort `--batch-size 1024` (note this deviates from the recipe, say so in the PR). |
| Container killed / `OOM` without "CUDA" (host RAM) | Raise `memory=` in the `train` decorator (default 48 GiB) or use `--max-samples 10000000`. |
| `Timeout` after 24 h | Should not happen; if per-epoch time in the log is > 8 minutes, switch to `SO101_GPU=A100` and/or check that the data loader is the bottleneck (CPU at 100%, GPU idle). |
| GPU capacity errors on A10G | `SO101_GPU=A100 modal run ...` or `SO101_GPU=L4 ...`. |
| Client disconnected during a long run | Nothing is lost: results are committed to the volume every 10 min. `modal app list`, `modal app logs so101-dnn-ik`, then `--command download`. |
| Image build fails on `pip_install` | Re-run (transient PyPI errors); pins are `torch>=2.3 numpy>=1.24 pandas>=2.0 scikit-learn>=1.3 matplotlib>=3.7 tqdm>=4.66`. |
| `download` says nothing at `/results/<name>` | `modal volume ls so101-ik-data results` to see the exact run names. |
| Sweep rows missing benchmark columns | The benchmark for that run failed or was skipped; run `--command benchmark --out-name so101_sweep_<size>` and then re-run `--command sweep --sizes <size>` or just re-download. |

Costs (order of magnitude, at Modal's list prices): A10G ~$1.1/h, so the 20M run
is ~$5-10 and the full sweep ~$10-25; CPU generation and benchmarks are cents.
