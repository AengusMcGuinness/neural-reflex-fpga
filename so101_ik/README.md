# so101_ik — seed-conditioned DNN-initialised inverse kinematics for the SO-101

Port of the ES 259 UR5e study ("DNN seed -> damped Newton-Raphson polish") to the
5-DOF SO-101 arm (TheRobotStudio SO-ARM100, `Simulation/SO101/so101_new_calib.urdf`).
The method is unchanged; the only deviation is 5 kinematic joints instead of 6,
so the network input is 14D and the output 10D. The gripper jaw is not a
kinematic joint and is excluded everywhere.

Joint order everywhere: `JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")`.

## Modules

| Module | What it is |
|---|---|
| `fk.py` | PoE forward kinematics. `JOINTS`, `N_JOINTS=5`, `JOINT_LIMITS` (rad), `skew`, `build_so101_model(tool_offset_m)` -> `{"w","v","M","limits","names"}`, `forward_kinematics(q, model)`, `fk_batch(q, model)` (vectorised over samples, loop over joints), `space_jacobian(q, model)` (6x5, rows `[linear; angular]`, space frame), `urdf_chain_fk(q, tool_offset_m)` (naive URDF-chain reference used by tests). `tool_offset_m` is an extra translation in the `gripper_frame_link` frame (pen tip). |
| `representations.py` | `encode_sincos` (5 -> 10D), `decode_sincos` (atan2), `decode_sincos_torch`, `normalize_sincos` (reshape `(-1,5,2)`, L2-normalise each pair), `rotmat_to_6d`, `r6d_to_rotmat`, `extract_pose_9d`, `POSE9D_COLS`, `JOINT_COLS` (`q1..q5`), `SEED_COLS` (`seed_q1..seed_q5`), `SINCOS_COLS`. |
| `model.py` | `SeedConditionedIKModel` (`INPUT_DIM=14`, `OUTPUT_DIM=10`), hidden stack 256 -> 512 -> 512 -> 256 with BatchNorm + Mish, residual skip 512->512 inside block 3 and skip from block-1 output into block-4 output, identical to the UR5e `ik_v3/model.py`. `build_model()`. |
| `generate_dataset.py` | CLI, writes the `.npz` dataset (format below). |
| `train.py` | CLI, trains the model and writes the results directory (below). |
| `newton.py` | `ik_newton_refine(T_target, q_init, model, task="se3"|"pen", max_iter=200, eps_v_mm=1.0, eps_w_rad=0.01)` -> `(q, iters, pos_err_mm, rot_err_deg, converged)`. Damped least squares `dq = solve(J^T J + lam I, J^T e)`, `lam = 0.01*||e||`, step clamp 0.5 rad, clip to `JOINT_LIMITS`. `task="se3"` is the 6D twist error of the reference (only converges for poses reachable by this 5-DOF arm, i.e. poses produced by FK; that is what the benchmark uses). `task="pen"` is the writing task: error `[position(3); theta*unit(cross(current_tool_z, desired_tool_z))]` (geodesic axis error, with a perpendicular fallback for antiparallel axes); the wrist_roll Jacobian column is zeroed, so roll stays exactly at its seed (the pen tip is ~8 mm off the roll axis, so a "free" roll drifted through its range). Metres internally. |
| `infer.py` | `IKSolver(model_dir, device="cpu", tool_offset_m=(0,0,0))` with `solve(T_target, q_seed)`, `solve_batch`, `verify(T_target, q)` -> `{pos_error_mm, rot_error_deg, T_predicted}`, `solve_refined(T_target, q_seed, task="se3"|"pen", **newton_kwargs)` -> `(q, info)` where `info` has `converged, iters, pos_err_mm, rot_err_deg, q_dnn`. |
| `benchmark.py` | CLI replicating the UR5e `benchmark_ik2.py`: DNN-start Newton vs best-of-5 uniform-random starts. |
| `modal_train.py` | Modal app `so101-dnn-ik` for GPU training (see `SO101_IK_TRAINING.md` at the repo root). Import-safe without `modal`. |

## Recipe (identical to the UR5e study)

* Data: `q* ~ Uniform(JOINT_LIMITS)`, `T = FK(q*)`, `pose9d = [x, y, z, r11, r21, r31, r12, r22, r32]` (first two columns of R, Zhou et al. 6D rotation), `seed = clip(q* + N(0, 0.5 rad), limits)`, target = sin/cos of `q*`.
* Inputs: independent `StandardScaler`s for pose (9D) and seed (5D), fit on the train split only, concatenated to 14D.
* Loss: `NormalizedSinCosLoss` = MSE after L2-normalising each predicted (sin, cos) pair.
* Optimiser: Adam 1e-3, `ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)`, batch 2048, grad-norm clip 1.0, up to 150 epochs, early stop patience 20, 80/10/10 split (seeded).
* Decode: normalise, then `atan2(sin, cos)` per joint.
* Inference: DNN guess -> damped Newton-Raphson polish (`newton.py`).

## CLIs

All commands run from the repo root with `-m`:

```bash
PY=.venv-ik/bin/python   # any python 3.11+ with torch/numpy/pandas/scikit-learn/matplotlib/tqdm

# dataset (20M samples ~ 1.5 GB npz, generated in 100k batches, < 15 GB RAM)
$PY -m so101_ik.generate_dataset --samples 20000000 --seed_noise 0.5 --rng_seed 42 \
    --output so101_seed_20000000.npz [--csv]

# training (uses CUDA when available; on CPU --num_threads T)
$PY -m so101_ik.train --data so101_seed_20000000.npz --out_dir so101_ik/results/so101_seed20m \
    [--epochs 150 --batch_size 2048 --lr 1e-3 --patience 20 --grad_clip 1.0 --seed 42 \
     --max_samples K --num_threads T]

# benchmark: DNN-init vs random-init Newton on 1000 random FK poses
$PY -m so101_ik.benchmark --model_dir so101_ik/results/so101_seed20m --n 1000 \
    [--out so101_ik/results/so101_seed20m/benchmark_summary.json --task se3]
$PY -m so101_ik.benchmark --no_model --n 1000      # random-init baseline only
```

## File formats

### Dataset `.npz`

Written by numpy, float32 arrays:

| key | shape | content |
|---|---|---|
| `pose9d` | `(N, 9)` | `[x, y, z, r11, r21, r31, r12, r22, r32]`, metres |
| `seed` | `(N, 5)` | seed joint angles (rad) = clipped `q + N(0, seed_noise)` |
| `q` | `(N, 5)` | ground-truth joint angles (rad) |
| `meta` | `()` string | small JSON: samples, seed_noise, rng_seed, joint names, limits |

`--csv` additionally writes a CSV with columns `POSE9D_COLS + SEED_COLS + JOINT_COLS`.
The dataset is deterministic given `--rng_seed`, so it is never stored in git.

### Results directory (`train.py --out_dir`)

| file | content |
|---|---|
| `model_best.pt` | `state_dict` of the best-val-loss epoch |
| `pose_scaler.pkl`, `seed_scaler.pkl` | pickled sklearn `StandardScaler`s (fit on train split) |
| `training_history.csv` | columns `epoch, train_loss, val_loss, val_mae_rad, val_mae_deg, lr` |
| `training_curves.png` | loss / val MAE / LR plots |
| `test_indices.npy` | indices of the held-out 10% test split |
| `results.json` | final metrics (best val loss, test MAE in rad/deg, per-joint MAE, % errors < 1/5/10 deg, epochs run, device, sample count, args) |
| `benchmark_summary.json` | written by `benchmark.py`: `summary.{dnn,random,seed_init_baseline}.{converged, convergence_percent, avg_iterations, median_iterations, mean_ms, ...}`, `summary.speedup_x`, `summary.speedup_vs_seed_init_x`, `summary.dnn_only_error.{pos_err_mm_mean, pos_err_mm_median, rot_err_deg_mean, rot_err_deg_median}`, `summary.timing_ms`, plus the first 30 examples |

Trained artefacts for the hand-off live under `so101_ik/results/<run_name>/` and the data-scaling table in `so101_ik/results/sweep_summary.csv`.
