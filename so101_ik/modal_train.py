"""Modal GPU pipeline for the SO-101 seed-conditioned DNN-IK model.

This file is the remote counterpart of the local CLIs in this package
(``so101_ik.generate_dataset``, ``so101_ik.train``, ``so101_ik.benchmark``).
It does not re-implement anything: every remote function shells out to the
same ``python -m so101_ik.<module>`` command you would run locally, with the
dataset and results living on a persistent ``modal.Volume`` mounted at /vol.

Run from the REPO ROOT (the directory that contains ``so101_ik/``), because
``add_local_python_source("so101_ik")`` resolves the package from the cwd:

    pip install modal && modal setup          # once
    modal run so101_ik/modal_train.py --command generate --samples 20000000
    modal run --detach so101_ik/modal_train.py --command train \
        --data-name so101_seed_20000000.npz --out-name so101_seed20m
    modal run so101_ik/modal_train.py --command benchmark --out-name so101_seed20m --n 1000
    modal run --detach so101_ik/modal_train.py --command sweep
    modal run so101_ik/modal_train.py --command download --name so101_seed20m

Volume layout (``so101-ik-data``):

    /vol/so101_seed_<samples>.npz          datasets (NOT committed to git)
    /vol/results/<out_name>/               one training run (model_best.pt, scalers,
                                           training_history.csv, training_curves.png,
                                           test_indices.npy, results.json,
                                           benchmark_summary.json)
    /vol/results/sweep_summary.csv         data-scaling sweep table

GPU type defaults to A10G; override with the environment variable SO101_GPU
(e.g. ``SO101_GPU=A100 modal run ...`` or ``SO101_GPU=H100``). It is read when
this file is imported, which is when ``modal run`` builds the function specs.

The module is import-safe without modal installed: ``import so101_ik.modal_train``
succeeds and only ``modal run`` / calling ``main()`` reports the missing package.
"""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import modal
except ImportError:  # pragma: no cover - exercised on machines without modal
    modal = None

# --------------------------------------------------------------------------
# Constants shared by every command
# --------------------------------------------------------------------------

APP_NAME = "so101-dnn-ik"
VOLUME_NAME = "so101-ik-data"
VOL = "/vol"
REMOTE_CWD = "/root"  # where add_local_python_source() places so101_ik/
RESULTS_SUBDIR = "results"

DEFAULT_SAMPLES = 20_000_000
DEFAULT_SEED_NOISE = 0.5
DEFAULT_RNG_SEED = 42

# The data-scaling sweep from the UR5e methodology, extended to 20M.
SWEEP_SIZES = [
    10_000,
    50_000,
    100_000,
    200_000,
    500_000,
    1_000_000,
    5_000_000,
    10_000_000,
    20_000_000,
]

GPU_TYPE = os.environ.get("SO101_GPU", "A10G")
COMMIT_EVERY_S = int(os.environ.get("SO101_COMMIT_EVERY_S", "600"))

LOCAL_RESULTS_DIR = Path("so101_ik") / RESULTS_SUBDIR

NO_MODAL_MESSAGE = (
    "so101_ik.modal_train needs the `modal` package, which is not installed in\n"
    "this interpreter. This file is only meant to be launched with the Modal CLI\n"
    "from the repo root:\n"
    "    pip install modal\n"
    "    modal setup            # or: modal token set --token-id ... --token-secret ...\n"
    "    modal run so101_ik/modal_train.py --command help\n"
    "Nothing else in so101_ik depends on modal; the local CLIs work without it."
)


# --------------------------------------------------------------------------
# Pure-Python helpers (no modal, no torch; safe to import anywhere)
# --------------------------------------------------------------------------


def dataset_name(samples: int) -> str:
    """File name of the dataset for a given sample count, e.g. so101_seed_20000000.npz."""
    return f"so101_seed_{int(samples)}.npz"


def human_size(n: int) -> str:
    """10_000 -> '10k', 1_000_000 -> '1m', 20_000_000 -> '20m'."""
    n = int(n)
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}m"
    if n % 1_000 == 0:
        return f"{n // 1_000}k"
    return str(n)


def sweep_run_name(size: int) -> str:
    """Name of the results directory for one sweep point, e.g. so101_sweep_1m."""
    return f"so101_sweep_{human_size(size)}"


def parse_sizes(spec: str) -> list[int]:
    """Parse '10k,1m,20000000' into ints; empty string -> SWEEP_SIZES."""
    if not spec.strip():
        return list(SWEEP_SIZES)
    out = []
    for tok in spec.split(","):
        tok = tok.strip().lower().replace("_", "")
        if not tok:
            continue
        mult = 1
        if tok.endswith("k"):
            mult, tok = 1_000, tok[:-1]
        elif tok.endswith("m"):
            mult, tok = 1_000_000, tok[:-1]
        out.append(int(float(tok) * mult))
    return out


def _run(cmd: list[str], cwd: str = REMOTE_CWD) -> None:
    """Run a subprocess with live output; raise on non-zero exit."""
    print("+", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=cwd, check=True)
    print(f"+ done in {time.time() - t0:.1f}s", flush=True)


def _flatten(prefix: str, obj, into: dict) -> None:
    """Flatten nested dicts of scalars into into["a.b.c"] = value (lists are skipped)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}{k}.", v, into)
    elif not isinstance(obj, (list, tuple)):
        into[prefix.rstrip(".")] = obj


def collect_run_summary(out_dir: str, out_name: str, max_samples: int | None = None) -> dict:
    """Build one flat summary row for a training run directory.

    Tolerant by design: results.json keys are flattened as-is, the last row of
    training_history.csv (epoch, train_loss, val_loss, val_mae_rad, val_mae_deg,
    lr) is appended, and benchmark_summary.json contributes convergence numbers
    when present. Missing files just leave their columns empty.
    """
    row: dict = {"run": out_name, "max_samples": max_samples if max_samples else ""}
    out = Path(out_dir)

    rj = out / "results.json"
    if rj.is_file():
        try:
            flat: dict = {}
            _flatten("", json.loads(rj.read_text()), flat)
            row.update({f"results.{k}": v for k, v in flat.items()})
        except Exception as exc:  # noqa: BLE001
            row["results.error"] = repr(exc)

    th = out / "training_history.csv"
    if th.is_file():
        try:
            with th.open() as fh:
                rows = list(csv.DictReader(fh))
            if rows:
                last = rows[-1]
                row["epochs_run"] = last.get("epoch", "")
                for k in ("train_loss", "val_loss", "val_mae_rad", "val_mae_deg", "lr"):
                    row[f"final_{k}"] = last.get(k, "")
                vals = [float(r["val_loss"]) for r in rows if r.get("val_loss")]
                if vals:
                    row["best_val_loss"] = min(vals)
                maes = [float(r["val_mae_deg"]) for r in rows if r.get("val_mae_deg")]
                if maes:
                    row["best_val_mae_deg"] = min(maes)
        except Exception as exc:  # noqa: BLE001
            row["history.error"] = repr(exc)

    bj = out / "benchmark_summary.json"
    if bj.is_file():
        try:
            bench = json.loads(bj.read_text())
            summary = bench.get("summary", bench)
            flat = {}
            _flatten("", summary, flat)
            for key in (
                "dnn.convergence_percent",
                "dnn.median_iterations",
                "dnn.avg_iterations",
                "random.convergence_percent",
                "random.median_iterations",
                "random.avg_iterations",
                "seed_init_baseline.convergence_percent",
                "seed_init_baseline.median_iterations",
                "speedup_x",
                "speedup_vs_seed_init_x",
                "dnn_only_error.pos_err_mm_mean",
                "dnn_only_error.pos_err_mm_median",
                "dnn_only_error.rot_err_deg_mean",
                "dnn_only_error.rot_err_deg_median",
            ):
                if key in flat:
                    row[f"bench.{key}"] = flat[key]
            for k, v in flat.items():
                if k.startswith("dnn_only") and f"bench.{k}" not in row:
                    row[f"bench.{k}"] = v
        except Exception as exc:  # noqa: BLE001
            row["bench.error"] = repr(exc)

    row["model_best_pt"] = (out / "model_best.pt").is_file()
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dict rows to CSV with the union of keys as header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_table(rows: list[dict], cols: list[str]) -> None:
    cols = [c for c in cols if any(c in r for r in rows)]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def _periodic_commit(volume, stop: threading.Event, every_s: int) -> None:
    """Commit the volume every `every_s` seconds until `stop` is set."""
    while not stop.wait(every_s):
        try:
            volume.commit()
            print(f"[volume] periodic commit at {time.strftime('%H:%M:%S')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[volume] periodic commit failed (will retry): {exc!r}", flush=True)


# --------------------------------------------------------------------------
# Modal app (only defined when modal is importable)
# --------------------------------------------------------------------------

if modal is not None:
    app = modal.App(APP_NAME)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    # Linux torch wheels on PyPI bundle CUDA (cu12x), so a plain pip install
    # gives a CUDA-enabled torch. If a future torch release changes that, add
    # extra_index_url="https://download.pytorch.org/whl/cu124" to pip_install.
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.3",
            "numpy>=1.24",
            "pandas>=2.0",
            "scikit-learn>=1.3",
            "matplotlib>=3.7",
            "tqdm>=4.66",
        )
        .add_local_python_source("so101_ik")
    )

    def _vol_path(*parts: str) -> str:
        return "/".join([VOL, *parts])

    def _results_dir(out_name: str) -> str:
        return _vol_path(RESULTS_SUBDIR, out_name)

    # ---------------------------------------------------------------- generate
    @app.function(
        image=image,
        volumes={VOL: volume},
        cpu=8,
        memory=32 * 1024,
        timeout=2 * 60 * 60,
    )
    def generate(
        samples: int = DEFAULT_SAMPLES,
        seed_noise: float = DEFAULT_SEED_NOISE,
        rng_seed: int = DEFAULT_RNG_SEED,
        overwrite: bool = False,
    ) -> str:
        """Generate the FK dataset on the volume with so101_ik.generate_dataset."""
        volume.reload()
        out = _vol_path(dataset_name(samples))
        if os.path.isfile(out) and not overwrite:
            size_gb = os.path.getsize(out) / 1e9
            print(f"[generate] {out} already exists ({size_gb:.2f} GB); skipping. "
                  f"Pass overwrite=True to regenerate.")
            return out
        _run([
            sys.executable, "-m", "so101_ik.generate_dataset",
            "--samples", str(int(samples)),
            "--seed_noise", str(seed_noise),
            "--rng_seed", str(int(rng_seed)),
            "--output", out,
        ])
        volume.commit()
        print(f"[generate] wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB) and committed volume")
        return out

    # ------------------------------------------------------------------- train
    @app.function(
        image=image,
        volumes={VOL: volume},
        gpu=GPU_TYPE,
        cpu=8,
        memory=48 * 1024,
        timeout=24 * 60 * 60,
    )
    def train(
        data_name: str,
        out_name: str,
        epochs: int = 150,
        batch_size: int = 2048,
        max_samples: int | None = None,
        lr: float = 1e-3,
        patience: int = 20,
        grad_clip: float = 1.0,
        seed: int = 42,
    ) -> dict:
        """Train one model with so101_ik.train; results land in /vol/results/<out_name>."""
        import torch  # inside the image

        volume.reload()
        data = _vol_path(data_name)
        if not os.path.isfile(data):
            raise FileNotFoundError(
                f"{data} not found on volume '{VOLUME_NAME}'. Run the generate command first "
                f"(e.g. --command generate --samples {DEFAULT_SAMPLES})."
            )
        out_dir = _results_dir(out_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"[train] torch {torch.__version__}, cuda={torch.cuda.is_available()}, "
              f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
        print(f"[train] data={data} ({os.path.getsize(data) / 1e9:.2f} GB) out_dir={out_dir}")

        cmd = [
            sys.executable, "-m", "so101_ik.train",
            "--data", data,
            "--out_dir", out_dir,
            "--epochs", str(int(epochs)),
            "--batch_size", str(int(batch_size)),
            "--lr", str(lr),
            "--patience", str(int(patience)),
            "--grad_clip", str(grad_clip),
            "--seed", str(int(seed)),
        ]
        if max_samples:
            cmd += ["--max_samples", str(int(max_samples))]

        stop = threading.Event()
        committer = threading.Thread(
            target=_periodic_commit, args=(volume, stop, COMMIT_EVERY_S), daemon=True
        )
        committer.start()
        try:
            _run(cmd)
        finally:
            stop.set()
            committer.join(timeout=5)
            volume.commit()
            print("[train] final volume commit done")

        summary = collect_run_summary(out_dir, out_name, max_samples)
        print("[train] summary:", json.dumps(summary, indent=2, default=str))
        return summary

    # --------------------------------------------------------------- benchmark
    @app.function(
        image=image,
        volumes={VOL: volume},
        cpu=8,
        memory=16 * 1024,
        timeout=4 * 60 * 60,
    )
    def benchmark(
        out_name: str,
        n: int = 1000,
        task: str = "se3",
        no_model: bool = False,
    ) -> dict:
        """Run so101_ik.benchmark against /vol/results/<out_name> on CPU."""
        volume.reload()
        model_dir = _results_dir(out_name)
        out_json = f"{model_dir}/benchmark_summary.json"
        if not no_model and not os.path.isfile(f"{model_dir}/model_best.pt"):
            raise FileNotFoundError(f"{model_dir}/model_best.pt not found; train '{out_name}' first.")
        os.makedirs(model_dir, exist_ok=True)
        cmd = [
            sys.executable, "-m", "so101_ik.benchmark",
            "--model_dir", model_dir,
            "--n", str(int(n)),
            "--out", out_json,
            "--task", task,
        ]
        if no_model:
            cmd.append("--no_model")
        _run(cmd)
        volume.commit()
        with open(out_json) as fh:
            result = json.load(fh)
        return result

    # --------------------------------------------------------- summarize_sweep
    @app.function(image=image, volumes={VOL: volume}, cpu=2, timeout=10 * 60)
    def summarize_sweep(run_names: list[str], sizes: list[int]) -> list[dict]:
        """Collect results.json / history / benchmark of each sweep run into a CSV on the volume."""
        volume.reload()
        rows = [
            collect_run_summary(_results_dir(name), name, size)
            for name, size in zip(run_names, sizes)
        ]
        write_csv(Path(_vol_path(RESULTS_SUBDIR, "sweep_summary.csv")), rows)
        volume.commit()
        return rows

    # ---------------------------------------------------------------- download
    def _download_from_volume(name: str, dest_root: Path = LOCAL_RESULTS_DIR) -> Path:
        """Copy /vol/results/<name> (dir or single file) to ./so101_ik/results/<name>."""
        remote = f"{RESULTS_SUBDIR}/{name}".strip("/")
        entries = list(volume.listdir(remote, recursive=True))
        if not entries:
            raise FileNotFoundError(f"nothing at /{remote} on volume '{VOLUME_NAME}'")
        file_type = modal.volume.FileEntryType.FILE
        files = [e for e in entries if e.type == file_type]
        if not files:
            raise FileNotFoundError(f"/{remote} contains no files")

        copied = 0
        for entry in files:
            rel = entry.path[len(RESULTS_SUBDIR):].lstrip("/") if entry.path.startswith(RESULTS_SUBDIR) else entry.path
            local = dest_root / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            with local.open("wb") as fh:
                for chunk in volume.read_file(entry.path):
                    fh.write(chunk)
            copied += 1
            print(f"  {entry.path} -> {local} ({local.stat().st_size / 1e6:.2f} MB)")
        print(f"[download] copied {copied} file(s) into {dest_root / name}")
        return dest_root / name

    # ------------------------------------------------------------------- main
    HELP = """
so101_ik/modal_train.py commands (always run from the repo root):

  --command generate  [--samples N] [--seed-noise 0.5] [--rng-seed 42] [--overwrite]
        CPU job. Writes /vol/so101_seed_<N>.npz. Skips if the file exists.

  --command train     --data-name so101_seed_20000000.npz --out-name NAME
                      [--epochs 150] [--batch-size 2048] [--max-samples K]
        GPU job (SO101_GPU env var selects A10G/A100/H100). Use `modal run --detach`.
        Results: /vol/results/NAME/. Benchmarks the run afterwards unless --no-bench.

  --command benchmark --out-name NAME [--n 1000] [--task se3] [--no-model]
        CPU job. Writes /vol/results/NAME/benchmark_summary.json.

  --command sweep     [--data-name ...] [--sizes 10k,50k,...,20m] [--parallel]
                      [--epochs 150] [--batch-size 2048] [--n 1000]
        Trains + benchmarks one run per size with --max_samples on the single
        20M file, then writes /vol/results/sweep_summary.csv and a local copy at
        so101_ik/results/sweep_summary.csv.

  --command download  --name NAME
        Copies /vol/results/NAME to ./so101_ik/results/NAME (NAME may also be
        a file such as sweep_summary.csv).

  --command help
"""

    @app.local_entrypoint()
    def main(
        command: str = "help",
        samples: int = DEFAULT_SAMPLES,
        seed_noise: float = DEFAULT_SEED_NOISE,
        rng_seed: int = DEFAULT_RNG_SEED,
        overwrite: bool = False,
        data_name: str = "",
        out_name: str = "",
        epochs: int = 150,
        batch_size: int = 2048,
        max_samples: int = 0,
        lr: float = 1e-3,
        patience: int = 20,
        grad_clip: float = 1.0,
        seed: int = 42,
        n: int = 1000,
        task: str = "se3",
        model: bool = True,
        bench: bool = True,
        sizes: str = "",
        parallel: bool = False,
        name: str = "",
    ):
        command = command.lower().strip()
        data_name = data_name or dataset_name(samples)
        print(f"[modal_train] app={APP_NAME} volume={VOLUME_NAME} gpu={GPU_TYPE} command={command}")

        if command in ("help", ""):
            print(HELP)
            return

        if command == "generate":
            path = generate.remote(samples, seed_noise, rng_seed, overwrite)
            print(f"[generate] dataset ready at {path}")
            return

        if command == "train":
            out_name = out_name or f"so101_seed{human_size(max_samples or samples)}"
            summary = train.remote(
                data_name, out_name, epochs, batch_size,
                max_samples or None, lr, patience, grad_clip, seed,
            )
            print(json.dumps(summary, indent=2, default=str))
            if bench:
                result = benchmark.remote(out_name, n, task, False)
                print(json.dumps(result.get("summary", result), indent=2, default=str))
            print(f"[train] fetch locally with: modal run so101_ik/modal_train.py "
                  f"--command download --name {out_name}")
            return

        if command == "benchmark":
            if not out_name:
                raise SystemExit("--out-name is required for benchmark")
            result = benchmark.remote(out_name, n, task, not model)
            print(json.dumps(result.get("summary", result), indent=2, default=str))
            return

        if command == "sweep":
            size_list = parse_sizes(sizes)
            run_names = [sweep_run_name(s) for s in size_list]
            print(f"[sweep] sizes={size_list}")
            print(f"[sweep] runs={run_names}")
            # Make sure the single source file exists (no-op if already generated).
            generate.remote(samples, seed_noise, rng_seed, False)

            train_args = [
                (data_name, rn, epochs, batch_size, s, lr, patience, grad_clip, seed)
                for rn, s in zip(run_names, size_list)
            ]
            if parallel:
                print("[sweep] launching all trainings in parallel (one GPU container each)")
                for summary in train.starmap(train_args):
                    print(json.dumps(summary, default=str))
                for result in benchmark.starmap([(rn, n, task, False) for rn in run_names]):
                    s = result.get("summary", result)
                    print(json.dumps({k: s.get(k) for k in ("label", "dnn", "random", "speedup_x")}, default=str))
            else:
                for args in train_args:
                    print(f"[sweep] training {args[1]} (max_samples={args[4]:,})")
                    summary = train.remote(*args)
                    print(json.dumps(summary, default=str))
                    result = benchmark.remote(args[1], n, task, False)
                    s = result.get("summary", result)
                    print(json.dumps({k: s.get(k) for k in ("label", "dnn", "random", "speedup_x")}, default=str))

            rows = summarize_sweep.remote(run_names, size_list)
            local_csv = LOCAL_RESULTS_DIR / "sweep_summary.csv"
            write_csv(local_csv, rows)
            print(f"\n[sweep] wrote {local_csv} (and /vol/results/sweep_summary.csv)")
            print_table(rows, [
                "run", "max_samples", "epochs_run", "final_train_loss", "final_val_loss",
                "best_val_mae_deg", "bench.dnn.convergence_percent",
                "bench.random.convergence_percent", "bench.dnn.median_iterations", "bench.speedup_x",
            ])
            return

        if command == "download":
            if not name:
                raise SystemExit("--name is required for download (e.g. --name so101_seed20m)")
            _download_from_volume(name)
            return

        raise SystemExit(f"unknown command {command!r}\n{HELP}")


def main_without_modal() -> None:  # pragma: no cover
    raise SystemExit(NO_MODAL_MESSAGE)


if __name__ == "__main__":
    if modal is None:
        main_without_modal()
    print("This file is a Modal app. Launch it with:\n"
          "    modal run so101_ik/modal_train.py --command help")
