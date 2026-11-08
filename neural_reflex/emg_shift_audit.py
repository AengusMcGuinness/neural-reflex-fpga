"""Post hoc input-shift diagnosis for a frozen EMG ridge decoder; no fitting.

Task, posture, and electrode-placement differences are confounded in these
recordings. Statistics and plots describe a shift, not its cause or a fix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .emg_benchmark import CONFIG, Decoder, TARGETS, prepare
from .emg_recordings import load_segments
from .emg_transfer_stats import paired_segments


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(name, path, pair, model):
    x, y = pair["x"], pair["y"]
    rms = np.expm1(x[:, -1, 16:24])
    z = (x[:, ::5] - model.scaler["mean"]) / model.scaler["scale"]
    unclipped = pair["clip_fraction"] == 0
    log_rms_z = (x[:, -1, 16:24] - model.scaler["mean"][16:24]) / model.scaler["scale"][16:24]
    predicted = model.predict(x)
    quantiles = np.percentile(rms, [10, 50, 90], axis=0)
    return {
        "name": name, "windows": int(len(x)), "unclipped_windows": int(unclipped.sum()),
        "rms_p10_by_channel": quantiles[0].tolist(),
        "rms_median_by_channel": quantiles[1].tolist(),
        "rms_p90_by_channel": quantiles[2].tolist(),
        "median_unclamped_train_z_log_rms_by_channel": np.median(log_rms_z, axis=0).tolist(),
        "proportion_abs_z_gt6_model_frames_all_features": float(np.mean(np.abs(z) > 6)),
        "proportion_windows_any_abs_z_gt6_model_frames": float(np.mean(np.any(np.abs(z) > 6, axis=(1, 2)))),
        "proportion_abs_z_gt6_model_frames_log_rms_by_channel":
            np.mean(np.abs(z[:, :, 16:24]) > 6, axis=(0, 1)).tolist(),
        "frozen_prediction_mean_deg": predicted.mean(axis=0).tolist(),
        "measured_mean_deg": np.nanmean(y, axis=0).tolist(),
        "frozen_prediction_fraction_at_zero": np.mean(predicted <= 1e-9, axis=0).tolist(),
        "frozen_prediction_fraction_at_180": np.mean(predicted >= 180 - 1e-9, axis=0).tolist(),
        "input_sha256": {filename: sha256(path / filename) for filename in ("emg.jsonl", "arm.jsonl")},
    }, rms


def plot_rms(distributions, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    colors = plt.get_cmap("tab20", len(distributions))(np.arange(len(distributions)))
    channels = np.arange(1, 9)
    offsets = np.linspace(-.28, .28, len(distributions)) if len(distributions) > 1 else [0]
    for (name, values), offset, color in zip(distributions, offsets, colors):
        q10, median, q90 = np.percentile(values, [10, 50, 90], axis=0)
        ax.errorbar(channels + offset, median,
                    yerr=[median - q10, q90 - median], fmt="o", capsize=3,
                    linewidth=1.5, markersize=5, color=color, label=name)
    ax.set_yscale("log")
    ax.set_xticks(channels, [f"ch{channel}" for channel in channels])
    ax.set_xlim(.45, 8.55)
    ax.set_ylabel("Filtered 100 ms RMS (ADC counts, log scale)")
    ax.set_xlabel("EMG channel")
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False, ncol=min(len(distributions), 4))
    ax.set_title("Frozen decoder input shift · median and 10th–90th percentile spread\n"
                 "Distribution spread, not confidence intervals; task, posture and electrode placement are confounded")
    fig.savefig(destination / "rms_distributions.png", dpi=170)
    fig.savefig(destination / "rms_distributions.pdf")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="original reach training block")
    parser.add_argument("blocks", nargs="+", type=Path, help="later blocks; no refitting")
    parser.add_argument("--models", required=True, type=Path, help="frozen ridge.npz or its directory")
    parser.add_argument("--side", required=True, choices=("left", "right"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    model_path = args.models / "ridge.npz" if args.models.is_dir() else args.models
    model = Decoder.load(model_path)
    if model.kind != "ridge":
        parser.error("--models must refer to the frozen ridge decoder")
    if args.output.exists():
        parser.error("output exists; choose a fresh diagnostic directory")

    _, splits = prepare(args.source, args.side)
    all_pairs = [("Training · reach", args.source, splits["train"], None)]
    for block in args.blocks:
        segments, acquisition = load_segments(block, args.side, clock_mode="unit")
        all_pairs.append((block.name, block, paired_segments(segments, history=CONFIG["history"]), acquisition))
    report, distributions = [], []
    for name, path, pair, acquisition in all_pairs:
        item, rms = describe(name, path, pair, model)
        if acquisition is not None:
            item["acquisition"] = {key: acquisition[key] for key in
                                   ("segment_count", "connection_epoch_count", "sequence_missing_batches",
                                    "clock_mode")}
        report.append(item)
        distributions.append((name, rms))

    args.output.mkdir(parents=True)
    payload = {"description": "Post hoc frozen-model input-shift diagnosis; no model or scaler updated.",
               "limitations": "Task, posture, and electrode-placement differences are confounded. "
                              "Fifty-millisecond camera label offset remains assumed; this is not a causal correction.",
               "model": {"kind": model.kind, "sha256": sha256(model_path), "targets": TARGETS},
               "latest_frame_rms": "expm1(feature 16:24) in the newest of 31 causal feature frames",
               "z_definition": "All paired windows, including ADC near-rail windows: "
                               "(feature - original training mean) / original training scale before the model's +/-6 clamp. "
                               "The ridge then clips each z to +/-6 over its seven selected history frames.",
               "recordings": report}
    (args.output / "shift.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    plot_rms(distributions, args.output)
    print(f"Saved frozen input-shift diagnostics for {len(report)} recordings to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
