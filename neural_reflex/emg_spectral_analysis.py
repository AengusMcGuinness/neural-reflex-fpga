"""Compare EMG spectral features with camera angles without fitting a decoder.

The source-selected pair checks are exploratory. The same features and fixed
label alignment are reused across recordings and window-length sensitivities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .emg_benchmark import TARGETS
from .emg_recordings import load_segments
from .emg_spectral_features import FEATURE_NAMES, paired_spectral_segments
from .emg_spectral_stats import correlation_summary, selected_intervals
from .emg_transfer_stats import nullsafe

PROTOCOL = Path(__file__).resolve().parents[1] / "reports/emg-20261107-spectral/protocol.json"
FEATURE_LABELS = ("Log RMS (reference)", "Log power 20–80 Hz", "Log power 80–150 Hz",
                  "Log power 150–250 Hz", "Log power 250–450 Hz", "Relative power 20–80 Hz",
                  "Relative power 80–150 Hz", "Relative power 150–250 Hz", "Relative power 250–450 Hz",
                  "Mean frequency", "Median frequency", "Spectral entropy")
TARGET_LABELS = ("Shoulder elevation", "Elbow bend", "Wrist bend proxy")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_source_pairs(summary):
    """Freeze one spectral pair per joint using source within-cue effect size."""
    matrix = np.asarray(summary["cue_demeaned_r"], float)
    pairs = []
    for j, target in enumerate(TARGETS):
        scores = np.abs(matrix[:, 1:, j])
        if not np.isfinite(scores).any():
            raise ValueError(f"no finite source spectral relationship for {target}")
        c, f = np.unravel_index(np.nanargmax(scores), scores.shape)
        f += 1
        pairs.append({"label": f"Source-selected {target}", "channel": int(c),
                      "feature_index": int(f), "feature_name": FEATURE_NAMES[f],
                      "target_index": j, "target": target})
    for j in (0, 1):
        pairs.append({"label": f"EMG 1 RMS reference {TARGETS[j]}", "channel": 0,
                      "feature_index": 0, "feature_name": FEATURE_NAMES[0],
                      "target_index": j, "target": TARGETS[j]})
    return pairs


def exploratory_maxima(summary):
    out = {}
    for key in ("pearson_r", "cue_demeaned_r"):
        matrix = np.asarray(summary[key], float)
        out[key] = []
        for j, target in enumerate(TARGETS):
            scores = np.abs(matrix[:, 1:, j])
            if not np.isfinite(scores).any():
                out[key].append(None)
                continue
            c, f = np.unravel_index(np.nanargmax(scores), scores.shape)
            f += 1
            out[key].append({"target": target, "channel": int(c) + 1,
                             "feature": FEATURE_NAMES[f], "r": float(matrix[c, f, j])})
    return out


def subset_pair(pair, mask):
    return {k: v[mask] for k, v in pair.items()
            if isinstance(v, np.ndarray) and len(v) == len(mask)}


def describe_pair(pair):
    summary = correlation_summary(pair)
    summary["exploratory_spectral_maxima"] = exploratory_maxima(summary)
    summary["rounds"] = {str(r): correlation_summary(subset_pair(pair, pair["rounds"] == r))
                         for r in np.unique(pair["rounds"]) if r > 0}
    summary["windows_with_near_rail_samples"] = int((pair["clip_fraction"] > 0).sum())
    summary["unclipped_only_diagnostic"] = correlation_summary(
        subset_pair(pair, pair["clip_fraction"] == 0))
    summary["segment_coverage"] = pair["segment_coverage"]
    return summary


def save_figure(fig, output, name):
    import matplotlib.pyplot as plt
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_plots(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(result["windows"]["250"]["recordings"])
    primary = result["windows"]["250"]["recordings"]
    for j, target in enumerate(TARGETS):
        fig, axes = plt.subplots(2, len(names), figsize=(5 * len(names) + 3, 10),
                                 squeeze=False, constrained_layout=True)
        for col, name in enumerate(names):
            for row, key in enumerate(("pearson_r", "cue_demeaned_r")):
                ax = axes[row, col]
                values = np.asarray(primary[name][key], float)[:, :, j].T
                im = ax.imshow(values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
                ax.set_xticks(range(8), range(1, 9))
                ax.set_yticks(range(len(FEATURE_NAMES)), FEATURE_LABELS if col == 0 else [""] * len(FEATURE_NAMES))
                ax.set_xlabel("EMG channel")
                ax.set_title(name + (" · overall" if row == 0 else " · cue means removed"))
                ax.axhline(.5, color="#444", linewidth=1)
                ax.axhline(4.5, color="#444", linewidth=.7)
                ax.axhline(8.5, color="#444", linewidth=.7)
                for f in range(len(FEATURE_NAMES)):
                    for c in range(8):
                        if np.isfinite(values[f, c]):
                            ax.text(c, f, f"{values[f,c]:+.2f}", ha="center", va="center", fontsize=7,
                                    color="white" if abs(values[f,c]) > .6 else "black")
        fig.colorbar(im, ax=axes, label="Pearson r", shrink=.8)
        fig.suptitle(f"{TARGET_LABELS[j]} · spectral features versus matched-window RMS\n"
                     "250 ms trailing windows; fixed +50 ms labels; all pair searches exploratory", fontsize=14)
        save_figure(fig, output, f"correlations_{target}")

    colors = ("#2764a1", "#bf613a", "#24795f")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for j, selection in enumerate(result["source_selections"][:3]):
        for row, (key, ci) in enumerate((("pearson_r", "pearson_ci95"),
                                        ("cue_demeaned_r", "cue_demeaned_ci95"))):
            ax = axes[row, j]
            for n, name in enumerate(names):
                pair = primary[name]["selected_intervals"]["5"]["pairs"][j]
                low, high = pair[ci]
                ax.plot([low, high], [n, n], color=colors[n % len(colors)], linewidth=3)
                ax.scatter([pair[key]], [n], color=colors[n % len(colors)], s=55, zorder=3)
            ax.axvline(0, color="#777", linewidth=1)
            ax.set_xlim(-1, 1); ax.set_ylim(-.5, len(names) - .5)
            ax.set_yticks(range(len(names)), names)
            ax.grid(axis="x", alpha=.2)
            if row == 0:
                ax.set_title(f"{TARGET_LABELS[j]}\nEMG {selection['channel'] + 1}: "
                             f"{FEATURE_LABELS[selection['feature_index']]}", fontsize=10)
            ax.set_xlabel("Overall r" if row == 0 else "Cue-demeaned r")
    fig.suptitle("Do source-selected spectral relationships carry over?", fontsize=15)
    fig.text(.5, .02, "5 s paired block-bootstrap intervals are pointwise and descriptive. "
             "Source selection inflates source effects; pairs stay fixed in picking.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.09, right=.98, bottom=.12, top=.84, hspace=.26, wspace=.23)
    save_figure(fig, output, "source_pair_replication")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("blocks", type=Path, nargs="+", help="source recording first, then transfer recordings")
    parser.add_argument("--side", required=True, choices=("left", "right"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; choose a new directory")
    if len(args.blocks) < 2 or len({p.name for p in args.blocks}) != len(args.blocks):
        parser.error("at least two distinctly named recordings required")
    if args.replicates < 1:
        parser.error("positive bootstrap replicate count required")
    args.output.mkdir(parents=True)
    result = {"protocol": json.loads(PROTOCOL.read_text()), "protocol_sha256": file_hash(PROTOCOL),
              "side": args.side, "bootstrap_replicates": args.replicates,
              "feature_order": FEATURE_NAMES, "target_order": TARGETS,
              "array_axes": "channel (0-based), feature, target", "recordings": {}, "windows": {}}
    loaded = {}
    for i, path in enumerate(args.blocks):
        segments, metadata = load_segments(path, args.side, clock_mode="ols" if i == 0 else "unit")
        loaded[path.name] = segments
        result["recordings"][path.name] = {"quality": metadata,
                "source_sha256": {name: file_hash(path/name) for name in ("emg.jsonl", "arm.jsonl")}}
    selections = None
    for samples in (512, 256):
        window_key = str(round(samples / 2048 * 1000))
        records = {}
        for name, segments in loaded.items():
            pair = paired_spectral_segments(segments, window_samples=samples)
            summary = describe_pair(pair)
            if selections is None:
                selections = choose_source_pairs(summary)
                result["source_selections"] = selections
            summary["selected_intervals"] = selected_intervals(pair, selections, args.replicates)
            records[name] = summary
            # Local cache for review/replotting only; no sample-level data duplicated in reports.
            np.savez_compressed(args.output / f"{name}_{window_key}ms.npz",
                                **{k: v for k, v in pair.items() if isinstance(v, np.ndarray)})
            print(f"{name}: {window_key} ms, {len(pair['y'])} paired windows", flush=True)
        result["windows"][window_key] = {"samples": samples, "frequency_bin_spacing_hz": 2048 / samples,
                                            "recordings": records}
    result = nullsafe(result)
    (args.output / "correlations.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save_plots(result, args.output)
    print(args.output / "correlations.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
