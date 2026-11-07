"""Descriptive EMG–arm-angle correlations for one paired recording.

The block-shuffle comparison is conditional on exchangeable blocks. It is not
a population-level significance test for a single scripted, nonstationary run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .emg_benchmark import CONFIG, TARGETS, load_recording, pair_segment
from .recording_audit import KNOWN_CUES

CHANNELS = tuple(f"EMG {i}" for i in range(1, 9))
LABEL_DELAY_S = .05
SEED = 7
BLOCK_SECONDS = (2, 5, 10)


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eight-by-three Pearson coefficients, masking missing labels per joint."""
    result = np.full((8, 3), np.nan)
    counts = np.zeros((8, 3), dtype=int)
    for joint in range(3):
        keep = np.isfinite(y[:, joint]) & np.isfinite(x).all(axis=1)
        count = int(keep.sum())
        counts[:, joint] = count
        if count < 3:
            continue
        xc = x[keep] - x[keep].mean(axis=0)
        yc = y[keep, joint] - y[keep, joint].mean()
        denom = np.sqrt((xc * xc).sum(axis=0) * (yc * yc).sum())
        np.divide(xc.T @ yc, denom, out=result[:, joint], where=denom > 0)
    return result, counts


def _demean_by_cue(x: np.ndarray, y: np.ndarray, cues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove each cue's mean separately, retaining the original missing-label mask."""
    out_x, out_y = x.copy(), y.copy()
    for cue in np.unique(cues):
        at = cues == cue
        if at.sum() < 2:
            continue
        out_x[at] -= np.mean(x[at], axis=0)
        for joint in range(3):
            valid = at & np.isfinite(y[:, joint])
            if valid.any():
                out_y[valid, joint] -= np.mean(y[valid, joint])
    return out_x, out_y


def correlation_matrices(x: np.ndarray, y: np.ndarray, cues: np.ndarray):
    overall, counts = _pearson(x, y)
    within, within_counts = _pearson(*_demean_by_cue(x, y, cues))
    assert np.array_equal(counts, within_counts)
    return overall, within, counts


def _safe_matrix(matrix: np.ndarray):
    return [[round(float(value), 4) if np.isfinite(value) else None for value in row]
            for row in matrix]


def block_shuffle(x: np.ndarray, y: np.ndarray, cues: np.ndarray, times: np.ndarray,
                  seconds: int, permutations: int, seed: int = SEED) -> dict:
    """Permute contiguous, equal-length EMG blocks jointly across eight channels.

    The final partial block is omitted; each block is a fixed number of feature
    rows approximately equal to ``seconds`` of the median observed cadence.
    """
    if permutations < 1 or seconds <= 0:
        raise ValueError("permutations and seconds must be positive")
    dt = float(np.median(np.diff(times)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("paired times must have positive cadence")
    width = max(1, round(seconds / dt))
    n_blocks = len(x) // width
    if n_blocks < 4:
        raise ValueError("at least four complete blocks required")
    n = n_blocks * width
    x, y, cues = x[:n], y[:n], cues[:n]
    observed_overall, observed_within, counts = correlation_matrices(x, y, cues)
    exceeds_overall = np.zeros((8, 3), dtype=int)
    exceeds_within = np.zeros((8, 3), dtype=int)
    null_overall_max, null_within_max = [], []
    rng = np.random.default_rng(seed)
    blocks = x.reshape(n_blocks, width, 8)
    for _ in range(permutations):
        shuffled = blocks[rng.permutation(n_blocks)].reshape(n, 8)
        overall, within, _ = correlation_matrices(shuffled, y, cues)
        max_overall = float(np.nanmax(np.abs(overall)))
        max_within = float(np.nanmax(np.abs(within)))
        null_overall_max.append(max_overall)
        null_within_max.append(max_within)
        exceeds_overall += max_overall >= np.abs(observed_overall)
        exceeds_within += max_within >= np.abs(observed_within)
    p_overall = (exceeds_overall + 1) / (permutations + 1)
    p_within = (exceeds_within + 1) / (permutations + 1)
    p_overall[~np.isfinite(observed_overall)] = np.nan
    p_within[~np.isfinite(observed_within)] = np.nan
    return {
        "requested_block_seconds": seconds,
        "actual_block_seconds": round(width * dt, 3),
        "block_windows": width,
        "complete_blocks": n_blocks,
        "used_windows": n,
        "discarded_tail_windows": len(times) - n,
        "joint_pair_counts": counts[0].tolist(),
        "observed_overall_r": _safe_matrix(observed_overall),
        "observed_within_cue_r": _safe_matrix(observed_within),
        "familywise_p_overall": _safe_matrix(p_overall),
        "familywise_p_within_cue": _safe_matrix(p_within),
        "null_max_abs_overall_p95": round(float(np.percentile(null_overall_max, 95)), 4),
        "null_max_abs_within_cue_p95": round(float(np.percentile(null_within_max, 95)), 4),
    }


def analyze_pairing(pair: dict, permutations: int = 1000, block_seconds=BLOCK_SECONDS) -> dict:
    x = np.asarray(pair["x"][:, -1, 16:24], float)
    y = np.asarray(pair["y"], float)
    times = np.asarray(pair["times"], float)
    raw_cues = np.asarray(pair["cues"])
    cues = np.array([c if isinstance(c, str) and c in KNOWN_CUES else "other"
                     for c in raw_cues])
    if x.ndim != 2 or x.shape[1] != 8 or y.shape != (len(x), 3) or len(times) != len(x):
        raise ValueError("expected aligned eight-channel EMG and three angle labels")
    if len(x) < 100 or not np.isfinite(x).all() or not np.isfinite(times).all():
        raise ValueError("too few finite paired windows")
    overall, within, counts = correlation_matrices(x, y, cues)
    rankings = sorted(((abs(float(value)), i, j) for i, row in enumerate(overall)
                       for j, value in enumerate(row) if np.isfinite(value)), reverse=True)
    top = [{"channel": CHANNELS[i], "joint": TARGETS[j],
            "overall_r": round(float(overall[i, j]), 4),
            "within_cue_r": round(float(within[i, j]), 4) if np.isfinite(within[i, j]) else None,
            "paired_windows": int(counts[i, j])} for _, i, j in rankings[:8]]
    results = {str(seconds): block_shuffle(x, y, cues, times, seconds, permutations)
               for seconds in block_seconds}
    return {
        "method": {"feature": "causal bandpass/notch-filtered log1p RMS; 100 ms window, 20 ms hop",
                   "feature_config": {**CONFIG, "history": 1},
                   "alignment": "EMG window end + fixed 50 ms to interpolated camera angle",
                   "label_delay_seconds": LABEL_DELAY_S,
                   "correlation": "Pearson; within-cue subtracts each scripted cue mean",
                   "shuffle": "jointly permute contiguous equal-window EMG blocks; trim final partial block",
                   "permutations": permutations, "seed": SEED,
                   "primary_block_seconds": 5,
                   "sensitivity_block_seconds": [2, 10],
                   "multiplicity": "max absolute r across all 8 x 3 channel-joint pairs, separately for overall and within-cue",
                   "inference": "empirical p conditional on exchangeable blocks only; one scripted nonstationary session does not establish population significance",
                   "reference": "https://doi.org/10.1371/journal.pbio.3002758"},
        "channels": list(CHANNELS), "joints": list(TARGETS),
        "paired_windows": len(x), "pair_counts": counts.tolist(),
        "overall_r": _safe_matrix(overall), "within_cue_r": _safe_matrix(within),
        "top_pairs": top, "block_shuffle_sensitivity": results,
        "limits": ["Band wrist has not been physically confirmed for the camera arm.",
                   "Camera capture/exposure latency relative to Kinesis host receipt is unmeasured.",
                   "The 50 ms alignment is fixed; no lag search was performed.",
                   "Correlations do not establish joint decodability, control quality, or causation."],
    }


def _cue_runs(cues, times):
    starts = np.r_[0, np.flatnonzero(cues[1:] != cues[:-1]) + 1]
    ends = np.r_[starts[1:], len(cues)]
    for start, end in zip(starts, ends):
        yield times[start], times[end - 1], cues[start]


def save_plots(pair: dict, report: dict, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    x = np.asarray(pair["x"][:, -1, 16:24], float)
    y = np.asarray(pair["y"], float)
    times = np.asarray(pair["times"], float)
    cues = np.array([c if isinstance(c, str) and c in KNOWN_CUES else "other"
                     for c in pair["cues"]])
    rel_t = times - times[0]
    matrices = [np.array(report["overall_r"], float), np.array(report["within_cue_r"], float)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, matrix, title in zip(axes, matrices, ("Overall Pearson r", "Within-cue Pearson r")):
        im = ax.imshow(matrix.T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(8), CHANNELS, rotation=45, ha="right")
        ax.set_yticks(range(3), [s.replace("_", " ") for s in TARGETS])
        ax.set_title(title)
        for joint in range(3):
            for channel in range(8):
                value = matrix[channel, joint]
                if np.isfinite(value):
                    ax.text(channel, joint, f"{value:+.2f}", ha="center", va="center",
                            color="white" if abs(value) > .55 else "black", fontsize=8)
    fig.colorbar(im, ax=axes, label="Pearson r", shrink=.75)
    fig.suptitle("EMG envelope versus camera-derived arm angle • one session")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"correlation_heatmap.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True)
    for joint, name in enumerate(TARGETS):
        axes[0].plot(rel_t, y[:, joint], linewidth=.9, label=name.replace("_", " "))
    axes[0].set_ylabel("Camera angle (degrees)")
    axes[0].legend(loc="upper right", ncol=3)
    axes[0].grid(alpha=.2)
    for start, end, cue in _cue_runs(cues, rel_t):
        axes[0].axvspan(start, end, color="#777777", alpha=.04)
        if end - start >= 2:
            axes[0].text((start + end) / 2, 1.01, str(cue).replace("_", " "),
                         transform=axes[0].get_xaxis_transform(), ha="center", va="bottom",
                         fontsize=7, rotation=30)
    im = axes[1].imshow(x.T, origin="lower", aspect="auto", interpolation="nearest",
                        extent=[rel_t[0], rel_t[-1], .5, 8.5], cmap="viridis")
    axes[1].set_yticks(range(1, 9))
    axes[1].set_ylabel("EMG channel")
    axes[1].set_xlabel("Seconds since first paired window")
    fig.colorbar(im, ax=axes[1], label="log(1 + RMS)", shrink=.8)
    fig.suptitle("Approximately aligned EMG envelopes and joint-angle proxies • fixed 50 ms delay")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"recording_overview.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    blocks = report["block_shuffle_sensitivity"]
    all_heights = []
    for ax, mode, title in zip(axes, ("overall", "within_cue"),
                               ("Overall", "Within cue")):
        lengths = [2, 5, 10]
        observed = [np.nanmax(np.abs(np.array(blocks[str(s)][f"observed_{mode}_r"], float)))
                    for s in lengths]
        null95 = [blocks[str(s)][f"null_max_abs_{mode}_p95"] for s in lengths]
        all_heights.extend(observed + null95)
        positions = np.arange(3)
        width = .34
        first = ax.bar(positions - width / 2, observed, width, label="Observed max |r|",
                       color="#2764a1")
        second = ax.bar(positions + width / 2, null95, width,
                        label="Shuffle null 95th percentile", color="#d58b36")
        ax.bar_label(first, fmt="%.2f", padding=3, fontsize=8)
        ax.bar_label(second, fmt="%.2f", padding=3, fontsize=8)
        ax.set_xticks(positions, ["2 s sensitivity", "5 s primary", "10 s sensitivity"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, max(all_heights) * 1.2)
    axes[0].set_ylabel("Maximum absolute Pearson r across 24 pairs")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Observed correlations versus exploratory block-shuffle comparison", y=.97)
    fig.text(.5, .04, "Single scripted session; exchangeability is uncertain, so bars are not standalone significance proof.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.08, right=.98, top=.81, bottom=.21, wspace=.09)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"block_shuffle_comparison.{suffix}", dpi=180,
                    bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("block", type=Path)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.permutations < 1:
        parser.error("--permutations must be positive")
    rec = load_recording(args.block, args.side)
    lo = max(rec["times"][0], rec["pose_times"][0])
    hi = min(rec["times"][-1], rec["pose_times"][-1])
    config = {**CONFIG, "history": 1}
    pair = pair_segment(rec, (lo, hi), LABEL_DELAY_S, config)
    report = analyze_pairing(pair, args.permutations)
    report["clock"] = rec["clock"]
    report["side"] = args.side
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "correlations.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save_plots(pair, report, args.output)
    print(args.output / "correlations.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
