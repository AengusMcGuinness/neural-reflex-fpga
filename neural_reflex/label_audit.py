"""Audit instructed pick targets against same-frame camera measurements.

Targets are synthetic instructions. The camera's ``followed`` values are noisy,
partly missing pose measurements, not ground truth of motor commands or EMG.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .emg_benchmark import label_values

ANGLES = ("shoulder_lift", "elbow_flex")
HAND = ("forward", "up")
FIELDS = ANGLES + HAND
PHASES = ("ready", "sync", "rest", "reach", "lower", "grip", "hold", "lift",
          "place", "open", "return", "rest_end")
GAP_S = .150


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _summary(values):
    values = np.asarray(values, float)
    if not len(values):
        return None
    p = np.percentile(values, [5, 50, 95])
    return {"p05": round(float(p[0]), 3), "median": round(float(p[1]), 3),
            "p95": round(float(p[2]), 3)}


def _stats(target, measured):
    target, measured = np.asarray(target, float), np.asarray(measured, float)
    valid = np.isfinite(target) & np.isfinite(measured)
    a, b = target[valid], measured[valid]
    if not len(a):
        return {"matched_frames": 0, "mae": None, "bias_measured_minus_target": None,
                "pearson_r": None, "error_percentiles": None,
                "target_percentiles_matched": None, "measured_percentiles_matched": None}
    error = b - a
    corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 1e-10 and np.std(b) > 1e-10 else None
    return {"matched_frames": int(len(a)), "mae": round(float(np.mean(np.abs(error))), 3),
            "bias_measured_minus_target": round(float(np.mean(error)), 3),
            "pearson_r": round(corr, 4) if corr is not None and np.isfinite(corr) else None,
            "error_percentiles": _summary(error), "target_percentiles_matched": _summary(a),
            "measured_percentiles_matched": _summary(b)}


def _read(block: Path):
    path = Path(block) / "arm.jsonl"
    if not path.is_file():
        raise ValueError("arm.jsonl is missing")
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"arm.jsonl:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"arm.jsonl:{line_number}: expected object")
            yield row


def _arrays(rows, side):
    times, rounds, phases = [], [], []
    target = {field: [] for field in FIELDS}
    followed = {field: [] for field in FIELDS}
    benchmark = {field: [] for field in ("shoulder_lift", "elbow_flex", "wrist_flex")}
    landmark_present = []
    for row in rows:
        t = row.get("host_time_s")
        if not _finite_number(t):
            raise ValueError("invalid arm timestamp")
        times.append(float(t))
        rnd = row.get("round")
        rounds.append(rnd if rnd in (1, 2) and type(rnd) is int else None)
        phase = row.get("cue")
        phases.append(phase if isinstance(phase, str) and phase in PHASES else "other")
        instructed = row.get("target")
        if not isinstance(instructed, dict):
            raise ValueError("target is missing or invalid")
        measured = row.get("followed")
        if measured is not None and not isinstance(measured, dict):
            raise ValueError("followed must be an object or null")
        for field in FIELDS:
            value = instructed.get(field)
            target[field].append(float(value) if _finite_number(value) else np.nan)
            value = measured.get(field) if measured is not None else None
            followed[field].append(float(value) if _finite_number(value) else np.nan)
        has_landmarks = row.get("world_landmarks") is not None and row.get("visibility") is not None
        landmark_present.append(has_landmarks)
        vals = label_values(row, side) if has_landmarks else np.full(3, np.nan)
        for field, value in zip(benchmark, vals):
            benchmark[field].append(float(value))
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("need at least two strictly increasing camera timestamps")
    return {"times": np.array(times), "rounds": np.array(rounds, dtype=object),
            "phases": np.array(phases), "target": {k: np.array(v) for k, v in target.items()},
            "followed": {k: np.array(v) for k, v in followed.items()},
            "benchmark": {k: np.array(v) for k, v in benchmark.items()},
            "landmark_present": np.array(landmark_present, bool)}


def audit_labels(block: Path, side: str = "right") -> tuple[dict, dict]:
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    rows = list(_read(Path(block)))
    data = _arrays(rows, side)
    t, rounds, phases = data["times"], data["rounds"], data["phases"]
    n = len(t)
    all_indices = np.ones(n, bool)

    def compare(mask):
        return {field: _stats(data["target"][field][mask], data["followed"][field][mask])
                for field in FIELDS}

    round_reports = {}
    for rnd in (1, 2):
        mask = rounds == rnd
        frames = int(mask.sum())
        round_reports[str(rnd)] = {
            "frames": frames,
            "followed_frames": int(np.isfinite(data["followed"][ANGLES[0]][mask]).sum()),
            "followed_fraction": round(float(np.isfinite(data["followed"][ANGLES[0]][mask]).mean()), 4)
                                 if frames else 0.0,
            "comparison": compare(mask),
        }
    phase_reports = {}
    for phase in PHASES:
        mask = phases == phase
        if mask.any():
            phase_reports[phase] = {"frames": int(mask.sum()),
                                    "followed_frames": int(np.isfinite(data["followed"][ANGLES[0]][mask]).sum())}
    report = {
        "method": {"comparison": "target and followed from the same camera-frame record; no lag optimization",
                   "timestamp": "host monotonic timestamp recorded after camera frame read",
                   "target_semantics": "synthetic on-screen instruction, not observed human pose",
                   "followed_semantics": "camera side-view signed shoulder lift; elbow uses shoulder-to-elbow against elbow-to-index/pinky midpoint",
                   "benchmark_semantics": "original frozen model label_values: unsigned 3D torso-relative shoulder lift, elbow uses elbow-to-wrist, wrist uses wrist-to-hand bend",
                   "hand_position_units": "forward/up divided by observed arm reach for followed; target uses model arm-length units; approximate dimensionless comparison",
                   "gap_break_seconds": GAP_S},
        "side": side,
        "camera_frames": n,
        "duration_s": round(float(t[-1] - t[0]), 3),
        "camera_frame_rate_hz": round(float((n - 1) / (t[-1] - t[0])), 3),
        "camera_gaps_over_150ms": int((np.diff(t) > GAP_S).sum()),
        "landmark_frames": int(data["landmark_present"].sum()),
        "followed_frames": int(np.isfinite(data["followed"][ANGLES[0]]).sum()),
        "followed_fraction": round(float(np.isfinite(data["followed"][ANGLES[0]]).mean()), 4),
        "benchmark_label_validity": {field: {"valid_frames": int(np.isfinite(vals).sum()),
                                              "fraction": round(float(np.isfinite(vals).mean()), 4)}
                                     for field, vals in data["benchmark"].items()},
        "target_distributions": {field: _summary(data["target"][field]) for field in FIELDS},
        "followed_distributions": {field: _summary(data["followed"][field][np.isfinite(data["followed"][field])])
                                   for field in FIELDS},
        "comparison": compare(all_indices),
        "rounds": round_reports,
        "phases": phase_reports,
        "limitations": ["Target angles are commands to imitate, not measured limb positions.",
                        "No camera exposure latency or human response lag was measured.",
                        "Signed side-view shoulder angles cannot be compared directly with the frozen model's unsigned torso-relative angles.",
                        "The measured elbow uses the hand midpoint while the frozen model uses the wrist; their magnitudes differ.",
                        "Camera gaps and occlusion reduce observed follow-through coverage."]
    }
    return report, data


def _plot_curve(ax, times, values, color, label):
    values = np.asarray(values, float)
    gap_after = np.flatnonzero(np.diff(times) > GAP_S)
    if len(gap_after):
        tt, vv = [], []
        gap_set = set(gap_after.tolist())
        for i, (t, value) in enumerate(zip(times, values)):
            tt.append(t)
            vv.append(value)
            if i in gap_set:
                tt.append((times[i] + times[i + 1]) / 2)
                vv.append(np.nan)
        times, values = np.asarray(tt), np.asarray(vv)
    ax.plot(times, values, color=color, linewidth=1.1, label=label)


def save_plots(report: dict, data: dict, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    output.mkdir(parents=True, exist_ok=True)
    times = data["times"] - data["times"][0]
    phases, rounds = data["phases"], data["rounds"]
    phase_colors = {name: i for i, name in enumerate(PHASES)}
    ids = np.array([phase_colors.get(name, len(PHASES)) for name in phases])
    fig, axes = plt.subplots(3, 1, figsize=(15, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [.25, 2, 2]}, constrained_layout=True)
    color_list = ["#9aa5ac", "#deaa73", "#a0b8a9", "#8db9d6", "#b0a6da", "#dca2aa",
                  "#a3a3a3", "#87c1aa", "#d5b486", "#c4a8d9", "#87a8cf", "#a0b8a9", "#bbbbbb"]
    axes[0].imshow(ids[None, :], aspect="auto", interpolation="nearest", cmap=ListedColormap(color_list),
                   vmin=-.5, vmax=len(color_list)-.5, extent=[times[0], times[-1], 0, 1])
    axes[0].set_yticks([])
    axes[0].set_ylabel("Phase")
    changes = np.r_[0, np.flatnonzero(phases[1:] != phases[:-1]) + 1, len(phases)]
    for a, b in zip(changes[:-1], changes[1:]):
        if times[b-1] - times[a] >= 3:
            axes[0].text((times[a] + times[b-1])/2, .5, phases[a].replace("_", " "),
                         ha="center", va="center", fontsize=7, rotation=45)
    for rnd in (1, 2):
        mask = rounds == rnd
        if mask.any():
            lo, hi = times[mask][[0, -1]]
            for ax in axes[1:]:
                ax.axvspan(lo, hi, color="#d4e7f0" if rnd == 1 else "#eee5d5", alpha=.12)
            axes[0].text((lo + hi)/2, 1.12, f"Round {rnd}", ha="center", va="bottom",
                         fontsize=10, transform=axes[0].get_xaxis_transform())
    for ax, field, title in zip(axes[1:], ANGLES,
                                ("Signed shoulder lift in guided side view", "Elbow flex using the hand midpoint")):
        _plot_curve(ax, times, data["target"][field], "#1f5d99", "On-screen target")
        _plot_curve(ax, times, data["followed"][field], "#d77f32", "Camera followed")
        ax.set_ylabel("Degrees")
        ax.set_title(title)
        ax.grid(alpha=.2)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Seconds since first camera frame")
    fig.suptitle("Pick practice: instructed target versus observed arm pose")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"target_vs_followed.{ext}", dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("block", type=Path)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report, data = audit_labels(args.block, args.side)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "labels.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save_plots(report, data, args.output)
    print(args.output / "labels.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
