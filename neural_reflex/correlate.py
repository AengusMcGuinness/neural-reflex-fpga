"""Correlation report for recorded arm blocks: is the data lined up, and what tracks what?

For every block (emg.jsonl + motion.jsonl + arm.jsonl) it measures:
  timing    camera delay vs the band's gyro, found by cross-correlating movement speed
  coverage  how many frames per protocol step show the band arm, and its angle range there
  EMG       each channel's activity (100 ms RMS) vs each arm signal
  motion    the band orientation axis that best follows each arm angle
Writes correlation.json and correlation.html in the session folder and prints a summary.
Grip strength is computed from EMG itself, so its EMG correlation is marked as circular.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .armfit import load_motion
from .handfit import emg_samples, read_jsonl
from .robot_map import DT, human_signals

SIGNALS = ("shoulder_lift", "elbow_flex", "wrist_flex", "shoulder_pan", "wrist_roll", "gripper")
CIRCULAR = {"gripper"}         # derived from EMG
MAX_LAG_S = 0.5
EDGE = 30                      # grid steps trimmed at both ends (filters and lag shifts)
VIEWER = Path(__file__).resolve().parent.parent / "web" / "correlate" / "viewer.html"


def zscore(x):
    x = np.asarray(x, dtype=float) - np.mean(x)
    return x / (np.std(x) + 1e-12)


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def best_lag(reference, delayed, max_steps):
    """Steps by which `delayed` trails `reference` (positive = later)."""
    ref, lagging = zscore(reference), zscore(delayed)
    scores = {k: float(np.mean(ref[EDGE:-EDGE] * np.roll(lagging, -k)[EDGE:-EDGE]))
              for k in range(-max_steps, max_steps + 1)}
    k = max(scores, key=scores.get)
    return k, scores[k], scores[0]


def shift(values, steps):
    return np.roll(np.asarray(values, float), -steps)[EDGE:-EDGE]


def block_report(block: Path) -> dict:
    signals = human_signals(block)
    rows = read_jsonl(block / "arm.jsonl")
    start = min(r["host_time_s"] for r in rows if r.get("world_landmarks"))
    grid = signals["time"] + start
    orient_t, orient_m, gyro_t, gyro_v = load_motion(read_jsonl(block / "motion.jsonl"))
    gyro = np.interp(grid, gyro_t, np.linalg.norm(gyro_v, axis=1))
    camera_speed = np.abs(np.gradient(signals["elbow_flex"])) + np.abs(np.gradient(signals["shoulder_lift"]))
    lag, at_best, at_zero = best_lag(gyro, camera_speed, int(MAX_LAG_S / DT))

    times, emg, _ = emg_samples(read_jsonl(block / "emg.jsonl"))
    window = int(0.1 / (times[1] - times[0]))
    envelopes = np.stack([np.interp(grid, times, np.convolve(emg[:, c] ** 2, np.ones(window) / window,
                                                             mode="same") ** 0.5) for c in range(8)], axis=1)
    orientation = np.stack([np.interp(grid, orient_t, orient_m[:, a]) for a in range(9)], axis=1)

    emg_corr, motion_corr = {}, {}
    for name in SIGNALS:
        target = shift(signals[name], lag)
        emg_corr[name] = [pearson(envelopes[EDGE:-EDGE, c], target) for c in range(8)]
        axes = [pearson(orientation[EDGE:-EDGE, a], target) for a in range(9)]
        usable = [(abs(v), i) for i, v in enumerate(axes) if v is not None]
        motion_corr[name] = {"best_abs_r": max(usable)[0] if usable else None,
                             "axis": max(usable)[1] if usable else None}

    coverage = []
    cues = [r.get("cue") for r in rows]
    for cue in dict.fromkeys(c for c in cues if c):
        step_rows = [r for r in rows if r.get("cue") == cue]
        mask = np.array([c == cue for c in signals["cue"]])
        seen = sum(1 for r in step_rows if (r.get("angles_by_side") or {}).get(signals["side"]))
        entry = {"step": cue, "frames": len(step_rows), "band_arm_visible": seen}
        for name in ("shoulder_lift", "elbow_flex"):
            if mask.any():
                values = np.asarray(signals[name])[mask]
                entry[name] = [float(np.percentile(values, 5)), float(np.percentile(values, 95))]
        coverage.append(entry)

    return {"block": block.name, "side": signals["side"], "seconds": float(signals["time"][-1]),
            "timing": {"camera_lag_ms": lag * DT * 1000, "match_at_lag": at_best, "match_at_zero": at_zero},
            "emg": emg_corr, "motion": motion_corr, "coverage": coverage,
            "pan_defined_pct": float(100 * np.mean(signals["pan_defined"]))}


def blocks_from(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if (p / "arm.jsonl").exists():
            out.append(p)
        elif (p / "blocks").is_dir():
            out += sorted(b for b in (p / "blocks").iterdir() if (b / "arm.jsonl").exists())
    if not out:
        raise SystemExit("no blocks with arm.jsonl found")
    return out


def summarize(reports: list[dict]) -> dict:
    """Average |r| per signal across blocks: the strongest EMG channel and the band tilt."""
    summary = {}
    for name in SIGNALS:
        per_channel = np.array([[abs(v) if v is not None else np.nan for v in r["emg"][name]] for r in reports])
        mean = np.nanmean(per_channel, axis=0)
        motion = [r["motion"][name]["best_abs_r"] for r in reports if r["motion"][name]["best_abs_r"] is not None]
        summary[name] = {"best_emg_channel": int(np.nanargmax(mean)) + 1 if np.isfinite(mean).any() else None,
                         "best_emg_abs_r": float(np.nanmax(mean)) if np.isfinite(mean).any() else None,
                         "band_tilt_abs_r": float(np.mean(motion)) if motion else None,
                         "circular": name in CIRCULAR}
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="block folders, or a session folder (uses all its blocks)")
    parser.add_argument("--output", help="where to write correlation.json/html (default: the session folder)")
    args = parser.parse_args(argv)
    blocks = blocks_from(args.paths)
    reports = [block_report(b) for b in blocks]
    summary = summarize(reports)
    for r in reports:
        t = r["timing"]
        visible = sum(c["band_arm_visible"] for c in r["coverage"])
        frames = sum(c["frames"] for c in r["coverage"])
        print(f"{r['block']}: {r['seconds']:.0f} s, band arm '{r['side']}' visible {100 * visible / max(frames, 1):.0f}%, "
              f"camera lag {t['camera_lag_ms']:+.0f} ms (match {t['match_at_zero']:.2f} -> {t['match_at_lag']:.2f})")
        missing = [c["step"] for c in r["coverage"] if c["frames"] and c["band_arm_visible"] / c["frames"] < 0.5]
        if missing:
            print(f"   arm mostly out of view during: {', '.join(missing)}")
    print(f"\n{'signal':15}{'best EMG ch':>12}{'|r|':>7}{'band tilt |r|':>15}")
    for name, s in summary.items():
        ch = "—" if s["best_emg_channel"] is None else f"ch{s['best_emg_channel']}"
        r_emg = "—" if s["best_emg_abs_r"] is None else f"{s['best_emg_abs_r']:.2f}"
        r_tilt = "—" if s["band_tilt_abs_r"] is None else f"{s['band_tilt_abs_r']:.2f}"
        print(f"{name:15}{ch:>12}{r_emg:>7}{r_tilt:>15}" + ("   (grip comes from EMG: circular)" if s["circular"] else ""))
    output = Path(args.output) if args.output else blocks[0].parent.parent
    data = {"blocks": reports, "summary": summary, "signals": list(SIGNALS)}
    (output / "correlation.json").write_text(json.dumps(data, indent=2) + "\n")
    page = VIEWER.read_text(encoding="utf-8").replace(
        "__CORRELATE_DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    (output / "correlation.html").write_text(page, encoding="utf-8")
    print(f"\nReport: {output / 'correlation.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
