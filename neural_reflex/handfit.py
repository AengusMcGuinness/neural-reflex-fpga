"""Pair a recorded block's EMG with its hand pose, fit a decoder, and write an overlay page.

For each 20 ms step the decoder sees the last 200 ms of conditioned EMG (plus three
earlier windows) and predicts both the 14 joint angles and all 21 hand landmarks.
The last part of every block is held out: the overlay page replays only that part,
so what you see is prediction on movement the model never trained on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .align import expand_batch, fit_band_clock, mark_sequence_gaps
from .config import (HOP_MS, JOINT_NAMES, LABEL_DELAY_S, LAGS_MS, MAX_POSE_GAP_S,
                     MIN_CONFIDENCE, SAMPLE_RATE_HZ, WINDOW_MS)
from .ridge_decoder import fit_ridge, predict_angles
from .signal import condition_emg, window_features

# Palm direction depends on where the camera is, not on muscle activity.
SCORED_ANGLES = tuple(name for name in JOINT_NAMES if not name.startswith("palm_"))
N_LANDMARKS = 21
REGULARIZATION = (1.0, 10.0, 100.0, 1000.0, 10000.0)
VIEWER = Path(__file__).resolve().parent.parent / "web" / "handfit" / "viewer.html"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def longest_connection(records: list[dict]) -> list[dict]:
    """The longest unbroken run of batches from one band connection (band clocks differ across
    reconnects, so runs are never mixed)."""
    runs, current = [], []
    for record in records:
        if current and record.get("band_connection_id") != current[-1].get("band_connection_id"):
            runs.append(current)
            current = []
        current.append(record)
    runs.append(current)
    return max(runs, key=len)


def emg_samples(records: list[dict], keep_longest: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-sample Mac time, conditioned EMG and validity for one block. If the band
    reconnected, raise, or with keep_longest use only the longest single-connection run."""
    if len(records) < 2:
        raise ValueError("block has fewer than two EMG batches")
    connections = {record.get("band_connection_id") for record in records}
    if len(connections) > 1:
        if not keep_longest:
            raise ValueError("band reconnected during this block; record it again")
        records = longest_connection(records)
    records = mark_sequence_gaps(records)
    slope, intercept = fit_band_clock([r["band_timestamp_us"] for r in records],
                                      [r["host_time_s"] for r in records])
    samples = [s for record in records for s in expand_batch(record, slope, intercept)]
    times = np.array([s["host_time_s"] for s in samples])
    valid = np.array([s["valid"] for s in samples])
    raw = np.array([s["emg"] for s in samples], dtype=np.float64)
    return times, condition_emg(raw, SAMPLE_RATE_HZ, valid), valid


def pose_targets(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Times and [14 angles + 63 landmark coordinates] for frames with a tracked hand."""
    tracked = [r for r in rows if r.get("angles") and r.get("world_landmarks")
               and r["confidence"] >= MIN_CONFIDENCE]
    tracked.sort(key=lambda r: r["host_time_s"])
    times = np.array([r["host_time_s"] for r in tracked])
    values = np.array([[r["angles"][name] for name in JOINT_NAMES] +
                       list(np.asarray(r["world_landmarks"], dtype=np.float64).ravel())
                       for r in tracked]).reshape(len(tracked), len(JOINT_NAMES) + 3 * N_LANDMARKS)
    return times, values


def interpolate(times: np.ndarray, values: np.ndarray, at: float) -> np.ndarray | None:
    after = int(np.searchsorted(times, at))
    if after == 0 or after == len(times):
        return None
    before = after - 1
    span = times[after] - times[before]
    if span <= 0 or span > MAX_POSE_GAP_S:
        return None
    weight = (at - times[before]) / span
    return (1 - weight) * values[before] + weight * values[after]


def pair_block(block_dir: Path) -> dict:
    times, emg, valid = emg_samples(read_jsonl(block_dir / "emg.jsonl"))
    pose_times, pose_values = pose_targets(read_jsonl(block_dir / "pose.jsonl"))
    if len(pose_times) < 20:
        raise ValueError(f"{block_dir.name}: only {len(pose_times)} frames with a tracked hand")
    window = int(SAMPLE_RATE_HZ * WINDOW_MS / 1000)
    hop = int(SAMPLE_RATE_HZ * HOP_MS / 1000)
    lags = [int(ms // HOP_MS) for ms in LAGS_MS]
    ends = list(range(window - 1, len(emg), hop))
    feats = [window_features(emg[e - window + 1:e + 1]) if valid[e - window + 1:e + 1].all()
             else None for e in ends]
    features, targets, label_times = [], [], []
    for position, end in enumerate(ends):
        if position < max(lags):
            continue
        parts = [feats[position - lag] for lag in lags]
        if any(part is None for part in parts):
            continue
        label_time = float(times[end] + LABEL_DELAY_S)
        target = interpolate(pose_times, pose_values, label_time)
        if target is None:
            continue
        features.append(np.concatenate(parts))
        targets.append(target)
        label_times.append(label_time)
    if len(features) < 50:
        raise ValueError(f"{block_dir.name}: only {len(features)} windows overlap a tracked hand. "
                         "Keep the hand in view while EMG records.")
    overlap = (max(times[0], pose_times[0]), min(times[-1], pose_times[-1]))
    return {"name": block_dir.name, "features": np.array(features), "targets": np.array(targets),
            "times": np.array(label_times), "overlap_s": float(overlap[1] - overlap[0]),
            "pose_frames": int(len(pose_times))}


def chronological_split(n: int, test_fraction: float, gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Train on the start, test on the end, with a gap so overlapping windows never leak."""
    cut = int(n * (1 - test_fraction))
    train = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)
    train[:max(0, cut - gap)] = True
    test[cut:] = True
    return train, test


def r2(actual: np.ndarray, predicted: np.ndarray, min_variance: float = 1.0) -> np.ndarray:
    """R² per column. A column whose variance is below min_variance (in its own units squared;
    1.0 suits angles in degrees) never really moved, so it gets NaN rather than a misleading score."""
    residual = ((actual - predicted) ** 2).sum(axis=0)
    total = ((actual - actual.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total / len(actual) < min_variance, np.nan, 1 - residual / total)


def smooth(values: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    out = np.array(values, dtype=np.float64)
    for i in range(1, len(out)):
        out[i] = alpha * out[i] + (1 - alpha) * out[i - 1]
    return out


def fit(blocks: list[dict], test_fraction: float = 0.2) -> dict:
    gap = int(1000 * (WINDOW_MS + max(LAGS_MS)) / HOP_MS / 1000) + 1
    train_parts, test_parts = [], []
    for block in blocks:
        train, test = chronological_split(len(block["features"]), test_fraction, gap)
        block["train"], block["test"] = train, test
        train_parts.append(train)
        test_parts.append(test)
    features = np.concatenate([b["features"] for b in blocks])
    targets = np.concatenate([b["targets"] for b in blocks])
    train = np.concatenate(train_parts)
    test = np.concatenate(test_parts)
    # Choose regularization on the last 20% of the training windows.
    train_idx = np.flatnonzero(train)
    inner = np.zeros_like(train)
    inner[train_idx[:int(len(train_idx) * 0.8)]] = True
    holdout = train & ~inner
    angle_cols = [JOINT_NAMES.index(name) for name in SCORED_ANGLES]
    best = max(REGULARIZATION, key=lambda reg: np.nan_to_num(np.nanmean(r2(
        targets[holdout][:, angle_cols],
        predict_angles(fit_ridge(features, targets, inner, reg), features[holdout])[:, angle_cols])),
        nan=-np.inf))
    model = fit_ridge(features, targets, train, best)
    predicted = predict_angles(model, features)
    scores = r2(targets[test], predicted[test])
    mae = np.abs(targets[test] - predicted[test]).mean(axis=0)
    always_mean = np.abs(targets[test] - targets[train].mean(axis=0)).mean(axis=0)
    return {"model": model, "predicted": predicted, "regularization": best,
            "angles": {name: {"r2": None if np.isnan(scores[i]) else float(scores[i]),
                              "mae_deg": float(mae[i]),
                              "mae_if_guessing_average_deg": float(always_mean[i])}
                       for i, name in enumerate(JOINT_NAMES)},
            "train_windows": int(train.sum()), "test_windows": int(test.sum())}


def overlay_data(blocks: list[dict], result: dict) -> dict:
    frames, offset = [], 0
    n_angles = len(JOINT_NAMES)
    for block in blocks:
        n = len(block["features"])
        predicted = result["predicted"][offset:offset + n]
        offset += n
        test = block["test"]
        predicted_test = smooth(predicted[test])
        actual = block["targets"][test]
        start = block["times"][test][0]
        for t, a, p in zip(block["times"][test], actual, predicted_test):
            frames.append({"block": block["name"], "t": round(float(t - start), 3),
                           "angles": [round(float(v), 1) for v in a[:n_angles]],
                           "predicted_angles": [round(float(v), 1) for v in p[:n_angles]],
                           "landmarks": [round(float(v), 4) for v in a[n_angles:]],
                           "predicted_landmarks": [round(float(v), 4) for v in p[n_angles:]]})
    return {"joint_names": list(JOINT_NAMES), "scored": list(SCORED_ANGLES),
            "hop_ms": HOP_MS, "frames": frames,
            "metrics": {k: v for k, v in result.items() if k in ("angles", "regularization",
                                                                  "train_windows", "test_windows")},
            "blocks": [{"name": b["name"], "overlap_s": b["overlap_s"],
                        "pose_frames": b["pose_frames"], "windows": int(len(b["features"]))}
                       for b in blocks]}


def write_viewer(data: dict, output: Path) -> None:
    page = VIEWER.read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    output.write_text(page.replace("__HANDFIT_DATA__", payload), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("blocks", nargs="+", help="block folders containing emg.jsonl and pose.jsonl")
    parser.add_argument("--output", help="folder for overlay.html, model.json and metrics.json "
                                         "(default: the first block's session folder)")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)
    if not 0.05 <= args.test_fraction <= 0.5:
        parser.error("--test-fraction must be 0.05..0.5")
    blocks = [pair_block(Path(path)) for path in args.blocks]
    result = fit(blocks, args.test_fraction)
    output = Path(args.output) if args.output else Path(args.blocks[0]).resolve().parent.parent
    output.mkdir(parents=True, exist_ok=True)
    data = overlay_data(blocks, result)
    (output / "model.json").write_text(json.dumps({**result["model"], "targets": list(JOINT_NAMES) +
                                                   [f"lm{i}_{a}" for i in range(N_LANDMARKS)
                                                    for a in "xyz"]}) + "\n")
    (output / "metrics.json").write_text(json.dumps(data["metrics"], indent=2) + "\n")
    write_viewer(data, output / "overlay.html")
    for block in data["blocks"]:
        print(f"{block['name']}: {block['overlap_s']:.0f} s overlap, {block['pose_frames']} hand frames, "
              f"{block['windows']} paired windows")
    print(f"train {result['train_windows']} / held-out {result['test_windows']} windows, "
          f"regularization {result['regularization']:g}")
    print(f"{'angle':18} {'R²':>6} {'error °':>8} {'guessing °':>10}")
    for name in SCORED_ANGLES:
        a = result["angles"][name]
        score = "   —  " if a["r2"] is None else f"{a['r2']:6.2f}"
        print(f"{name:18} {score} {a['mae_deg']:8.1f} {a['mae_if_guessing_average_deg']:10.1f}"
              + ("   (didn't move)" if a["r2"] is None else ""))
    print(f"Overlay: {output / 'overlay.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
