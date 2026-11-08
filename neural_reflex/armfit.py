"""Predict webcam arm angles from band EMG and band motion, and write an overlay page.

Steps for each recorded block (emg.jsonl, motion.jsonl, arm.jsonl):
1. Work out which arm wears the band: the arm whose forearm turns when the band's gyro
   says it turns. This survives a mirrored camera, which swaps MediaPipe's left and right.
2. Every 20 ms, build features from the last 200 ms of EMG (plus three earlier windows) and
   from the band's orientation and gyro, and pair them with that arm's angles.
3. Hold out every fifth 5-second chunk (with guard gaps), fit ridge regression on the rest,
   and score EMG-only, motion-only and combined inputs on the held-out chunks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .arm_pose import SIDES, side_angles
from .config import HOP_MS, LAGS_MS, SAMPLE_RATE_HZ, WINDOW_MS
from .handfit import emg_samples, r2, read_jsonl, smooth
from .ridge_decoder import fit_ridge, predict_angles
from .signal import window_features

TARGETS = ("shoulder_lift", "elbow_flex")     # the arm angles the robot reproduces (grip comes from EMG)
# Steps of the older "full" protocol that the robot can reproduce; robot-protocol rounds are all usable.
ROBOT_STEPS_FULL = {"lift_forward", "elbow", "home"}
REACHABLE_LIFT = (80.0, 185.0)     # straight up to just below shoulder height
MIN_WINDOWS = 1500                 # 30 s of paired data
LABEL_DELAY_S = 0.05        # camera frames arrive slightly after the movement
MAX_ARM_GAP_S = 0.15
CHUNK_S, GUARD_S = 5.0, 0.6
REGULARIZATION = (1.0, 10.0, 100.0, 1000.0, 10000.0)
VIEWER = Path(__file__).resolve().parent.parent / "web" / "armfit" / "viewer.html"


def quaternion_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])


def load_motion(events: list[dict]):
    orient = sorted((e["host_time_s"], e["payload"]["values"]) for e in events if e["type"] == "orientation")
    gyro = sorted((e["host_time_s"], e["payload"]["values"],
                   e["payload"].get("observed_degrees_per_second_per_count", 0.07))
                  for e in events if e["type"] == "gyro")
    if len(orient) < 10 or len(gyro) < 10:
        raise ValueError("motion.jsonl has too few orientation/gyro events; record with --motion-output")
    return (np.array([t for t, _ in orient]), np.array([quaternion_matrix(np.array(v, float)) for _, v in orient]),
            np.array([t for t, _, _ in gyro]), np.array([np.array(v, float) * k for _, v, k in gyro]))


def arm_tracks(rows: list[dict]):
    """Per side: frame times, target angles, and forearm direction (for matching the gyro)."""
    tracks = {}
    for side, index in SIDES.items():
        times, values, forearm = [], [], []
        for row in rows:
            if not row.get("world_landmarks"):
                continue
            angles = side_angles(row["world_landmarks"], row["visibility"], side)
            if angles is None:
                continue
            p = np.asarray(row["world_landmarks"])
            direction = p[index["wrist"]] - p[index["elbow"]]
            times.append(row["host_time_s"])
            values.append([angles[k] for k in TARGETS])
            forearm.append(direction / max(np.linalg.norm(direction), 1e-9))
        tracks[side] = (np.array(times), np.array(values).reshape(-1, len(TARGETS)), np.array(forearm))
    return tracks


def band_arm(tracks, gyro_t, gyro_v):
    """Correlate each arm's forearm turning speed with the band's gyro speed."""
    scores = {}
    for side, (t, _, forearm) in tracks.items():
        if len(t) < 30:
            scores[side] = float("nan")
            continue
        dt = np.diff(t)
        keep = (dt > 0) & (dt < 0.2)
        turn = np.degrees(np.arccos(np.clip((forearm[1:] * forearm[:-1]).sum(axis=1), -1, 1)))[keep] / dt[keep]
        mid = ((t[1:] + t[:-1]) / 2)[keep]
        speed = np.interp(mid, gyro_t, np.linalg.norm(gyro_v, axis=1))
        scores[side] = float(np.corrcoef(turn, speed)[0, 1]) if len(turn) > 20 else float("nan")
    usable = {s: v for s, v in scores.items() if not np.isnan(v)}
    if not usable:
        raise ValueError("neither arm was visible long enough to compare with the band")
    return max(usable, key=usable.get), scores


def interpolate(times, values, at, max_gap):
    after = int(np.searchsorted(times, at))
    if after == 0 or after == len(times) or times[after] - times[after - 1] > max_gap:
        return None
    w = (at - times[after - 1]) / (times[after] - times[after - 1])
    return (1 - w) * values[after - 1] + w * values[after]


def pair_block(block_dir: Path, side: str | None = None) -> dict:
    times, emg, valid = emg_samples(read_jsonl(block_dir / "emg.jsonl"))
    orient_t, orient_m, gyro_t, gyro_v = load_motion(read_jsonl(block_dir / "motion.jsonl"))
    rows = read_jsonl(block_dir / "arm.jsonl")
    tracks = arm_tracks(rows)
    chosen, scores = band_arm(tracks, gyro_t, gyro_v)
    if side:
        chosen = side
    arm_t, arm_v, _ = tracks[chosen]
    cues = sorted((r["host_time_s"], r.get("cue"), r.get("round")) for r in rows)
    cue_t = np.array([c[0] for c in cues])

    window = int(SAMPLE_RATE_HZ * WINDOW_MS / 1000)
    hop = int(SAMPLE_RATE_HZ * HOP_MS / 1000)
    lags = [int(ms // HOP_MS) for ms in LAGS_MS]
    ends = list(range(window - 1, len(emg), hop))
    emg_rows = [window_features(emg[e - window + 1:e + 1]) if valid[e - window + 1:e + 1].all() else None
                for e in ends]
    emg_x, motion_x, y, label_t, cue, rounds = [], [], [], [], [], []
    for position, end in enumerate(ends):
        if position < max(lags):
            continue
        parts = [emg_rows[position - lag] for lag in lags]
        if any(p is None for p in parts):
            continue
        t_end = float(times[end])
        target = interpolate(arm_t, arm_v, t_end + LABEL_DELAY_S, MAX_ARM_GAP_S)
        if target is None:
            continue
        o = int(np.clip(np.searchsorted(orient_t, t_end), 1, len(orient_t) - 1))
        if abs(orient_t[o] - t_end) > 0.1:
            continue
        in_window = (gyro_t > t_end - WINDOW_MS / 1000) & (gyro_t <= t_end)
        if not in_window.any():
            continue
        g = gyro_v[in_window]
        emg_x.append(np.concatenate(parts))
        motion_x.append(np.concatenate([orient_m[o], g.mean(axis=0), np.abs(g).mean(axis=0)]))
        y.append(target)
        label_t.append(t_end)
        current = cues[max(0, int(np.searchsorted(cue_t, t_end)) - 1)]
        cue.append(current[1])
        rounds.append(current[2])
    if len(y) < 100:
        raise ValueError(f"{block_dir.name}: only {len(y)} windows overlap a visible {chosen} arm")
    label_t = np.array(label_t)
    return {"name": block_dir.name, "side": chosen, "side_scores": scores,
            "emg": np.array(emg_x), "motion": np.array(motion_x), "targets": np.array(y),
            "times": label_t, "cues": cue, "rounds": rounds, "arm_frames": {s: int(len(v[0])) for s, v in tracks.items()}}


def chunk_split(times: np.ndarray):
    """Every fifth 5 s chunk is test; drop windows near a train/test boundary."""
    rel = times - times[0]
    chunk = (rel // CHUNK_S).astype(int)
    test = chunk % 5 == 2
    inner_val = chunk % 5 == 4
    near = np.zeros(len(times), dtype=bool)
    for boundary in np.flatnonzero(np.diff(test.astype(int)) != 0):
        near |= np.abs(rel - rel[boundary]) < GUARD_S
    return ~test & ~near, test & ~near, inner_val & ~test & ~near


def fit_inputs(features, targets, train, test, inner_val):
    inner = train & ~inner_val
    best = max(REGULARIZATION, key=lambda reg: np.nan_to_num(np.nanmean(
        r2(targets[inner_val], predict_angles(fit_ridge(features, targets, inner, reg), features[inner_val]))),
        nan=-np.inf))
    model = fit_ridge(features, targets, train, best)
    predicted = predict_angles(model, features)
    scores = r2(targets[test], predicted[test])
    mae = np.abs(targets[test] - predicted[test]).mean(axis=0)
    guess = np.abs(targets[test] - targets[train].mean(axis=0)).mean(axis=0)
    return {"model": model, "predicted": predicted, "regularization": best,
            "scores": {k: {"r2": None if np.isnan(scores[i]) else float(scores[i]), "mae_deg": float(mae[i]),
                           "mae_if_guessing_average_deg": float(guess[i])} for i, k in enumerate(TARGETS)}}


def robot_moves(block: dict) -> np.ndarray:
    """Windows from robot-reproducible steps, with the arm where the robot can reach."""
    cues = np.array(block["cues"], dtype=object)
    has_rounds = any(r is not None for r in block["rounds"])
    steps = np.array([r is not None for r in block["rounds"]]) if has_rounds else \
        np.isin(cues, list(ROBOT_STEPS_FULL))
    lift = block["targets"][:, TARGETS.index("shoulder_lift")]
    return steps & (lift >= REACHABLE_LIFT[0]) & (lift <= REACHABLE_LIFT[1])


def select(block: dict, mask: np.ndarray) -> dict:
    out = dict(block)
    for key in ("emg", "motion", "targets", "times"):
        out[key] = block[key][mask]
    for key in ("cues", "rounds"):
        out[key] = [v for v, keep in zip(block[key], mask) if keep]
    return out


def fit_rounds(features, targets, groups):
    """Leave one round out: every window is predicted by a model that never saw its round."""
    predicted = np.full_like(targets, np.nan, dtype=float)
    regs = []
    labels = sorted(set(groups))
    for held in labels:
        test = groups == held
        others = [g for g in labels if g != held]
        inner_val = groups == others[-1]
        train = ~test
        inner = train & ~inner_val
        reg = max(REGULARIZATION, key=lambda r_: np.nan_to_num(np.nanmean(
            r2(targets[inner_val], predict_angles(fit_ridge(features, targets, inner, r_), features[inner_val]))),
            nan=-np.inf))
        regs.append(reg)
        predicted[test] = predict_angles(fit_ridge(features, targets, train, reg), features[test])
    scores = r2(targets, predicted)
    mae = np.abs(targets - predicted).mean(axis=0)
    guess = np.mean([np.abs(targets[groups == g] - targets[groups != g].mean(axis=0)).mean(axis=0)
                     for g in labels], axis=0)
    model = fit_ridge(features, targets, np.ones(len(targets), dtype=bool), float(np.median(regs)))
    return {"model": model, "predicted": predicted, "regularization": float(np.median(regs)),
            "scores": {k: {"r2": None if np.isnan(scores[i]) else float(scores[i]), "mae_deg": float(mae[i]),
                           "mae_if_guessing_average_deg": float(guess[i])} for i, k in enumerate(TARGETS)}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("blocks", nargs="+", help="block folders with emg.jsonl, motion.jsonl, arm.jsonl")
    parser.add_argument("--side", choices=("left", "right"),
                        help="force which MediaPipe arm wears the band (default: match it to the gyro)")
    parser.add_argument("--all-movements", action="store_true",
                        help="use every window, not only robot-reproducible moves within the robot's reach")
    parser.add_argument("--output", help="folder for overlay and metrics (default: the session folder)")
    args = parser.parse_args(argv)
    blocks = []
    for path in args.blocks:
        b = pair_block(Path(path), args.side)
        s = b["side_scores"]
        keep = np.ones(len(b["targets"]), dtype=bool) if args.all_movements else robot_moves(b)
        print(f"{b['name']}: band arm = MediaPipe '{b['side']}' (gyro match: left {s['left']:.2f}, right {s['right']:.2f}); "
              f"{len(b['targets'])} paired windows, {int(keep.sum())} used "
              f"({'all movements' if args.all_movements else 'robot-reproducible moves within reach'})")
        blocks.append(select(b, keep))
    targets = np.concatenate([b["targets"] for b in blocks])
    if len(targets) < MIN_WINDOWS:
        raise SystemExit(f"Only {len(targets) * HOP_MS / 1000:.0f} s of usable movement; at least "
                         f"{MIN_WINDOWS * HOP_MS / 1000:.0f} s is needed to train and score. Record with the robot "
                         "protocol (record-arm --protocol robot, the default), or pass --all-movements.")
    emg = np.concatenate([b["emg"] for b in blocks])
    motion = np.concatenate([b["motion"] for b in blocks])
    inputs = {"EMG only": emg, "band motion only": motion, "EMG + band motion": np.hstack([emg, motion])}
    groups = np.array([f"{b['name']}#{r}" for b in blocks for r in b["rounds"]], dtype=object)
    by_round = all(r is not None for b in blocks for r in b["rounds"]) and len(set(groups)) >= 2
    if by_round:
        results = {name: fit_rounds(x, targets, groups) for name, x in inputs.items()}
        test = np.ones(len(targets), dtype=bool)
        print(f"{len(set(groups))} rounds: each scored by a model trained on the other rounds ({len(targets)} windows)")
    else:
        splits = [chunk_split(b["times"]) for b in blocks]
        train, test, inner_val = (np.concatenate([s[i] for s in splits]) for i in range(3))
        results = {name: fit_inputs(x, targets, train, test, inner_val) for name, x in inputs.items()}
        print(f"no repeated rounds: held-out 5 s chunks, train {int(train.sum())} / held-out {int(test.sum())} windows")
    print(f"{'R² on held-out':22}" + "".join(f"{k:>16}" for k in TARGETS))
    for name, res in results.items():
        print(f"{name:22}" + "".join(f"{('—' if res['scores'][k]['r2'] is None else format(res['scores'][k]['r2'], '.2f')):>16}"
                                     for k in TARGETS))
    best = results["EMG + band motion"]
    print(f"{'error ° (combined)':22}" + "".join(f"{best['scores'][k]['mae_deg']:>16.1f}" for k in TARGETS))
    print(f"{'error ° (guessing)':22}" + "".join(f"{best['scores'][k]['mae_if_guessing_average_deg']:>16.1f}" for k in TARGETS))

    output = Path(args.output) if args.output else Path(args.blocks[0]).resolve().parent.parent
    frames, offset = [], 0
    for block in blocks:
        n = len(block["targets"])
        mask = test[offset:offset + n]
        pred = smooth(best["predicted"][offset:offset + n][mask])
        t0 = block["times"][mask][0] if mask.any() else 0
        for t, a, p, c in zip(block["times"][mask], block["targets"][mask], pred,
                              np.array(block["cues"], dtype=object)[mask]):
            frames.append({"t": round(float(t - t0), 2), "cue": c, "actual": [round(float(v), 1) for v in a],
                           "predicted": [round(float(v), 1) for v in p]})
        offset += n
    metrics = {name: {"regularization": r["regularization"], **r["scores"]} for name, r in results.items()}
    metrics["evaluation"] = "leave one round out" if by_round else "held-out 5 s chunks"
    metrics["windows"] = "all movements" if args.all_movements else "robot-reproducible moves within reach"
    data = {"targets": list(TARGETS), "hop_ms": HOP_MS, "frames": frames,
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, dict)},
            "blocks": [{"name": b["name"], "side": b["side"], "side_scores": b["side_scores"]} for b in blocks]}
    (output / "arm_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    page = VIEWER.read_text(encoding="utf-8").replace(
        "__ARMFIT_DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    (output / "arm_overlay.html").write_text(page, encoding="utf-8")
    print(f"Overlay: {output / 'arm_overlay.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
