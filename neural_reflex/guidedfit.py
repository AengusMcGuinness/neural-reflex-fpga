"""Learn band -> movement from guided (follow-the-target) recordings, and drive the simulated robot.

1. Follow check: how far behind the on-screen target you moved (cross-correlation of the target
   with what the camera saw), and how close you got once that delay is removed.
2. Decoder: every 20 ms, EMG + band-motion features (same as armfit) are labelled with the
   on-screen target shown `delay` earlier, i.e. what you were actually doing. Ridge models predict
   hand forward/up (arm lengths) and grip. Each round is scored by a model that never saw it.
3. Robot: held-out predictions go through hand_map (calibrated to the target's table and spots)
   and pick_sim's inverse kinematics, and are compared with where the true target would put the
   gripper. Writes guided_metrics.json and guided_preview.html in the block folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import guided
from .armfit import load_motion
from .config import HOP_MS, LAGS_MS, SAMPLE_RATE_HZ, WINDOW_MS
from .handfit import emg_samples, r2, read_jsonl
from .ridge_decoder import fit_ridge, predict_angles
from .signal import window_features

LABELS = ("forward", "up", "grip")
SKIP_STEPS = {"ready", "sync", "rest"}    # setup: getting ready, sync clenches (fist, grip label 0), settling
MAX_DELAY_S = 1.5
REGULARIZATION = (10.0, 100.0, 1e3, 1e4, 1e5, 1e6)
MIN_VARIANCE = 1e-4       # labels are arm lengths and 0..1 grip, not degrees
REPO = Path(__file__).resolve().parent.parent
VIEWER = REPO / "web" / "hand_map" / "viewer.html"


def target_track(rows):
    out = [(r["host_time_s"], r["target"], r.get("followed")) for r in rows if r.get("target")]
    if len(out) < 100:
        raise ValueError("this block has no on-screen targets; record it with --protocol follow")
    t = np.array([o[0] for o in out])
    tgt = {k: np.array([o[1][k] for o in out], dtype=float) for k in ("forward", "up", "grip", "shoulder_lift", "elbow_flex")}
    fol = {k: np.array([o[2][k] if o[2] else np.nan for o in out], dtype=float)
           for k in ("forward", "up", "shoulder_lift", "elbow_flex")}
    return {"time": t, "target": tgt, "followed": fol, "step": [o[1]["step"] for o in out],
            "round": [o[1]["round"] for o in out], "spot": [o[1]["spot"] for o in out]}


def follow_delay(track, dt=0.02):
    """Seconds your arm trailed the target, from forward+up position (camera frames only)."""
    t = track["time"]
    grid = np.arange(t[0], t[-1], dt)
    seen = ~np.isnan(track["followed"]["forward"])
    score = {}
    for steps in range(0, int(MAX_DELAY_S / dt) + 1):
        shift = steps * dt
        err = []
        for k in ("forward", "up"):
            target = np.interp(grid - shift, t, track["target"][k])
            followed = np.interp(grid, t[seen], track["followed"][k][seen])
            err.append(np.abs(target - followed))
        score[shift] = float(np.mean(err))
    best = min(score, key=score.get)
    return best, score[best], score[0.0]


def compliance(track, delay):
    t = track["time"]
    seen = ~np.isnan(track["followed"]["forward"])
    out = {"camera_saw_arm_pct": float(100 * seen.mean()), "delay_s": delay}
    for k in ("forward", "up", "shoulder_lift", "elbow_flex"):
        target = np.interp(t[seen] - delay, t, track["target"][k])
        followed = track["followed"][k][seen]
        r = np.corrcoef(target, followed)[0, 1] if np.std(followed) > 1e-9 else np.nan
        out[k] = {"mae": float(np.mean(np.abs(target - followed))), "r": float(r),
                  "bias": float(np.mean(followed - target))}
    return out


def band_windows(block: Path):
    """End time, EMG features (lagged windows) and band-motion features every 20 ms."""
    records = read_jsonl(block / "emg.jsonl")
    times, emg, valid = emg_samples(records, keep_longest=True)
    if len({r.get("band_connection_id") for r in records}) > 1:
        print(f"band reconnected during this block: using its longest connection, "
              f"{times[-1] - times[0]:.0f} of {records[-1]['host_time_s'] - records[0]['host_time_s']:.0f} s")
    motion_file = block / "motion.jsonl"
    try:
        orient_t, orient_m, gyro_t, gyro_v = load_motion(read_jsonl(motion_file)) if motion_file.exists() else (None,) * 4
    except ValueError:
        orient_t = orient_m = gyro_t = gyro_v = None
    if orient_t is None:
        print("no band orientation/gyro in this block: fitting EMG only")
    window = int(SAMPLE_RATE_HZ * WINDOW_MS / 1000)
    hop = int(SAMPLE_RATE_HZ * HOP_MS / 1000)
    lags = [int(ms // HOP_MS) for ms in LAGS_MS]
    ends = list(range(window - 1, len(emg), hop))
    rows = [window_features(emg[e - window + 1:e + 1]) if valid[e - window + 1:e + 1].all() else None for e in ends]
    t_out, emg_x, motion_x = [], [], []
    for position, end in enumerate(ends):
        if position < max(lags) or any(rows[position - lag] is None for lag in lags):
            continue
        t_end = float(times[end])
        if orient_t is not None:
            o = int(np.clip(np.searchsorted(orient_t, t_end), 1, len(orient_t) - 1))
            in_window = (gyro_t > t_end - WINDOW_MS / 1000) & (gyro_t <= t_end)
            if abs(orient_t[o] - t_end) > 0.1 or not in_window.any():
                continue
            g = gyro_v[in_window]
            motion_x.append(np.concatenate([orient_m[o], g.mean(axis=0), np.abs(g).mean(axis=0)]))
        t_out.append(t_end)
        emg_x.append(np.concatenate([rows[position - lag] for lag in lags]))
    return np.array(t_out), np.array(emg_x), (np.array(motion_x) if orient_t is not None else None)


def label_windows(track, times, delay):
    """Target shown `delay` before each window = what you were doing then. Plus step/round."""
    shown = times - delay
    t = track["time"]
    inside = (shown >= t[0]) & (shown <= t[-1])
    idx = np.clip(np.searchsorted(t, shown) - 1, 0, len(t) - 1)
    y = np.column_stack([np.interp(shown, t, track["target"][k]) for k in LABELS])
    steps = [track["step"][i] for i in idx]
    rounds = [track["round"][i] for i in idx]
    spots = [track["spot"][i] for i in idx]
    keep = inside & np.array([s not in SKIP_STEPS for s in steps])
    return y, steps, rounds, spots, keep


def fit_rounds(x, y, groups):
    """Leave one round out; windows outside any round are always training data. Regularization is
    chosen by the pooled held-out score across rounds: adjacent windows of the same round are too
    similar to choose it honestly, and that picked far too little and overfit. With only 2 rounds
    this choice makes the scores slightly optimistic."""
    rounds = sorted({g for g in groups if g is not None})
    if len(rounds) < 2:
        raise ValueError("need at least 2 rounds to test on unseen movement")
    groups = np.array(groups, dtype=object)
    tested = np.isin(groups, rounds)
    out = np.full_like(y, np.nan)
    chosen = []
    for col in range(y.shape[1]):                  # each output gets its own regularization
        best = None
        for reg in REGULARIZATION:
            predicted = np.full(len(y), np.nan)
            for held in rounds:
                test = groups == held
                predicted[test] = predict_angles(fit_ridge(x, y[:, col:col + 1], ~test, reg), x[test])[:, 0]
            score = r2(y[tested, col:col + 1], predicted[tested, None], MIN_VARIANCE)[0]
            score = -np.inf if np.isnan(score) else score
            if best is None or score > best[0]:
                best = (score, predicted, reg)
        out[:, col] = best[1]
        chosen.append(best[2])
    return out, tested, [float(c) for c in chosen]


def emg_fist_check(block: Path, track, delay):
    """Does EMG rise when the screen says FIST? Per channel fist / open strength, on round steps.
    Returns (best ratio, its channel number, all ratios), or None. Channels differ between blocks
    because the band sits a little differently each time it is put on."""
    times, emg, _ = emg_samples(read_jsonl(block / "emg.jsonl"), keep_longest=True)
    k = int(0.1 * SAMPLE_RATE_HZ)
    env = np.stack([np.convolve(emg[:, c] ** 2, np.ones(k) / k, mode="same") ** 0.5 for c in range(8)], axis=1)
    shown = times - delay
    inside = (shown >= track["time"][0]) & (shown <= track["time"][-1])
    grip = np.interp(shown, track["time"], track["target"]["grip"])
    idx = np.clip(np.searchsorted(track["time"], shown) - 1, 0, len(track["time"]) - 1)
    in_round = np.array([track["round"][i] is not None for i in idx])
    fist, open_ = inside & in_round & (grip > 0.9), inside & in_round & (grip < 0.1)
    if not fist.any() or not open_.any():
        return None
    ratios = np.median(env[fist], axis=0) / np.maximum(np.median(env[open_], axis=0), 1e-9)
    best = int(np.argmax(ratios))
    return float(ratios[best]), best + 1, [round(float(r), 2) for r in ratios]


def scores(y, predicted, tested):
    out = {}
    s = r2(y[tested], predicted[tested], MIN_VARIANCE)
    for i, k in enumerate(LABELS):
        out[k] = {"r2": None if np.isnan(s[i]) else float(s[i]),
                  "mae": float(np.mean(np.abs(y[tested, i] - predicted[tested, i])))}
    grip_true, grip_pred = y[tested, 2] > 0.5, predicted[tested, 2] > 0.5
    out["grip"]["fist_vs_open_accuracy"] = float(np.mean(grip_true == grip_pred))
    angles_true = np.array([guided.arm_ik(f, u) for f, u in y[tested, :2]])
    angles_pred = np.array([guided.arm_ik(f, u) for f, u in predicted[tested, :2]])
    for i, k in enumerate(("shoulder_lift", "elbow_flex")):
        out[k] = {"mae_deg": float(np.mean(np.abs(angles_true[:, i] - angles_pred[:, i]))),
                  "r2": float(r2(angles_true[:, i:i + 1], angles_pred[:, i:i + 1])[0])}   # degrees
    return out


def smooth(values, seconds, dt=HOP_MS / 1000):
    alpha = dt / (seconds + dt)
    out = np.array(values, dtype=float)
    for i in range(1, len(out)):
        out[i] = out[i - 1] + alpha * (out[i] - out[i - 1])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("block", help="a block recorded with --protocol follow")
    parser.add_argument("--delay", type=float, help="force the follow delay in seconds (default: measured)")
    parser.add_argument("--labels", choices=("target", "camera"), default="target",
                        help="target: the on-screen target shown `delay` earlier (default); camera: where the "
                             "camera saw your hand (grip still comes from the target)")
    args = parser.parse_args(argv)
    block = Path(args.block)
    track = target_track(read_jsonl(block / "arm.jsonl"))
    delay, err_at, err_zero = follow_delay(track)
    if args.delay is not None:
        delay = args.delay
    follow = compliance(track, delay)
    print(f"{block.name}: camera saw your arm {follow['camera_saw_arm_pct']:.0f}% of frames")
    print(f"you trailed the target by {delay * 1000:.0f} ms (hand error {err_zero:.2f} -> {err_at:.2f} arm lengths "
          f"once that delay is removed)")
    print("how closely you copied it (after the delay):  " + "   ".join(
        f"{k.replace('_', ' ')} r={follow[k]['r']:.2f} err {follow[k]['mae']:.2f}{'°' if 'flex' in k or 'lift' in k else ''}"
        for k in ("forward", "up", "shoulder_lift", "elbow_flex")))

    check = emg_fist_check(block, track, delay)
    ratio = check[0] if check else None
    if check:
        verdict = "good" if ratio >= 2 else "weak" if ratio >= 1.3 else "NO fist response: grip can't be learned"
        print(f"EMG during FIST vs open-hand steps: {ratio:.2f}x on ch{check[1]} ({verdict}); "
              "per channel " + " ".join(f"ch{c + 1} {r:.1f}" for c, r in enumerate(check[2])))

    times, emg_x, motion_x = band_windows(block)
    y, steps, rounds, spots, keep = label_windows(track, times, delay)
    if args.labels == "camera":
        seen = ~np.isnan(track["followed"]["forward"])
        for i, k in enumerate(("forward", "up")):
            y[:, i] = np.interp(times, track["time"][seen], track["followed"][k][seen])
    times, emg_x, y = times[keep], emg_x[keep], y[keep]
    motion_x = motion_x[keep] if motion_x is not None else None
    steps = [s for s, k in zip(steps, keep) if k]
    rounds = [r for r, k in zip(rounds, keep) if k]
    spots = [s for s, k in zip(spots, keep) if k]
    inputs = {"EMG only": emg_x}
    if motion_x is not None:
        inputs.update({"band motion only": motion_x, "EMG + band motion": np.hstack([emg_x, motion_x])})
    results = {}
    for name, x in inputs.items():
        predicted, tested, reg = fit_rounds(x, y, rounds)
        results[name] = {"predicted": predicted, "tested": tested, "regularization": reg,
                         "scores": scores(y, predicted, tested)}
    n_test = int(next(iter(results.values()))["tested"].sum())
    print(f"\n{len(y)} windows labelled by the {'on-screen target' if args.labels == 'target' else 'camera'}; "
          f"each round predicted by a model trained on the other ({n_test} held out)")
    print(f"{'held-out':20}{'hand fwd R²':>12}{'hand up R²':>12}{'grip R²':>9}{'fist/open':>11}{'shoulder err':>14}{'elbow err':>11}")
    for name, res in results.items():
        s = res["scores"]
        fmt = lambda v: "—" if v is None else f"{v:.2f}"
        print(f"{name:20}{fmt(s['forward']['r2']):>12}{fmt(s['up']['r2']):>12}{fmt(s['grip']['r2']):>9}"
              f"{100 * s['grip']['fist_vs_open_accuracy']:>10.0f}%{s['shoulder_lift']['mae_deg']:>13.0f}°"
              f"{s['elbow_flex']['mae_deg']:>10.0f}°")

    # Drive the simulated robot from the best model's held-out predictions.
    from .dashboard import load_home
    from .hand_map import MAP_FILE, robot_targets, solve
    from .pick_sim import (FORWARD_TICK_SIGN, GEOMETRY, PLANAR, TICKS_PER_DEGREE, forward_kinematics, gravity_torque,
                           joint_limits)
    from .robot_map import follow as follower, load_map
    best = results.get("EMG + band motion", results["EMG only"])
    tested = best["tested"]
    pred = best["predicted"][tested].copy()
    pred[:, 0], pred[:, 1] = smooth(pred[:, 0], 0.25), smooth(pred[:, 1], 0.25)
    pred[:, 2] = np.clip(smooth(pred[:, 2], 0.3), 0, 1)
    true = y[tested]
    cfg = json.loads(MAP_FILE.read_text())
    human = {"table_up": guided.TABLE_UP + 0.02, "forward_range": [min(guided.TARGETS), max(guided.TARGETS)]}
    rc, home = load_map(), load_home()
    limits = joint_limits(rc, home)
    px, pz = robot_targets({"forward": pred[:, 0], "up": pred[:, 1], "arm_m": 0.55}, human, cfg["robot"])
    tx, tz = robot_targets({"forward": true[:, 0], "up": true[:, 1], "arm_m": 0.55}, human, cfg["robot"])
    angles, _, moved = solve(px, pz, limits, cfg["max_pitch_change_deg"])
    tips = np.array([forward_kinematics(a)[-1] for a in angles])
    robot_err = np.hypot(tips[:, 0] - tx, tips[:, 1] - tz)
    t_steps = [s for s, k in zip(steps, tested) if k]
    at_grip = np.array([s in ("grip", "hold") for s in t_steps])
    speed = np.abs(np.gradient(angles, HOP_MS / 1000, axis=0))
    print(f"\nrobot gripper vs where the true target would put it: median {np.median(robot_err) * 100:.1f} cm, "
          f"90% within {np.percentile(robot_err, 90) * 100:.1f} cm; during grip/hold median {np.median(robot_err[at_grip]) * 100:.1f} cm"
          if at_grip.any() else "")
    print(f"robot joint speed p95: " + ", ".join(f"{j} {np.percentile(speed[:, k], 95):.0f}°/s" for k, j in enumerate(PLANAR)) +
          f" (cap 32); unreachable targets moved {100 * moved.mean():.0f}%")

    metrics = {"block": block.name, "follow": follow, "delay_s": delay, "labels": args.labels,
               "emg_fist_ratio": ratio,
               "models": {n: {"regularization": r["regularization"], **r["scores"]} for n, r in results.items()},
               "robot": {"median_error_cm": float(np.median(robot_err) * 100),
                         "p90_error_cm": float(np.percentile(robot_err, 90) * 100)}}
    (block / "guided_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    ticks = {j: home[j] + FORWARD_TICK_SIGN[j] * angles[:, k] * TICKS_PER_DEGREE for k, j in enumerate(PLANAR)}
    cap, accel = rc["speed_caps_ticks_per_s"]["fast"], rc["acceleration_ticks_per_s2"]
    executed = np.column_stack([(follower(ticks[j], ticks[j][0], cap, accel, HOP_MS / 1000) - home[j])
                                / TICKS_PER_DEGREE * FORWARD_TICK_SIGN[j] for j in PLANAR])
    step = 2
    preview = {"dt": HOP_MS / 1000 * step, "cue": t_steps[::step], "block": block.name,
               "true_label": "on-screen target" if args.labels == "target" else "your hand as the camera saw it",
               "source": f"on-screen targets; your delay {delay * 1000:.0f} ms; held-out rounds predicted from the band",
               "target": np.c_[px, pz][::step].round(4).tolist(), "target_true": np.c_[tx, tz][::step].round(4).tolist(),
               "tip": tips[::step].round(4).tolist(), "angles": angles[::step].round(1).tolist(),
               "executed": executed[::step].round(1).tolist(), "moved": moved[::step].tolist(),
               "grip": pred[::step, 2].round(2).tolist(), "grip_true": true[::step, 2].round(2).tolist(),
               "torque": [round(gravity_torque(a)["shoulder_lift"], 2) for a in angles[::step]],
               "human": pred[::step, :2].round(3).tolist(), "human_true": true[::step, :2].round(3).tolist(),
               "human_calibration": human, "robot": cfg["robot"], "limits": {j: list(v) for j, v in limits.items()},
               "geometry": GEOMETRY}
    page = VIEWER.read_text(encoding="utf-8").replace(
        "__HAND_DATA__", json.dumps(preview, separators=(",", ":")).replace("</", "<\\/"))
    (block / "guided_preview.html").write_text(page, encoding="utf-8")
    print(f"\nMetrics: {block / 'guided_metrics.json'}\nPreview: {block / 'guided_preview.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
