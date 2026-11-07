"""Translate a recorded block's human movement into SO-ARM101 joint targets, and check what
the robot can actually follow.

Nothing here talks to the arm. It writes:
  robot_plan.json     joint targets every 20 ms (slowed down if needed), for a later replay
  robot_preview.html  human arm vs. simulated robot, per-joint limits, clipping and lag
and prints a feasibility table: which joints hit their limits, how fast the movement asks
the motors to go compared with their caps, and how much slower a replay has to run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .arm_pose import side_angles
from .armfit import arm_tracks, band_arm, load_motion
from .dashboard import load_home
from .handfit import emg_samples, read_jsonl
from .leader_arm import JOINTS

REPO = Path(__file__).resolve().parent.parent
MAP_FILE = REPO / "config" / "robot_map.json"
VIEWER = REPO / "web" / "robot_map" / "viewer.html"
DT = 0.02


def load_map(path: Path = MAP_FILE) -> dict:
    config = json.loads(path.read_text())
    missing = set(JOINTS) - set(config["joints"]) | set(JOINTS) - set(config["limits"])
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    return config


def _cue_window(rows, cue):
    times = [r["host_time_s"] for r in rows if r.get("cue") == cue]
    return (min(times), max(times)) if times else None


def human_signals(block_dir: Path, side: str | None = None) -> dict:
    """Human angles on a 20 ms grid: camera arm angles, band roll and EMG clench strength."""
    rows = read_jsonl(block_dir / "arm.jsonl")
    orient_t, orient_m, gyro_t, gyro_v = load_motion(read_jsonl(block_dir / "motion.jsonl"))
    chosen = side or band_arm(arm_tracks(rows), gyro_t, gyro_v)[0]
    frames = []
    for r in rows:
        if not r.get("world_landmarks"):
            continue
        a = (r.get("angles_by_side") or {}).get(chosen) or side_angles(r["world_landmarks"], r["visibility"], chosen)
        if a:
            frames.append((r["host_time_s"], a["shoulder_pan"], a["shoulder_lift"], a["elbow_flex"], a["wrist_flex"]))
    if len(frames) < 50:
        raise ValueError(f"only {len(frames)} frames show the {chosen} arm")
    f = np.array(frames, dtype=float)                     # pan may be NaN (None -> nan)
    grid = np.arange(f[0, 0], f[-1, 0], DT)
    out = {"time": grid - grid[0], "side": chosen}
    for col, name in ((2, "shoulder_lift"), (3, "elbow_flex"), (4, "wrist_flex")):
        out[name] = np.interp(grid, f[:, 0], f[:, col])
    pan_ok = ~np.isnan(f[:, 1])
    pan = np.interp(grid, f[pan_ok, 0], f[pan_ok, 1]) if pan_ok.sum() > 2 else np.zeros_like(grid)
    # Hold pan wherever the nearest camera frame had no defined pan (arm near vertical).
    nearest = np.clip(np.searchsorted(f[:, 0], grid), 0, len(f) - 1)
    defined = pan_ok[nearest]
    for i in range(len(pan)):
        if not defined[i]:
            pan[i] = pan[i - 1] if i else 0.0
    out["shoulder_pan"] = pan
    out["pan_defined"] = defined
    # Gaps longer than 0.3 s in the camera track are marked so the robot holds still there.
    gap = np.zeros_like(grid, dtype=bool)
    for a, b in zip(f[:-1, 0], f[1:, 0]):
        if b - a > 0.3:
            gap |= (grid > a) & (grid < b)
    out["camera_gap"] = gap

    # Wrist roll from the band: rotation of the band's x axis about its long axis, relative to rest.
    roll = np.degrees(np.unwrap(np.arctan2(orient_m[:, 7], orient_m[:, 8])))
    roll_grid = np.interp(grid, orient_t, roll)
    rest = _cue_window(rows, "rest")
    ref = np.median(roll_grid[(grid >= rest[0]) & (grid <= rest[1])]) if rest else np.median(roll_grid)
    out["wrist_roll"] = roll_grid - ref

    # Clench strength: mean RMS of conditioned EMG over 100 ms, 0 at rest, 1 at the sync clenches.
    times, emg, _ = emg_samples(read_jsonl(block_dir / "emg.jsonl"))
    power = np.convolve(np.mean(emg ** 2, axis=1), np.ones(205) / 205, mode="same") ** 0.5
    env = np.interp(grid, times, power)
    sync = _cue_window(rows, "sync")
    low = np.median(env[(grid >= rest[0]) & (grid <= rest[1])]) if rest else np.percentile(env, 10)
    high = np.percentile(env[(grid >= sync[0]) & (grid <= sync[1])], 95) if sync else np.percentile(env, 99)
    out["gripper"] = np.clip((env - low) / max(high - low, 1e-9), 0, 1)
    cue_rows = sorted((r["host_time_s"], r.get("cue")) for r in rows)
    cue_t = np.array([c[0] for c in cue_rows])
    out["cue"] = [cue_rows[max(0, int(np.searchsorted(cue_t, g)) - 1)][1] for g in grid]
    return out


def lowpass(values: np.ndarray, seconds: float, dt: float = DT) -> np.ndarray:
    """Causal one-pole low-pass (what a live controller could also do), time constant `seconds`."""
    if seconds <= 0:
        return values
    alpha = dt / (seconds + dt)
    out = np.array(values, dtype=float)
    for i in range(1, len(out)):
        out[i] = out[i - 1] + alpha * (out[i] - out[i - 1])
    return out


def translate(signals: dict, config: dict, home: dict) -> tuple[dict, dict]:
    """Robot target ticks per joint, and where each target had to be clamped."""
    k = config["ticks_per_degree"]
    targets, clamped = {}, {}
    for joint in JOINTS:
        spec = config["joints"][joint]
        lo, hi = config["limits"][joint]
        lo, hi = lo + config["margin_ticks"], hi - config["margin_ticks"]
        human = lowpass(np.asarray(signals[joint], dtype=float), spec.get("smooth_s", 0.0))
        if joint == "gripper":
            raw = home[joint] + spec["sign"] * spec["gain"] * human
        else:
            raw = home[joint] + spec["sign"] * spec["gain"] * (human - spec["human_at_home"]) * k
        hold = signals["camera_gap"] if joint in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex") else None
        if hold is not None and hold.any():
            raw = raw.copy()
            for i in np.flatnonzero(hold):
                raw[i] = raw[i - 1] if i else home[joint]
        targets[joint] = np.clip(raw, lo, hi)
        clamped[joint] = (raw < lo) | (raw > hi)
    return targets, clamped


def follow(target: np.ndarray, start: float, cap: float, accel: float, dt: float = DT) -> np.ndarray:
    """Same speed/acceleration-limited profile as the arm controller's motion loop."""
    position, velocity = float(start), 0.0
    out = np.empty(len(target))
    for i, goal in enumerate(target):
        difference = goal - position
        desired = np.copysign(min(cap, np.sqrt(2 * accel * abs(difference))), difference)
        velocity += np.clip(desired - velocity, -accel * dt, accel * dt)
        velocity = np.clip(velocity, -cap, cap)
        step = velocity * dt
        position = goal if abs(step) >= abs(difference) else position + step
        if position == goal:
            velocity = 0.0
        out[i] = position
    return out


def stretch(values: np.ndarray, factor: float) -> np.ndarray:
    """Play a trajectory `factor` times slower on the same 20 ms grid."""
    n = len(values)
    src = np.arange(int((n - 1) * factor) + 1) / factor
    return np.interp(src, np.arange(n), values)


def feasibility(targets, home, config, speed="fast"):
    cap = config["speed_caps_ticks_per_s"][speed]
    accel = config["acceleration_ticks_per_s2"]
    k = config["ticks_per_degree"]
    report = {}
    for joint in JOINTS:
        t = targets[joint]
        need = np.abs(np.gradient(t, DT))
        p95 = float(np.percentile(need, 95))
        slowdown = 1.0
        while slowdown < 12:
            stretched = stretch(t, slowdown)
            err = np.abs(follow(stretched, home[joint], cap, accel) - stretched)
            if np.percentile(err, 95) <= 5 * k:     # within 5 degrees 95% of the time
                break
            slowdown = round(slowdown + 0.5, 1)
        realtime = np.abs(follow(t, home[joint], cap, accel) - t)
        report[joint] = {"target_range_deg": [float((t.min() - home[joint]) / k), float((t.max() - home[joint]) / k)],
                         "p95_speed_deg_s": p95 / k, "cap_deg_s": cap / k,
                         "realtime_lag_p95_deg": float(np.percentile(realtime, 95) / k),
                         "slowdown_for_5deg": slowdown}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("block", help="block folder with arm.jsonl, emg.jsonl, motion.jsonl")
    parser.add_argument("--side", choices=("left", "right"))
    parser.add_argument("--speed", choices=("fine", "normal", "fast"), default="fast")
    parser.add_argument("--slowdown", type=float, help="replay this many times slower (default: what the slowest joint needs)")
    args = parser.parse_args(argv)
    block = Path(args.block)
    config, home = load_map(), load_home()
    if home is None:
        raise SystemExit("config/arm_home.json is required")
    signals = human_signals(block, args.side)
    targets, clamped = translate(signals, config, home)
    report = feasibility(targets, home, config, args.speed)
    k = config["ticks_per_degree"]
    slowdown = args.slowdown or max(r["slowdown_for_5deg"] for j, r in report.items())
    cap = config["speed_caps_ticks_per_s"][args.speed]
    print(f"{block.name}: {signals['time'][-1]:.0f} s of movement, band arm = MediaPipe '{signals['side']}'")
    print(f"{'joint':14}{'dir checked':>12}{'robot range used':>20}{'clipped':>9}{'needs °/s':>11}{'cap °/s':>9}"
          f"{'lag @1x':>9}{'slow-mo':>9}")
    for joint in JOINTS:
        r, spec = report[joint], config["joints"][joint]
        lo, hi = r["target_range_deg"]
        print(f"{joint:14}{'yes' if spec['verified'] else 'NO':>12}{f'{lo:+.0f}°..{hi:+.0f}°':>20}"
              f"{100 * clamped[joint].mean():>8.0f}%{r['p95_speed_deg_s']:>11.0f}{r['cap_deg_s']:>9.0f}"
              f"{r['realtime_lag_p95_deg']:>8.0f}°{r['slowdown_for_5deg']:>8.1f}x")
    print(f"Replay plan: {slowdown:g}x slower at '{args.speed}' speed "
          f"({signals['time'][-1] * slowdown / 60:.1f} min). Pan undefined (held) {100 * (~signals['pan_defined']).mean():.0f}% "
          f"of the time; camera gaps held {100 * signals['camera_gap'].mean():.0f}%.")

    planned = {j: stretch(targets[j], slowdown) for j in JOINTS}
    executed = {j: follow(planned[j], home[j], cap, config["acceleration_ticks_per_s2"]) for j in JOINTS}
    human_slow = {j: stretch(np.asarray(signals[j], float), slowdown) for j in JOINTS}
    cue_slow = [signals["cue"][min(len(signals["cue"]) - 1, int(i / slowdown))] for i in range(len(planned["gripper"]))]
    plan = {"block": block.name, "speed": args.speed, "slowdown": slowdown, "dt_s": DT, "home": home,
            "targets": {j: [int(round(v)) for v in planned[j]] for j in JOINTS}}
    (block / "robot_plan.json").write_text(json.dumps(plan) + "\n")
    step = max(1, int(round(0.04 / DT)))   # 25 fps is plenty for the preview
    preview = {"joints": list(JOINTS), "dt_s": DT * step, "slowdown": slowdown, "speed": args.speed,
               "home": home, "limits": config["limits"], "margin": config["margin_ticks"], "k": k,
               "verified": {j: config["joints"][j]["verified"] for j in JOINTS},
               "report": report, "clamped_pct": {j: float(100 * clamped[j].mean()) for j in JOINTS},
               "cue": cue_slow[::step],
               "human": {j: [round(float(v), 1) for v in human_slow[j][::step]] for j in JOINTS},
               "target": {j: [int(v) for v in planned[j][::step]] for j in JOINTS},
               "robot": {j: [int(v) for v in executed[j][::step]] for j in JOINTS}}
    page = VIEWER.read_text(encoding="utf-8").replace(
        "__ROBOT_DATA__", json.dumps(preview, separators=(",", ":")).replace("</", "<\\/"))
    (block / "robot_preview.html").write_text(page, encoding="utf-8")
    print(f"Plan: {block / 'robot_plan.json'}\nPreview: {block / 'robot_preview.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
