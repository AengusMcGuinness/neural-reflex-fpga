"""Map your hand position to the SO-ARM101 gripper position in one plane, then solve its joints.

1. Hand position: the band arm's hand (index/pinky midpoint) relative to its shoulder, split into
   FORWARD and UP in your torso frame and divided by your arm length. When the hips are hidden
   (sitting at a table), "up" falls back to the camera's vertical.
2. Robot target: your table height lines up with the robot's table, your forward reach is scaled
   into the robot's safe pick zone, and height above the table is scaled by up_scale
   (config/hand_map.json). Pick-protocol steps, when present, calibrate the table height and reach.
3. Joints: inverse kinematics per frame, choosing a feasible gripper angle close to the previous
   one; targets the robot cannot reach are moved to the nearest point it can.
4. Grip: EMG clench strength, as in robot_map.

Nothing talks to the arm. Writes hand_plan.json and hand_preview.html in the block folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .arm_pose import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER, SIDES
from .armfit import arm_tracks, band_arm, load_motion
from .dashboard import load_home
from .handfit import read_jsonl
from .pick_sim import (FORWARD_TICK_SIGN, GEOMETRY, GRIPPER_CLOSE_TICKS, PLANAR, TICKS_PER_DEGREE, check_pose,
                       forward_kinematics, gravity_torque, inverse_kinematics, joint_limits)
from .robot_map import DT, follow, human_signals, load_map, lowpass

REPO = Path(__file__).resolve().parent.parent
MAP_FILE = REPO / "config" / "hand_map.json"
VIEWER = REPO / "web" / "hand_map" / "viewer.html"
PITCHES = np.arange(180, 104, -1, dtype=float)   # 1° steps so the gripper angle can change smoothly
CALIBRATION_STEPS = {"touch", "grip"}      # hand resting on a table target in the pick protocol


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def body_frame(points, visibility):
    """Up and forward unit vectors. Uses the hips when visible, else the camera's vertical."""
    shoulders = (points[LEFT_SHOULDER] + points[RIGHT_SHOULDER]) / 2
    right = points[RIGHT_SHOULDER] - points[LEFT_SHOULDER]
    hips_seen = min(visibility[LEFT_HIP], visibility[RIGHT_HIP]) >= 0.5
    up = _unit(shoulders - (points[LEFT_HIP] + points[RIGHT_HIP]) / 2) if hips_seen else np.array([0.0, -1.0, 0.0])
    right = _unit(right - np.dot(right, up) * up)
    forward = np.cross(up, right)
    if np.dot(points[0] - shoulders, forward) < 0:            # the nose is in front
        forward = -forward
    return up, forward, hips_seen


def hand_track(rows, side):
    """Per frame: time, forward and up of the hand from the shoulder, in arm lengths."""
    index = SIDES[side]
    out, lengths = [], []
    for r in rows:
        if not r.get("world_landmarks"):
            continue
        vis = r["visibility"]
        if min(vis[index[k]] for k in ("shoulder", "elbow", "wrist")) < 0.5 or \
                min(vis[LEFT_SHOULDER], vis[RIGHT_SHOULDER]) < 0.5:
            continue
        p = np.asarray(r["world_landmarks"], dtype=float)
        up, forward, hips = body_frame(p, vis)
        shoulder = p[index["shoulder"]]
        hand = (p[index["index"]] + p[index["pinky"]]) / 2
        lengths.append(np.linalg.norm(p[index["elbow"]] - shoulder) + np.linalg.norm(p[index["wrist"]] - p[index["elbow"]]))
        out.append((r["host_time_s"], float(np.dot(hand - shoulder, forward)), float(np.dot(hand - shoulder, up)),
                    r.get("cue"), hips))
    if len(out) < 50:
        raise ValueError(f"only {len(out)} frames show the {side} arm and both shoulders")
    arm = float(np.median(lengths))
    t = np.array([o[0] for o in out])
    return {"time": t, "forward": np.array([o[1] for o in out]) / arm, "up": np.array([o[2] for o in out]) / arm,
            "cue": [o[3] for o in out], "hips_seen_pct": 100 * float(np.mean([o[4] for o in out])), "arm_m": arm}


def calibrate(track, config):
    """Table height and reach range: from pick-protocol 'touch'/'grip' steps if present."""
    human = dict(config["human"])
    touch = np.array([c in CALIBRATION_STEPS for c in track["cue"]])
    source = "config defaults"
    if touch.sum() > 30:
        human["table_up"] = float(np.median(track["up"][touch]))
        human["forward_range"] = [float(np.percentile(track["forward"][touch], 5)),
                                  float(np.percentile(track["forward"][touch], 95))]
        source = f"{int(touch.sum())} frames with your hand on the table targets"
    return human, source


def robot_targets(track, human, robot):
    f0, f1 = human["forward_range"]
    x0, x1 = robot["forward_range_m"]
    span = max(f1 - f0, 1e-6)
    x = np.clip(x0 + (track["forward"] - f0) / span * (x1 - x0), *robot.get("x_limits_m", (-np.inf, np.inf)))
    z = robot["grasp_height_m"] + (track["up"] - human["table_up"]) * track["arm_m"] * robot["up_scale"]
    return x, np.clip(z, robot["grasp_height_m"], robot["max_height_m"])


def solve(x, z, limits, max_pitch_change, start_pitch=180.0):
    """Joint angles per frame. Prefers a gripper angle near the previous one; if the target is
    unreachable, searches nearby points (higher first, then toward the base or away)."""
    angles = np.zeros((len(x), 3))
    pitches = np.zeros(len(x))
    moved = np.zeros(len(x), dtype=bool)
    pitch_prev, prev = start_pitch, np.zeros(3)
    offsets = [(0, 0)] + [(dx, dz) for dz in (0, 0.01, 0.02, 0.04, 0.06) for dx in (0, -0.01, 0.01, -0.02, 0.02, -0.04, 0.04)
                          if (dx, dz) != (0, 0)]
    order = sorted(PITCHES, key=lambda p: (abs(p - pitch_prev) > max_pitch_change, abs(p - pitch_prev), -p))
    for i in range(len(x)):
        order = sorted(PITCHES, key=lambda p: (abs(p - pitch_prev) > max_pitch_change, abs(p - pitch_prev), -p))
        found = None
        for dx, dz in offsets:
            for p in order:
                a = inverse_kinematics(x[i] + dx, z[i] + dz, p)
                if a is not None and not check_pose(a, limits):
                    found = (a, p, (dx, dz) != (0, 0))
                    break
            if found:
                break
        if found:
            prev, pitch_prev, moved[i] = found[0], found[1], found[2]
        else:
            moved[i] = True                                       # nothing nearby: hold still
        angles[i], pitches[i] = prev, pitch_prev
    return angles, pitches, moved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("block", help="block folder with arm.jsonl, emg.jsonl, motion.jsonl")
    parser.add_argument("--side", choices=("left", "right"))
    args = parser.parse_args(argv)
    block = Path(args.block)
    config = json.loads(MAP_FILE.read_text())
    robot_config, home = load_map(), load_home()
    limits = joint_limits(robot_config, home)
    rows = read_jsonl(block / "arm.jsonl")
    _, _, gyro_t, gyro_v = load_motion(read_jsonl(block / "motion.jsonl"))
    side = args.side or band_arm(arm_tracks(rows), gyro_t, gyro_v)[0]
    track = hand_track(rows, side)
    human, source = calibrate(track, config)

    grid = np.arange(track["time"][0], track["time"][-1], DT)
    sampled = {"time": grid, "arm_m": track["arm_m"],
               "forward": lowpass(np.interp(grid, track["time"], track["forward"]), config["smooth_s"]),
               "up": lowpass(np.interp(grid, track["time"], track["up"]), config["smooth_s"])}
    cue_idx = np.clip(np.searchsorted(track["time"], grid), 0, len(track["time"]) - 1)
    cues = [track["cue"][i] for i in cue_idx]
    x, z = robot_targets(sampled, human, config["robot"])
    angles, pitches, moved = solve(x, z, limits, config["max_pitch_change_deg"])
    signals = human_signals(block, side)
    grip = np.interp(grid - grid[0], signals["time"], lowpass(signals["gripper"], 0.5))

    ticks = {j: home[j] + FORWARD_TICK_SIGN[j] * angles[:, k] * TICKS_PER_DEGREE for k, j in enumerate(PLANAR)}
    ticks["gripper"] = home["gripper"] + GRIPPER_CLOSE_TICKS * grip
    for j in ("shoulder_pan", "wrist_roll"):
        ticks[j] = np.full(len(grid), float(home[j]))
    cap, accel = robot_config["speed_caps_ticks_per_s"]["fast"], robot_config["acceleration_ticks_per_s2"]
    executed = {j: follow(ticks[j], home[j], cap, accel) for j in ticks}
    speed = {j: float(np.percentile(np.abs(np.gradient(angles[:, k], DT)), 95)) for k, j in enumerate(PLANAR)}
    lag = {j: float(np.percentile(np.abs(executed[j] - ticks[j]), 95) / TICKS_PER_DEGREE) for j in PLANAR}
    tips = np.array([forward_kinematics(a)[-1] for a in angles])
    miss = np.hypot(tips[:, 0] - x, tips[:, 1] - z)
    torque = np.array([gravity_torque(a)["shoulder_lift"] for a in angles])

    print(f"{block.name}: {grid[-1] - grid[0]:.0f} s, band arm '{side}', your arm {track['arm_m'] * 100:.0f} cm "
          f"(hips visible {track['hips_seen_pct']:.0f}% of frames)")
    print(f"calibration from {source}: table at {human['table_up']:+.2f} arm lengths from the shoulder, "
          f"reach {human['forward_range'][0]:.2f}–{human['forward_range'][1]:.2f}")
    print(f"robot tip: {x.min() * 100:.0f}–{x.max() * 100:.0f} cm out, {z.min() * 100:.0f}–{z.max() * 100:.0f} cm up; "
          f"moved to the nearest reachable point {100 * moved.mean():.0f}% of the time, "
          f"tip within 1 cm of target {100 * np.mean(miss < 0.01):.0f}%")
    print(f"shoulder load: median {np.median(torque):.1f}, max {torque.max():.1f} kg·cm")
    print("95th-percentile joint speed (°/s): " + ", ".join(f"{j} {speed[j]:.0f}" for j in PLANAR) +
          f"  (cap {cap / TICKS_PER_DEGREE:.0f}); lag " + ", ".join(f"{j} {lag[j]:.0f}°" for j in PLANAR))

    (block / "hand_plan.json").write_text(json.dumps({
        "side": side, "dt_s": DT, "calibration": human, "calibration_source": source, "home": home,
        "targets": {j: [int(round(v)) for v in ticks[j]] for j in ticks}}) + "\n")
    step = 2
    preview = {"dt": DT * step, "cue": cues[::step], "target": np.c_[x, z][::step].round(4).tolist(),
               "tip": tips[::step].round(4).tolist(), "angles": angles[::step].round(1).tolist(),
               "executed": np.c_[[(executed[j][::step] - home[j]) / TICKS_PER_DEGREE * FORWARD_TICK_SIGN[j]
                                  for j in PLANAR]].T.round(1).tolist(),
               "moved": moved[::step].tolist(), "grip": grip[::step].round(2).tolist(),
               "torque": torque[::step].round(2).tolist(),
               "human": np.c_[sampled["forward"], sampled["up"]][::step].round(3).tolist(),
               "human_calibration": human, "robot": config["robot"], "limits": {j: list(v) for j, v in limits.items()},
               "geometry": GEOMETRY,
               "source": source, "block": block.name}
    page = VIEWER.read_text(encoding="utf-8").replace(
        "__HAND_DATA__", json.dumps(preview, separators=(",", ":")).replace("</", "<\\/"))
    (block / "hand_preview.html").write_text(page, encoding="utf-8")
    print(f"Plan: {block / 'hand_plan.json'}\nPreview: {block / 'hand_preview.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
