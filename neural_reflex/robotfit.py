"""Band signals -> SO-ARM101 motor angles, learned from guided recordings; writes a robot plan.

Labels: your hand position every 20 ms (camera by default, or the on-screen target), mapped through
hand_map into the robot's plane and solved into shoulder-lift, elbow and wrist-flex angles. Grip is
ignored. Inputs: the same EMG and band-motion features as guidedfit. Every round of every block is
predicted by a model trained on all the other rounds, so the plan for a block comes from a model
that never saw that movement.

The plan (robot_motor_plan.json in the block folder) is smoothed, kept inside the joint limits and
above the table, and slowed down until the arm controller can follow it within about 5 degrees.
neural_reflex.robot_replay sends it to the real arm. Nothing here talks to the arm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import guided
from .dashboard import load_home
from .guidedfit import band_windows, fit_rounds, follow_delay, label_windows, target_track
from .hand_map import MAP_FILE, robot_targets, solve
from .handfit import r2, read_jsonl
from .pick_sim import FORWARD_TICK_SIGN, PLANAR, TICKS_PER_DEGREE, check_pose, forward_kinematics, joint_limits
from .robot_map import follow, load_map, lowpass, stretch

DT = 0.02
MAX_FOLLOW_ERROR_DEG = 5.0


def block_dataset(block: Path, labels: str):
    rows = read_jsonl(block / "arm.jsonl")
    track = target_track(rows)
    delay = follow_delay(track)[0]
    times, emg_x, motion_x = band_windows(block)
    if motion_x is None:
        raise ValueError(f"{block.name} has no band orientation/gyro")
    y, steps, rounds, _, keep = label_windows(track, times, delay)
    hand = y[:, :2].copy()
    if labels == "camera":
        seen = ~np.isnan(track["followed"]["forward"])
        for i, k in enumerate(("forward", "up")):
            hand[:, i] = np.interp(times, track["time"][seen], track["followed"][k][seen])
    keep &= np.array([r is not None for r in rounds])
    return {"name": block.name, "times": times[keep], "emg": emg_x[keep], "motion": motion_x[keep],
            "hand": hand[keep], "steps": [s for s, k in zip(steps, keep) if k],
            "groups": [f"{block.name}#{r}" for r, k in zip(rounds, keep) if k], "delay": delay}


def hand_to_robot(hand, limits, config):
    human = {"table_up": guided.TABLE_UP + 0.02, "forward_range": [min(guided.TARGETS), max(guided.TARGETS)]}
    x, z = robot_targets({"forward": hand[:, 0], "up": hand[:, 1], "arm_m": 0.55}, human, config["robot"])
    angles, _, moved = solve(x, z, limits, config["max_pitch_change_deg"])
    return angles, moved


def make_feasible(angles, limits):
    """Clamp to joint limits; hold the last allowed pose wherever the arm would hit the table."""
    out = angles.copy()
    for k, joint in enumerate(PLANAR):
        out[:, k] = np.clip(out[:, k], *limits[joint])
    held = np.zeros(len(out), dtype=bool)
    last = np.zeros(3)
    for i in range(len(out)):
        if check_pose(out[i], limits):
            out[i], held[i] = last, True
        else:
            last = out[i]
    return out, held


def slowdown_for(ticks, cap, accel):
    """Slowdown so the controller tracks the plan within MAX_FOLLOW_ERROR_DEG, starting at the
    plan's first pose (the replay moves there separately, slowly, before starting)."""
    factor = 1.0
    while factor < 20:
        worst = 0.0
        for joint, values in ticks.items():
            planned = stretch(values, factor)
            done = follow(planned, planned[0], cap, accel, DT)
            worst = max(worst, float(np.max(np.abs(done - planned)) / TICKS_PER_DEGREE))
        if worst <= MAX_FOLLOW_ERROR_DEG:
            return factor, worst
        factor = round(factor + 0.5, 1)
    return factor, worst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("blocks", nargs="+", help="guided block folders (recorded with --protocol follow)")
    parser.add_argument("--labels", choices=("camera", "target"), default="camera")
    parser.add_argument("--plan-block", help="which block to write a robot plan for (default: the last)")
    parser.add_argument("--plan-from", choices=("recorded", "band"), default="recorded",
                        help="recorded: your movement as the camera measured it, mapped to the robot (tests the "
                             "robot side); band: the decoder's held-out predictions from band signals")
    parser.add_argument("--speed", choices=("fine", "normal", "fast"), default="normal",
                        help="arm controller speed the plan must suit (default normal, ~16°/s)")
    args = parser.parse_args(argv)
    config, robot_config, home = json.loads(MAP_FILE.read_text()), load_map(), load_home()
    limits = joint_limits(robot_config, home)

    data = []
    for path in args.blocks:
        try:
            d = block_dataset(Path(path), args.labels)
        except ValueError as error:
            print(f"skipping {Path(path).name}: {error}")
            continue
        d["angles"], d["moved"] = hand_to_robot(d["hand"], limits, config)
        data.append(d)
        print(f"{d['name']}: {len(d['times'])} windows, delay {d['delay'] * 1000:.0f} ms, "
              f"robot label range " + ", ".join(f"{j} {d['angles'][:, k].min():.0f}..{d['angles'][:, k].max():.0f}°"
                                                for k, j in enumerate(PLANAR)))
    if not data:
        raise SystemExit("no usable blocks")
    y = np.concatenate([d["angles"] for d in data])
    groups = [g for d in data for g in d["groups"]]
    emg, motion = np.concatenate([d["emg"] for d in data]), np.concatenate([d["motion"] for d in data])
    inputs = {"EMG only": emg, "band motion only": motion, "EMG + band motion": np.hstack([emg, motion])}
    results = {}
    print(f"\n{len(set(groups))} rounds; each predicted by a model trained on the others. Held-out motor angles:")
    print(f"{'input':20}" + "".join(f"{j + ' R²':>18}{'err':>7}" for j in PLANAR))
    for name, x in inputs.items():
        pred, tested, regs = fit_rounds(x, y, groups)
        s = r2(y[tested], pred[tested])
        err = np.abs(y[tested] - pred[tested]).mean(axis=0)
        results[name] = {"pred": pred, "tested": tested, "r2": s, "err": err, "regs": regs}
        print(f"{name:20}" + "".join(f"{('—' if np.isnan(v) else f'{v:.2f}'):>18}{e:>6.0f}°" for v, e in zip(s, err)))
    best_name = max(results, key=lambda n: np.nanmean(results[n]["r2"]))
    best = results[best_name]
    print(f"using: {best_name}")

    # Robot plan for one block, from held-out predictions.
    target = Path(args.plan_block).name if args.plan_block else data[-1]["name"]
    offset = 0
    for d in data:
        if d["name"] == target:
            break
        offset += len(d["times"])
    else:
        raise SystemExit(f"{target} is not among the usable blocks")
    n = len(d["times"])
    if args.plan_from == "band":
        pred = best["pred"][offset:offset + n]
        if np.nanmax(best["r2"]) <= 0.1:
            print(f"WARNING: the band decoder has no real skill here (best held-out R² "
                  f"{np.nanmax(best['r2']):.2f}); the robot will mostly sit near an average pose.")
    else:
        pred = d["angles"]
    grid = np.arange(d["times"][0], d["times"][-1], DT)
    smooth = np.column_stack([lowpass(np.interp(grid, d["times"], pred[:, k]), 0.3, DT) for k in range(3)])
    truth = np.column_stack([np.interp(grid, d["times"], d["angles"][:, k]) for k in range(3)])
    feasible, held = make_feasible(smooth, limits)
    ticks = {j: home[j] + FORWARD_TICK_SIGN[j] * feasible[:, k] * TICKS_PER_DEGREE for k, j in enumerate(PLANAR)}
    cap = robot_config["speed_caps_ticks_per_s"][args.speed]
    factor, worst = slowdown_for(ticks, cap, robot_config["acceleration_ticks_per_s2"])
    tips = np.array([forward_kinematics(a)[-1] for a in feasible])
    true_tips = np.array([forward_kinematics(a)[-1] for a in truth])
    tip_err = np.hypot(*(tips - true_tips).T)
    print(f"\nplan for {target} from {'band predictions' if args.plan_from == 'band' else 'your recorded movement'}: "
          f"{grid[-1] - grid[0]:.0f} s of movement, replayed {factor:g}x slower at "
          f"'{args.speed}' speed ({(grid[-1] - grid[0]) * factor / 60:.1f} min); arm follows within {worst:.1f}°")
    print(f"gripper tip vs your actual movement mapped to the robot: median {np.median(tip_err) * 100:.1f} cm, "
          f"90% within {np.percentile(tip_err, 90) * 100:.1f} cm; held still {100 * held.mean():.0f}% (table/limits)")
    print("joint ranges in the plan: " + ", ".join(
        f"{j} {feasible[:, k].min():.0f}..{feasible[:, k].max():.0f}° from home" for k, j in enumerate(PLANAR)))

    plan = {"block": target, "source": args.plan_from, "model": best_name if args.plan_from == "band" else None,
            "labels": args.labels, "speed": args.speed,
            "slowdown": factor, "dt_s": DT, "home": home, "joints": list(PLANAR),
            "direction_checked": {"shoulder_lift": True, "elbow_flex": False, "wrist_flex": False},
            "limits_ticks": {j: robot_config["limits"][j] for j in PLANAR},
            "margin_ticks": robot_config["margin_ticks"],
            "targets": {j: [int(round(v)) for v in stretch(ticks[j], factor)] for j in PLANAR},
            "truth_deg": {j: [round(float(v), 1) for v in stretch(truth[:, k], factor)[::5]] for k, j in enumerate(PLANAR)},
            "scores": {name: {"r2": [None if np.isnan(v) else round(float(v), 3) for v in r["r2"]],
                              "err_deg": [round(float(v), 1) for v in r["err"]]} for name, r in results.items()}}
    out = Path([p for p in args.blocks if Path(p).name == target][0]) / "robot_motor_plan.json"
    out.write_text(json.dumps(plan) + "\n")
    print(f"\nPlan: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
