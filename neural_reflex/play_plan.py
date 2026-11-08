"""Play a robot plan (robot_plan.json or robot_plan_decoded.json) on the leader arm.

    ./scripts/play_plan <plan.json> --port /dev/cu.YOUR_ARM_PORT --enable-motion   # moves the arm
    python -m neural_reflex.play_plan <plan.json> --dry-run                          # no hardware

Safety, matching the web controller:
- Only the plan's moving joints get torque; the others stay limp. Each is enabled at its
  measured position, and the run refuses to start if any joint is outside its limits or
  already has torque on.
- Targets are clamped to the motor limits minus margin_ticks from config/robot_map.json.
- Goals ramp from the measured pose to the plan's start at 'fine' speed, then follow the plan
  with every step limited to the speed cap.
- If a joint trails its goal by more than 60 ticks for 0.5 s, or on any error or Ctrl-C,
  torque is turned off on every joint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .leader_arm import JOINTS, disable_all_torque, inspect, make_bus, validate_motion_state
from .robot_map import follow, load_map

TRACKING_TOLERANCE = 60
TRACKING_TIMEOUT = 0.5
CHECK_EVERY = 5      # read positions every 5th step


def load_plan(path: Path, config: dict) -> tuple[dict, list[str], float]:
    plan = json.loads(path.read_text())
    moving = plan.get("moving") or [j for j in JOINTS if len(set(plan["targets"][j])) > 1]
    margin = config["margin_ticks"]
    targets = {}
    for j in moving:
        lo, hi = config["limits"][j]
        targets[j] = np.clip(np.asarray(plan["targets"][j], dtype=float), lo + margin, hi - margin)
    return targets, moving, float(plan.get("dt_s", 0.02))


def ramp(start: float, end: float, cap: float, dt: float) -> np.ndarray:
    n = max(1, int(np.ceil(abs(end - start) / (cap * dt))))
    return np.linspace(start, end, n + 1)[1:]


def schedule(targets: dict, start: dict, cap: float, fine: float, dt: float) -> dict:
    """Per-joint goal sequence: fine-speed ramp to the plan start, then the plan, step-limited."""
    lead = max(len(ramp(start[j], targets[j][0], fine, dt)) for j in targets)
    out = {}
    for j, t in targets.items():
        r = ramp(start[j], t[0], fine, dt)
        r = np.concatenate([r, np.full(lead - len(r), t[0])])
        goals, g = [], float(start[j])
        for target in np.concatenate([r, t]):
            g += float(np.clip(target - g, -cap * dt, cap * dt))
            goals.append(int(round(g)))
        out[j] = goals
    return out


def play(bus, goals: dict, dt: float, log=print) -> None:
    moving = list(goals)
    states = inspect(bus)
    validate_motion_state(states)
    enabled = []
    try:
        for j in moving:
            bus.write("Goal_Position", j, states[j].position, normalize=False)
            bus.enable_torque(j)
            if int(bus.read("Torque_Enable", j, normalize=False)) != 1:
                raise RuntimeError(f"{j} torque did not enable")
            enabled.append(j)
        n = len(goals[moving[0]])
        lagging_since: dict[str, float] = {}
        t0 = time.monotonic()
        for step in range(n):
            for j in moving:
                bus.write("Goal_Position", j, goals[j][step], normalize=False)
            if step % CHECK_EVERY == 0:
                now = time.monotonic()
                for j in moving:
                    err = abs(int(bus.read("Present_Position", j, normalize=False)) - goals[j][step])
                    if err <= TRACKING_TOLERANCE:
                        lagging_since.pop(j, None)
                    elif now - lagging_since.setdefault(j, now) > TRACKING_TIMEOUT:
                        raise RuntimeError(f"{j} is {err} ticks from its goal; stopping")
            if step % 250 == 0:
                log(f"  {step * dt:6.1f}s / {n * dt:.1f}s  " + "  ".join(f"{j}={goals[j][step]}" for j in moving))
            delay = t0 + (step + 1) * dt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        log("Plan finished.")
    finally:
        errors = disable_all_torque(bus)
        if errors:
            log("Could not verify torque off for " + ", ".join(errors) + "; disconnect 12 V power.")
        else:
            log("Torque off on all joints.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("plan", help="robot_plan.json or robot_plan_decoded.json")
    parser.add_argument("--port", help="arm USB serial port")
    parser.add_argument("--enable-motion", action="store_true", help="required to move the arm")
    parser.add_argument("--dry-run", action="store_true", help="simulate against the saved home pose")
    parser.add_argument("--speed", choices=("fine", "normal", "fast"), help="default: the plan's speed")
    args = parser.parse_args(argv)

    config = load_map()
    plan_json = json.loads(Path(args.plan).read_text())
    targets, moving, dt = load_plan(Path(args.plan), config)
    speed = args.speed or plan_json.get("speed", "normal")
    cap = config["speed_caps_ticks_per_s"][speed]
    fine = config["speed_caps_ticks_per_s"]["fine"]
    print(f"{Path(args.plan).name}: {plan_json.get('source', 'camera')}; moving {', '.join(moving)}; "
          f"{len(next(iter(targets.values()))) * dt:.0f} s at '{speed}' ({cap} ticks/s)")

    if args.dry_run:
        start = {j: plan_json["home"][j] for j in moving}
        goals = schedule(targets, start, cap, fine, dt)
        for j in moving:
            g = np.asarray(goals[j], float)
            sim = follow(g, start[j], cap, config["acceleration_ticks_per_s2"], dt)
            print(f"  {j:14} goals {int(g.min())}..{int(g.max())}, max step {int(np.abs(np.diff(g)).max())} ticks, "
                  f"simulated max lag {int(np.abs(sim - g).max())} ticks (stop at {TRACKING_TOLERANCE})")
        print(f"  {len(goals[moving[0]]) * dt:.0f} s including the ramp from home. Dry run only; nothing moved.")
        return 0

    if not (args.port and args.enable_motion):
        parser.error("moving the arm needs --port and --enable-motion (or use --dry-run)")
    if not sys.stdin.isatty():
        parser.error("run from an interactive terminal so you can confirm and Ctrl-C")
    bus = make_bus(args.port)
    bus.connect()
    try:
        states = inspect(bus)
        start = {j: states[j].position for j in moving}
        goals = schedule(targets, start, cap, fine, dt)
        print("Joints that will get torque: " + ", ".join(f"{j} (now {start[j]})" for j in moving))
        print("Clear the sweep area and keep the 12 V plug in reach. Ctrl-C stops and turns torque off.")
        if input("Type 'go' to start: ").strip() != "go":
            print("Not started.")
            return 1
        play(bus, goals, dt)
    except KeyboardInterrupt:
        print("\nStopped by Ctrl-C.")
    finally:
        bus.disconnect(disable_torque=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
