#!/usr/bin/env python3
"""Smoothly sweep one joint out and back through ArmController, for direction checks.

Unlike `leader_arm nudge` (discrete 20-tick jumps), this ramps the joint at a capped
speed with ArmController's acceleration limit, so the motion is continuous and easy to
watch. Torque is turned off on exit, error or Ctrl+C.

    PYTHONPATH=.local/hiwonder-SoArm-101/src .venv-ik/bin/python -m so101_ik.joint_sweep \
        --port /dev/ttyACM0 --joint shoulder_pan --ticks 150 --speed fine
"""

from __future__ import annotations

import argparse
import threading
import time

from neural_reflex.arm_web import ArmController
from neural_reflex.leader_arm import JOINTS, make_bus

SETTLE_TICKS = 12   # the servos settle within about 10 ticks of a target
HOLD_S = 1.0


def wait_until(controller, joint: str, target: int, timeout: float) -> int:
    t0 = time.monotonic()
    while True:
        controller.heartbeat(controller.motion_epoch)
        snap = controller.status()
        if snap.get("fault"):
            raise RuntimeError(snap["fault"])
        pos = {j["name"]: j["position"] for j in snap["joints"]}[joint]
        if abs(pos - target) <= SETTLE_TICKS:
            return pos
        if time.monotonic() - t0 > timeout:
            raise RuntimeError(f"{joint} stopped at {pos}, target {target}")
        time.sleep(0.05)


def sweep(controller, joint: str, ticks: int, speed: str, repeats: int) -> None:
    stop = threading.Event()

    def motion_loop():
        while not stop.wait(0.02):
            controller.motion_step()

    threading.Thread(target=motion_loop, daemon=True).start()
    try:
        controller.set_speed(speed)
        snap = controller.torque(joint, True, controller.motion_epoch)
        if snap.get("fault"):
            raise RuntimeError(snap["fault"])
        start = {j["name"]: j["position"] for j in snap["joints"]}[joint]
        timeout = abs(ticks) / ArmController.SPEEDS[speed] + 5.0
        print(f"{joint}: start={start}, sweeping {ticks:+d} ticks at '{speed}' "
              f"({ArmController.SPEEDS[speed]} ticks/s), {repeats}x")
        for n in range(repeats):
            for label, target in (("out", start + ticks), ("back", start)):
                controller.target(joint, target, controller.motion_epoch)
                pos = wait_until(controller, joint, target, timeout)
                print(f"  {label}: target={target} measured={pos}")
                hold_until = time.monotonic() + HOLD_S
                while time.monotonic() < hold_until:
                    controller.target(joint, target, controller.motion_epoch)
                    time.sleep(0.1)
    finally:
        stop.set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", required=True)
    parser.add_argument("--joint", required=True, choices=JOINTS)
    parser.add_argument("--ticks", type=int, default=150, help="signed sweep size, |ticks| <= 300")
    parser.add_argument("--speed", choices=sorted(ArmController.SPEEDS), default="fine")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= abs(args.ticks) <= 300:
        parser.error("--ticks must be within +-300 (about 26 degrees)")
    if not 1 <= args.repeats <= 5:
        parser.error("--repeats must be 1..5")
    controller = ArmController(make_bus(args.port), timeout=30.0)
    try:
        sweep(controller, args.joint, args.ticks, args.speed, args.repeats)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except (RuntimeError, ValueError) as error:
        print(f"Stopped: {error}")
    finally:
        controller.stop()
        controller.close()
        print("Torque off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
