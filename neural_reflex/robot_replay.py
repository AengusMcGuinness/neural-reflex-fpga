"""Send a robot_motor_plan.json to the real SO-ARM101, slowly and with the arm controller's safety.

DRY RUN by default: prints what would move and how far, and touches nothing. With --execute it:
  1. checks the arm reads straight-up home (within ~13 degrees) on every planned joint
  2. powers shoulder lift, elbow and wrist flex (all three hold, so the arm cannot flop), but only
     MOVES the joints you list with --joints (default: shoulder_lift, the only direction checked)
  3. moves slowly to the plan's first pose, then plays the plan in real time
  4. returns to home, then asks you to support the arm before turning the motors off
Ctrl+C at any point holds the arm where it is and asks before turning the motors off.
All motion goes through ArmController: speed and acceleration ramps, joint limits, a tracking
check that stops the motors if a joint can't follow, and torque-off on any fault.
Only one program can use the arm's port: stop the dashboard first.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

PLANAR = ("shoulder_lift", "elbow_flex", "wrist_flex")
TICKS_PER_DEGREE = 4096 / 360
GEOMETRY_FILE = Path(__file__).resolve().parent.parent / "config" / "robot_geometry.json"
# Link masses (kg) and centre of mass as a fraction along each link, from the SO-101 URDF (as pick_sim).
MASSES = ((0, 0.103, 0.776), (1, 0.104, 0.726), (2, 0.079, 0.245), (2, 0.099, 0.541))
SAFE_KG_CM, MAX_KG_CM = 4.0, 6.0      # 40% / 60% of the leader servos' 10 kg-cm stall
HOME_TOLERANCE = 150            # ticks (~13 degrees) from home before anything moves
ARRIVE_TOLERANCE = 15
MOVE_TIMEOUT_S = 40


def shoulder_load(lift, elbow, wrist, geometry=None) -> float:
    """kg-cm the shoulder motor holds at this pose (degrees from straight up, forward positive)."""
    import math
    g = geometry or json.loads(GEOMETRY_FILE.read_text())
    links = (g["upper_arm_m"], g["forearm_m"], g["hand_m"])
    heading, x, xs = 0.0, 0.0, [0.0]
    for length, angle in zip(links, (lift, elbow, wrist)):
        heading += math.radians(angle)
        x += length * math.sin(heading)
        xs.append(x)
    return sum(m * abs(xs[i] + f * (xs[i + 1] - xs[i])) * 100 for i, m, f in MASSES)


def executed_degrees(plan, joints, first, last, max_degrees):
    """Degrees from home each planar joint will actually take: moving joints follow the plan
    (clamped to +-max_degrees), the others hold at home."""
    home, out = plan["home"], {}
    for j in PLANAR:
        if j in joints:
            values = [(t - home[j]) / TICKS_PER_DEGREE for t in plan["targets"][j][first:last]]
            out[j] = [max(-max_degrees, min(max_degrees, v)) for v in values]
        else:
            out[j] = [0.0] * (last - first)
    return out


def apply_limits(plan, joints, first, last, max_degrees):
    """Clamp the moving joints' targets in place and return the worst shoulder load."""
    deg = executed_degrees(plan, joints, first, last, max_degrees)
    for j in joints:   # tick direction = forward for all three joints (pick_sim.FORWARD_TICK_SIGN)
        plan["targets"][j][first:last] = [int(round(plan["home"][j] + d * TICKS_PER_DEGREE)) for d in deg[j]]
    geometry = json.loads(GEOMETRY_FILE.read_text())
    loads = [shoulder_load(a, b, c, geometry) for a, b, c in zip(deg["shoulder_lift"], deg["elbow_flex"], deg["wrist_flex"])]
    return max(loads), sum(v > SAFE_KG_CM for v in loads) / len(loads)


def load_plan(path: Path, joints: list[str], start_s: float, seconds: float | None):
    plan = json.loads(path.read_text())
    unknown = set(joints) - set(plan["joints"])
    if unknown:
        raise ValueError(f"plan has no {sorted(unknown)}")
    n = len(plan["targets"][plan["joints"][0]])
    first = int(start_s / plan["dt_s"])
    last = n if seconds is None else min(n, first + int(seconds / plan["dt_s"]))
    if not 0 <= first < last:
        raise ValueError("start/seconds select no part of the plan")
    for j in plan["joints"]:
        lo, hi = plan["limits_ticks"][j]
        values = plan["targets"][j][first:last]
        if min(values) < lo or max(values) > hi:
            raise ValueError(f"{j} plan leaves its limits {lo}..{hi}")
    return plan, first, last


def summary(plan, joints, first, last, max_degrees=180.0, load=None) -> str:
    home, dt = plan["home"], plan["dt_s"]
    lines = [f"plan: {plan['block']} ({plan.get('source', 'recorded')}), {(last - first) * dt:.0f} s "
             f"({plan['slowdown']:g}x slower than you moved) at '{plan['speed']}' speed"]
    for j in PLANAR:
        values = plan["targets"][j][first:last]
        lo, hi = (min(values) - home[j]) / TICKS_PER_DEGREE, (max(values) - home[j]) / TICKS_PER_DEGREE
        start = (values[0] - home[j]) / TICKS_PER_DEGREE
        if j in joints:
            state = "MOVES" + ("" if plan["direction_checked"].get(j) else "  (direction NOT checked on the arm)")
        else:
            state = "holds at home"
        lines.append(f"  {j:14} {state}: start {start:+.0f}°, range {lo:+.0f}..{hi:+.0f}° from straight up")
    lines.append("  base rotation, wrist roll, gripper: motors stay off")
    if max_degrees < 180:
        lines.append(f"  moving joints limited to ±{max_degrees:g}° from straight up (--max-degrees)")
    if load is not None:
        lines.append(f"  shoulder load: max {load[0]:.1f} kg·cm ({10 * load[0]:.0f}% of stall), "
                     f"over the safe {SAFE_KG_CM:g} kg·cm {100 * load[1]:.0f}% of the time")
    return "\n".join(lines)


class Replay:
    def __init__(self, controller, plan, joints, first, last, sleep=time.sleep, clock=time.monotonic, log=print):
        self.c, self.plan, self.joints = controller, plan, joints
        self.first, self.last = first, last
        self.sleep, self.clock, self.log = sleep, clock, log
        self.stop_loop = threading.Event()
        self.loop = None

    def _motion_loop(self):
        while not self.stop_loop.wait(0.02):
            self.c.motion_step()

    def _snapshot(self):
        return self.c.status()

    def _positions(self, snap):
        return {j["name"]: j["position"] for j in snap["joints"]}

    def _check(self, snap):
        if snap.get("fault"):
            raise RuntimeError(snap["fault"])

    def _send(self, targets):
        epoch = self.c.motion_epoch
        for name, tick in targets.items():
            snap = self.c.target(name, int(tick), epoch)
            self._check(snap)

    def _wait_arrive(self, targets, label):
        deadline = self.clock() + MOVE_TIMEOUT_S
        while True:
            self._send(targets)                         # also renews the controller's lease
            snap = self._snapshot()
            self._check(snap)
            pos = self._positions(snap)
            if all(abs(pos[j] - t) <= ARRIVE_TOLERANCE for j, t in targets.items()):
                return
            if self.clock() > deadline:
                raise RuntimeError(f"arm did not reach the {label} pose in {MOVE_TIMEOUT_S} s")
            self.sleep(0.1)

    def start(self):
        snap = self._snapshot()
        self._check(snap)
        home = self.plan["home"]
        pos = self._positions(snap)
        off = {j: pos[j] - home[j] for j in PLANAR if abs(pos[j] - home[j]) > HOME_TOLERANCE}
        if off:
            raise RuntimeError("put the arm in its straight-up home pose first; off by " +
                               ", ".join(f"{j} {v / TICKS_PER_DEGREE:+.0f}°" for j, v in off.items()))
        self.c.set_speed(self.plan["speed"])
        self.loop = threading.Thread(target=self._motion_loop, daemon=True)
        self.loop.start()
        for j in PLANAR:                                 # all three hold, so nothing can flop
            snap = self.c.torque(j, True, self.c.motion_epoch)
            self._check(snap)
        self.log("motors on (shoulder lift, elbow, wrist flex holding)")

    def run(self):
        home, targets, dt = self.plan["home"], self.plan["targets"], self.plan["dt_s"]
        first_pose = {j: targets[j][self.first] for j in self.joints}
        self.log("moving slowly to the first pose...")
        self._wait_arrive(first_pose, "first")
        self.log(f"playing {(self.last - self.first) * dt:.0f} s (Ctrl+C holds the arm)")
        t0 = self.clock()
        for i in range(self.first, self.last):
            self._send({j: targets[j][i] for j in self.joints})
            if (i - self.first) % 25 == 0:
                self._check(self._snapshot())
            wait = t0 + (i - self.first + 1) * dt - self.clock()
            if wait > 0:
                self.sleep(wait)
        self.log("returning home...")
        self._wait_arrive({j: home[j] for j in self.joints}, "home")
        self.log("back at home")

    def hold(self):
        try:
            self.c.pause(self.c.motion_epoch)
        except Exception:                                # noqa: BLE001 - holding is best effort
            pass

    def finish(self):
        self.stop_loop.set()
        if self.loop:
            self.loop.join(timeout=1)
        self.c.close()                                   # turns every motor off


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("plan", help="robot_motor_plan.json from neural_reflex.robotfit")
    parser.add_argument("--port", help="arm USB serial port (needed with --execute)")
    parser.add_argument("--joints", nargs="+", choices=PLANAR, default=["shoulder_lift"],
                        help="joints to MOVE (others hold at home). Default: shoulder_lift only")
    parser.add_argument("--start", type=float, default=0.0, help="start this many seconds into the plan")
    parser.add_argument("--seconds", type=float, help="play only this many seconds")
    parser.add_argument("--max-degrees", type=float, default=20.0,
                        help="keep each moving joint within this many degrees of straight up (default 20)")
    parser.add_argument("--execute", action="store_true", help="actually move the arm (default: dry run)")
    args = parser.parse_args(argv)
    if not 1 <= args.max_degrees <= 180:
        parser.error("--max-degrees must be 1..180")
    plan, first, last = load_plan(Path(args.plan), args.joints, args.start, args.seconds)
    load = apply_limits(plan, args.joints, first, last, args.max_degrees)
    print(summary(plan, args.joints, first, last, args.max_degrees, load))
    if load[0] > MAX_KG_CM:
        print(f"\nREFUSED: the shoulder would hold {load[0]:.1f} kg·cm, over the {MAX_KG_CM:g} kg·cm limit for these "
              "servos. Lower --max-degrees, or move elbow/wrist too so the arm folds instead of leaning straight.")
        return 2
    unchecked = [j for j in args.joints if not plan["direction_checked"].get(j)]
    if not args.execute:
        print("\nDRY RUN: nothing moved. Add --execute --port /dev/cu.… to move the arm.")
        return 0
    if not args.port:
        parser.error("--execute needs --port")
    if unchecked:
        print(f"\nWARNING: {', '.join(unchecked)} direction is not checked on the arm; it may bend the other way.")
    print("\nClear the arm's area, keep the 12 V plug in reach, and have a hand ready to support the arm.")
    if input("Type MOVE and press Enter to start: ").strip() != "MOVE":
        print("Not started.")
        return 1
    from .arm_web import ArmController
    from .leader_arm import make_bus
    controller = ArmController(make_bus(args.port), timeout=15.0)
    replay = Replay(controller, plan, args.joints, first, last)
    code = 0
    try:
        replay.start()
        replay.run()
    except KeyboardInterrupt:
        print("\nStopped: holding the arm where it is.")
        replay.hold()
    except RuntimeError as error:
        print(f"\nStopped: {error}")
        replay.hold()
        code = 1
    finally:
        try:
            input("Support the arm with your hand, then press Enter to turn the motors off... ")
        finally:
            replay.finish()
            print("Motors off.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
