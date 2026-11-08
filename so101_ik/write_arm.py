"""Turn a write_plan.py plan into servo ticks and (with --execute) drive the SO-ARM101 through it.

Dry run by default: converts every waypoint with config/so101_ik_calibration.json
(tick = home + sign * (q - q_home) * 4096 / 2pi), clamps to the motor limits minus the safety
margin (and the elbow to the 148 deg mechanical stop from config/air_arm.json) and REFUSES
(exit 2) when any waypoint would be clamped by more than max_clamp_ticks, then prints the
per-waypoint tick table, the peak elbow bend and the peak tick velocity against the 360 ticks/s cap.

--check --port PORT reads the arm without touching torque (like leader_arm probe): current ticks,
the URDF angles they map to, the FK pen tip and tool axis, the delta to the plan's first hover
point, the how_to_verify checklist, a simulated 'fine' approach to the first hover point (refused
if the pen tip would pass below the writing plane) and the straight-up start check.

--execute --port PORT runs the same check first and refuses to drive unless shoulder_lift,
elbow_flex and wrist_flex are within HOME_TOLERANCE of config/arm_home.json (--allow_any_start
overrides) and every joint in the calibration is marked verified (--i_verified_calibration
overrides).  It then drives the arm exactly the way neural_reflex/air_arm.py does, through
ArmController's safety (bounded ramp, lease heartbeat, tracking check): torque on per joint, the
approach to the first hover point at 'fine' speed, one target per joint per loop at --rate_hz,
heartbeat every loop, status every 0.25 s, pause and "support the arm" prompt on exit or Ctrl+C.
While the MOVE prompt waits, a keep-alive thread re-sends the hover targets so the controller's
inactivity timeout cannot release torque under a raised arm.  Between waypoints the targets are
interpolated in tick space so that no joint is asked to move faster than the chosen speed cap.
The gripper is never enabled.  Everything is logged to data/runs/write-<timestamp>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CALIBRATION_FILE = REPO / "config" / "so101_ik_calibration.json"
AIR_ARM_FILE = REPO / "config" / "air_arm.json"
HOME_FILE = REPO / "config" / "arm_home.json"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
WEIGHT_BEARING = ("shoulder_lift", "elbow_flex", "wrist_flex")   # must start straight up (air_arm PLANAR)
TICKS_PER_RAD = 4096.0 / (2.0 * math.pi)
TICKS_PER_DEGREE = 4096.0 / 360.0
HARD_CAP_TICKS_PER_S = 360.0
HOME_TOLERANCE = 150         # ticks (~13 deg) from straight up, as neural_reflex.air_arm.HOME_TOLERANCE
SETTLE_TICKS = 25            # "arrived" at the first hover point
SETTLE_TIMEOUT_S = 60.0
KEEPALIVE_PERIOD_S = 0.5     # re-send the hover targets this often while the MOVE prompt waits
APPROACH_SIM_DT_S = 0.02     # FK cadence of the simulated approach in --check


# ---------- calibration ---------------------------------------------------------------------------
def load_calibration(path=CALIBRATION_FILE) -> dict:
    cal = json.loads(Path(path).read_text())
    for j in JOINTS:
        if j not in cal["joints"]:
            raise ValueError(f"calibration has no entry for {j}")
    caps = cal.get("speed_caps_ticks_per_s", {})
    too_fast = {k: v for k, v in caps.items() if float(v) > HARD_CAP_TICKS_PER_S}
    if too_fast:
        raise ValueError(f"speed_caps_ticks_per_s above the {HARD_CAP_TICKS_PER_S:.0f} ticks/s hard cap "
                         f"in {path}: {too_fast}")
    return cal


def load_air_config(path=AIR_ARM_FILE) -> dict:
    return json.loads(Path(path).read_text())


def load_home(path=HOME_FILE) -> dict:
    return json.loads(Path(path).read_text())["positions"]


def q_to_ticks(q_rad, cal: dict) -> np.ndarray:
    """URDF radians (5,) -> float ticks (5,), no rounding or clamping."""
    q = np.asarray(q_rad, dtype=np.float64).flatten()
    tpr = float(cal.get("ticks_per_rad", TICKS_PER_RAD))
    out = np.empty(len(JOINTS))
    for i, j in enumerate(JOINTS):
        c = cal["joints"][j]
        out[i] = c["home_tick"] + c["sign"] * (q[i] - c["q_home_rad"]) * tpr
    return out


def ticks_to_q(ticks, cal: dict) -> np.ndarray:
    """Float or int ticks (5,) -> URDF radians (5,); exact inverse of q_to_ticks."""
    t = np.asarray(ticks, dtype=np.float64).flatten()
    tpr = float(cal.get("ticks_per_rad", TICKS_PER_RAD))
    out = np.empty(len(JOINTS))
    for i, j in enumerate(JOINTS):
        c = cal["joints"][j]
        out[i] = c["q_home_rad"] + (t[i] - c["home_tick"]) / (c["sign"] * tpr)
    return out


def elbow_bend_deg(q_elbow_rad, cal: dict) -> np.ndarray:
    """Elbow bend in degrees from the straight-up home, positive in the direction of increasing ticks
    (the direction air_arm.json's elbow_max_deg mechanical stop is measured in)."""
    c = cal["joints"]["elbow_flex"]
    return np.degrees(c["sign"] * (np.asarray(q_elbow_rad, dtype=np.float64) - c["q_home_rad"]))


def elbow_cap_ticks(cal: dict, elbow_max_deg: float) -> int:
    """Highest elbow tick allowed by the mechanical stop: home + elbow_max_deg * ticks_per_degree."""
    c = cal["joints"]["elbow_flex"]
    tpd = float(cal.get("ticks_per_degree", TICKS_PER_DEGREE))
    return int(math.floor(c["home_tick"] + float(elbow_max_deg) * tpd))


def tick_limits(cal: dict, elbow_max_deg: float | None = None) -> dict:
    """Motor limits minus the margin; the elbow's upper limit is also capped at the mechanical stop."""
    margin = int(cal.get("margin_ticks", 60))
    limits = {j: (int(cal["joints"][j]["limits_ticks"][0]) + margin,
                  int(cal["joints"][j]["limits_ticks"][1]) - margin) for j in JOINTS}
    if elbow_max_deg is None:
        elbow_max_deg = load_air_config().get("elbow_max_deg")
    if elbow_max_deg is not None:
        lo, hi = limits["elbow_flex"]
        limits["elbow_flex"] = (lo, min(hi, elbow_cap_ticks(cal, elbow_max_deg)))
    return limits


def plan_to_ticks(plan: dict, cal: dict, elbow_max_deg: float | None = None):
    """Return (ticks (N,5) int, clamp (N,5) int ticks removed by clamping, limits dict)."""
    limits = tick_limits(cal, elbow_max_deg)
    q = np.array([w["q_rad"] for w in plan["waypoints"]], dtype=np.float64).reshape(-1, len(JOINTS))
    raw = np.rint(q_to_ticks_batch(q, cal)).astype(int)
    lo = np.array([limits[j][0] for j in JOINTS])
    hi = np.array([limits[j][1] for j in JOINTS])
    clamped = np.clip(raw, lo, hi)
    return clamped, np.abs(raw - clamped), limits


def q_to_ticks_batch(q, cal: dict) -> np.ndarray:
    return np.array([q_to_ticks(row, cal) for row in np.asarray(q, dtype=np.float64).reshape(-1, len(JOINTS))])


def check_clamping(clamp: np.ndarray, cal: dict) -> list[str]:
    """Human-readable offences: joints clamped by more than max_clamp_ticks."""
    max_clamp = int(cal.get("max_clamp_ticks", 5))
    problems = []
    for i, j in enumerate(JOINTS):
        bad = np.flatnonzero(clamp[:, i] > max_clamp)
        if len(bad):
            problems.append(f"{j}: {len(bad)} waypoints clamped by up to {int(clamp[bad, i].max())} ticks "
                            f"(first at waypoint {int(bad[0])}, limit {max_clamp})")
    return problems


def peak_elbow_bend_deg(plan: dict, cal: dict) -> float:
    q_elbow = [w["q_rad"][JOINTS.index("elbow_flex")] for w in plan["waypoints"]]
    return float(np.max(elbow_bend_deg(q_elbow, cal))) if q_elbow else 0.0


# ---------- schedule ------------------------------------------------------------------------------
def tick_velocities(ticks: np.ndarray, rate_hz: float) -> np.ndarray:
    """Peak ticks/s per joint if one waypoint were sent per loop at rate_hz."""
    if len(ticks) < 2:
        return np.zeros(len(JOINTS))
    return np.abs(np.diff(ticks, axis=0)).max(axis=0) * rate_hz


def expand_schedule(ticks: np.ndarray, rate_hz: float, cap_ticks_per_s: float):
    """Interpolate in tick space so each loop step moves every joint <= cap/rate ticks.

    Returns (targets (M,5) int, waypoint_index (M,) int)."""
    per_step = max(1.0, cap_ticks_per_s / rate_hz)
    targets, owner = [ticks[0]], [0]
    for k in range(1, len(ticks)):
        delta = ticks[k] - ticks[k - 1]
        n = max(1, int(math.ceil(np.abs(delta).max() / per_step)))
        for s in range(1, n + 1):
            targets.append(np.rint(ticks[k - 1] + delta * (s / n)).astype(int))
            owner.append(k)
    return np.array(targets, dtype=int), np.array(owner, dtype=int)


def print_table(plan: dict, ticks: np.ndarray, clamp: np.ndarray, out=None, every: int = 1):
    out = out or sys.stdout
    header = f"{'#':>4} {'t[s]':>6} pen {'x[mm]':>7} {'y[mm]':>7} {'z[mm]':>6} | " + " ".join(f"{j[:9]:>9}" for j in JOINTS)
    out.write(header + "\n" + "-" * len(header) + "\n")
    for i, (w, row, c) in enumerate(zip(plan["waypoints"], ticks, clamp)):
        if i % every and i != len(ticks) - 1:
            continue
        x, y, z = (1000 * v for v in w["xyz"])
        mark = "*" if c.max() else " "
        out.write(f"{i:>4} {w['t']:>6.2f}  {'v' if w['pen_down'] else '^'}  {x:>7.1f} {y:>7.1f} {z:>6.1f} | "
                  + " ".join(f"{int(v):>9d}" for v in row) + mark + "\n")
    out.write("(v = pen down, ^ = pen up, * = clamped)\n")


# ---------- pre-flight check ----------------------------------------------------------------------
def read_arm_ticks(bus) -> dict:
    """Open the bus, read the six joints without touching torque, close it (leader_arm.run_probe)."""
    from neural_reflex.leader_arm import inspect
    bus.connect()
    try:
        states = inspect(bus)
    finally:
        bus.disconnect(disable_torque=False)
    return {name: {"position": s.position, "torque_enabled": s.torque_enabled, "operating_mode": s.operating_mode,
                   "minimum": s.minimum, "maximum": s.maximum} for name, s in states.items()}


def simulate_approach(start_ticks, target_ticks, cal: dict, model, cap_ticks_per_s: float,
                      dt: float = APPROACH_SIM_DT_S) -> tuple[float, int]:
    """Step every joint toward the target at the cap, FK the pen tip every dt.

    Returns (lowest pen-tip z [m], number of steps)."""
    from so101_ik.fk import forward_kinematics
    start = np.asarray(start_ticks, dtype=np.float64)
    target = np.asarray(target_ticks, dtype=np.float64)
    per_step = cap_ticks_per_s * dt
    delta = target - start
    n = int(math.ceil(np.abs(delta).max() / per_step)) if per_step > 0 else 0
    z_min = math.inf
    for s in range(n + 1):
        ticks = start + np.clip(delta, -per_step * s, per_step * s)
        z_min = min(z_min, float(forward_kinematics(ticks_to_q(ticks, cal), model)[2, 3]))
    return z_min, n


def arm_check(states: dict, plan: dict, cal: dict, home: dict, first_target, *, fine_cap: float,
              allow_any_start: bool = False, i_verified_calibration: bool = False, out=None) -> list[str]:
    """Print the pre-flight report for the arm as read (never writes to it). Returns the refusals."""
    from so101_ik.fk import build_so101_model, forward_kinematics
    out = out or sys.stdout
    problems = []
    current = np.array([states[j]["position"] for j in JOINTS], dtype=np.float64)
    home_t = np.array([home[j] for j in JOINTS], dtype=np.float64)
    first = np.asarray(first_target, dtype=np.float64)
    q = ticks_to_q(current, cal)
    offset = tuple(plan.get("params", {}).get("pen_tip_offset_m", (0.0, 0.0, 0.05)))
    model = build_so101_model(offset)
    T = forward_kinematics(q, model)
    plane_z = float(plan.get("params", {}).get("plane_z_m", 0.0))

    out.write("ARM CHECK (read-only; torque untouched)\n")
    out.write(f"{'joint':14} {'ticks':>6} {'torque':>6} {'q[rad]':>8} {'q[deg]':>7} {'home':>6} {'d_home':>7} "
              f"{'hover0':>7} {'d_hover':>8}\n")
    for i, j in enumerate(JOINTS):
        out.write(f"{j:14} {int(current[i]):6d} {'on' if states[j]['torque_enabled'] else 'off':>6} "
                  f"{q[i]:8.3f} {math.degrees(q[i]):7.1f} {int(home_t[i]):6d} {int(current[i] - home_t[i]):+7d} "
                  f"{int(first[i]):7d} {int(first[i] - current[i]):+8d}\n")
    g = states.get("gripper")
    if g is not None:
        out.write(f"{'gripper':14} {g['position']:6d} {'on' if g['torque_enabled'] else 'off':>6}   (never driven)\n")
    on = [j for j in JOINTS if states[j]["torque_enabled"]]
    if on:
        problems.append(f"torque is already on for {', '.join(on)}; turn it off before driving")
    tip = T[:3, 3] * 1000
    out.write(f"FK pen tip (offset {offset} m): x {tip[0]:.1f} y {tip[1]:.1f} z {tip[2]:.1f} mm; "
              f"tool z-axis in base frame ({T[0, 2]:+.3f}, {T[1, 2]:+.3f}, {T[2, 2]:+.3f})"
              f"  (straight up = (0, 0, +1), pen down = (0, 0, -1))\n")
    out.write(f"elbow bend now {float(elbow_bend_deg(q[JOINTS.index('elbow_flex')], cal)):.1f} deg from straight up\n")

    off = {j: current[JOINTS.index(j)] - home_t[JOINTS.index(j)] for j in WEIGHT_BEARING
           if abs(current[JOINTS.index(j)] - home_t[JOINTS.index(j)]) > HOME_TOLERANCE}
    if off:
        text = ("not straight up: " + ", ".join(f"{j} {d:+.0f} ticks ({d / TICKS_PER_DEGREE:+.0f} deg)"
                                                 for j, d in off.items())
                + f" from config/arm_home.json (tolerance {HOME_TOLERANCE} ticks)")
        if allow_any_start:
            out.write(f"WARNING (--allow_any_start): {text}\n")
        else:
            problems.append(text + "; put the arm straight up or pass --allow_any_start")
    else:
        out.write(f"start pose: shoulder_lift/elbow_flex/wrist_flex within {HOME_TOLERANCE} ticks of straight up\n")

    z_min, n_steps = simulate_approach(current, first, cal, model, fine_cap)
    out.write(f"simulated approach to the first hover point at 'fine' ({fine_cap:.0f} ticks/s): "
              f"{n_steps * APPROACH_SIM_DT_S:.1f} s, lowest pen tip z {z_min * 1000:.1f} mm "
              f"(plane z {plane_z * 1000:.1f} mm)\n")
    if z_min < plane_z:
        problems.append(f"the approach would push the pen tip {(plane_z - z_min) * 1000:.1f} mm below the writing "
                        f"plane; start from straight up or raise the plan")

    unverified = [j for j in JOINTS if not cal["joints"][j].get("verified")]
    out.write("calibration checklist (how_to_verify):\n")
    for j in JOINTS:
        c = cal["joints"][j]
        out.write(f"  [{'x' if c.get('verified') else ' '}] {j}: {c.get('how_to_verify', '(no instructions)')}\n")
    if unverified:
        text = f"calibration not verified on the arm for: {', '.join(unverified)}"
        if i_verified_calibration:
            out.write(f"WARNING (--i_verified_calibration): {text}\n")
        else:
            problems.append(text + "; do the checklist, set \"verified\": true, or pass --i_verified_calibration")
    return problems


# ---------- execution -----------------------------------------------------------------------------
class RunLog:
    def __init__(self, path):
        self.file = open(path, "a", buffering=1, encoding="utf-8") if path else None

    def write(self, **row):
        if self.file:
            row["t"] = round(time.monotonic(), 3)
            self.file.write(json.dumps({k: v for k, v in row.items() if v is not None}) + "\n")

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


def measured(controller) -> dict:
    snap = controller.status()
    if snap.get("fault"):
        raise RuntimeError(snap["fault"])
    return {j["name"]: j["position"] for j in snap["joints"]}


class KeepAlive:
    """Re-send the hover targets and a heartbeat every period so the controller's inactivity
    timeout cannot release torque while the MOVE prompt waits.  Errors are kept for the caller."""

    def __init__(self, controller, targets: dict, epoch: int, period: float = KEEPALIVE_PERIOD_S):
        self.controller, self.targets, self.epoch, self.period = controller, targets, epoch, period
        self.stop = threading.Event()
        self.error: Exception | None = None
        self.sent = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            try:
                for j, t in self.targets.items():
                    self.controller.target(j, t, self.epoch)
                snap = self.controller.heartbeat(self.epoch)
                if snap.get("fault"):
                    raise RuntimeError(snap["fault"])
                self.sent += 1
            except Exception as error:   # noqa: BLE001
                self.error = error
                return
            self.stop.wait(self.period)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=2.0)
        return False


def require_enabled(controller) -> dict:
    """status() -> positions; RuntimeError unless all five joints are torque-enabled and fault-free."""
    snap = controller.status()
    if snap.get("fault"):
        raise RuntimeError(snap["fault"])
    off = [j["name"] for j in snap["joints"] if j["name"] in JOINTS and not j.get("torque_enabled")]
    if off:
        raise RuntimeError(f"refusing to start: torque is not enabled on {', '.join(off)}")
    return {j["name"]: j["position"] for j in snap["joints"]}


def drive(controller, targets: np.ndarray, owner: np.ndarray, plan: dict, *, speed: str, rate_hz: float,
          start_pause_s: float, log: RunLog, ask=input, out=None):
    """Approach the first target at 'fine', hold until the user types MOVE, then run the schedule."""
    out = out or sys.stdout
    controller.set_speed("fine")
    stop = threading.Event()

    def motion_loop():
        while not stop.wait(0.02):
            controller.motion_step()

    threading.Thread(target=motion_loop, daemon=True).start()
    try:
        positions = measured(controller)
        log.write(event="start", positions=positions)
        start_pose = {j: int(positions[j]) for j in JOINTS}
        for j in JOINTS:                                  # never the gripper
            snap = controller.torque(j, True, controller.motion_epoch)
            if snap.get("fault"):
                raise RuntimeError(snap["fault"])
        epoch = controller.motion_epoch
        first = {j: int(targets[0][i]) for i, j in enumerate(JOINTS)}
        out.write(f"moving to the first hover point at 'fine' speed: {first}\n")
        t0 = time.monotonic()
        while True:
            for j, t in first.items():
                controller.target(j, t, epoch)
            controller.heartbeat(epoch)
            positions = measured(controller)
            log.write(event="approach", step=0, waypoint=0, targets=first, positions=positions)
            worst = max(abs(positions[j] - first[j]) for j in JOINTS)
            out.write(f"\r  approaching: worst joint {worst:4d} ticks away   ")
            out.flush()
            if worst <= SETTLE_TICKS:
                break
            if time.monotonic() - t0 > SETTLE_TIMEOUT_S:
                raise RuntimeError("the arm did not reach the first hover point in time")
            time.sleep(0.25)
        out.write(f"\narrived; holding at the first hover point for {start_pause_s:.0f} s...\n")
        t1 = time.monotonic()
        while time.monotonic() - t1 < start_pause_s:
            controller.heartbeat(epoch)
            time.sleep(0.2)
        # Torque stays on while the prompt waits: the keep-alive refreshes the inactivity deadline.
        with KeepAlive(controller, first, epoch) as keepalive:
            answer = ask("Check the pen height above the surface. Type MOVE and press Enter to write: ").strip()
        if keepalive.error is not None:
            raise RuntimeError(f"lost the arm while waiting at the prompt: {keepalive.error}")
        if answer != "MOVE":
            out.write("Not started.\n")
            log.write(event="not started")
            return False
        positions = require_enabled(controller)
        log.write(event="writing", n_steps=int(len(targets)), positions=positions)
        controller.set_speed(speed)
        out.write(f"writing at '{speed}' speed\n")
        period = 1.0 / rate_hz
        next_status = 0.0
        pen_state = None
        for step, (row, k) in enumerate(zip(targets, owner)):
            now = time.monotonic()
            tgt = {j: int(row[i]) for i, j in enumerate(JOINTS)}
            for j, t in tgt.items():
                controller.target(j, t, epoch)
            controller.heartbeat(epoch)
            if now >= next_status:
                next_status = now + 0.25
                positions = measured(controller)
            pen = plan["waypoints"][int(k)]["pen_down"]
            if pen != pen_state:
                pen_state = pen
                out.write(f"\r  step {step:5d}/{len(targets)} waypoint {int(k):4d}  pen {'DOWN' if pen else 'up  '}   ")
                out.flush()
            log.write(step=step, waypoint=int(k), pen_down=pen, targets=tgt, positions=positions)
            time.sleep(max(0.0, period - (time.monotonic() - now)))
        # let the ramp finish the last move
        t2 = time.monotonic()
        while time.monotonic() - t2 < 2.0:
            controller.heartbeat(epoch)
            time.sleep(0.2)
        positions = measured(controller)
        log.write(event="done", positions=positions)
        out.write("\nfinished writing.\n")
        # Park back at the pose the run started from (upright) before torque is released, so the
        # arm is not left outstretched over the table when the motors go limp.
        controller.set_speed("fine")
        out.write(f"parking back at the start pose at 'fine' speed: {start_pose}\n")
        # Two stages: first bring the shoulder/pan/wrist home while the elbow stays bent (the band
        # helps the shoulder), then unbend the elbow. Unbending a bent elbow while the shoulder is
        # still forward stalled the elbow servo on a live run.
        stages = [{j: t for j, t in start_pose.items() if j != "elbow_flex"}, dict(start_pose)]
        for stage_no, stage in enumerate(stages, 1):
            t3 = time.monotonic()
            while True:
                for j, t in stage.items():
                    controller.target(j, t, epoch)
                controller.heartbeat(epoch)
                positions = measured(controller)
                log.write(event="park", stage=stage_no, targets=stage, positions=positions)
                worst = max(abs(positions[j] - stage[j]) for j in stage)
                out.write(f"\r  parking stage {stage_no}/2: worst joint {worst:4d} ticks away   ")
                out.flush()
                if worst <= SETTLE_TICKS or time.monotonic() - t3 > SETTLE_TIMEOUT_S:
                    break
                time.sleep(0.25)
        out.write("\nparked.\n")
        return True
    finally:
        stop.set()


# ---------- CLI -----------------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--plan", required=True, help="plan.json from so101_ik.write_plan")
    p.add_argument("--calibration", default=str(CALIBRATION_FILE))
    p.add_argument("--air_config", default=str(AIR_ARM_FILE), help="config with elbow_max_deg (mechanical stop)")
    p.add_argument("--home", default=str(HOME_FILE), help="recorded straight-up pose")
    p.add_argument("--check", action="store_true", help="read the arm (no torque) and print the pre-flight report")
    p.add_argument("--execute", action="store_true", help="drive the arm (default: dry run)")
    p.add_argument("--port", help="arm USB serial port (needed with --check / --execute)")
    p.add_argument("--speed", choices=("fine", "normal", "fast"), default="normal")
    p.add_argument("--rate_hz", type=float, default=20.0, help="target updates per second")
    p.add_argument("--start_pause_s", type=float, default=3.0, help="hold at the first hover point before the MOVE prompt")
    p.add_argument("--countdown_s", type=float, default=0.0,
                   help="hands-free: replace the MOVE prompts with a countdown of this many seconds (0 = prompt)")
    p.add_argument("--allow_any_start", action="store_true",
                   help=f"drive even if the arm is more than {HOME_TOLERANCE} ticks from straight up")
    p.add_argument("--i_verified_calibration", action="store_true",
                   help="drive even though joints in the calibration are marked \"verified\": false")
    p.add_argument("--every", type=int, default=1, help="print every n-th waypoint in the dry-run table")
    p.add_argument("--log", help="run log (default data/runs/write-<timestamp>.jsonl)")
    args = p.parse_args(argv)

    plan = json.loads(Path(args.plan).read_text())
    cal = load_calibration(args.calibration)
    air = load_air_config(args.air_config)
    home = load_home(args.home)
    elbow_max_deg = air.get("elbow_max_deg")
    if not plan.get("waypoints"):
        print("the plan has no waypoints")
        return 2
    ticks, clamp, limits = plan_to_ticks(plan, cal, elbow_max_deg)
    problems = check_clamping(clamp, cal)
    caps = cal.get("speed_caps_ticks_per_s", {"fine": 60, "normal": 180, "fast": 360})
    cap = float(caps[args.speed])
    fine_cap = float(caps["fine"])
    raw_vel = tick_velocities(ticks, args.rate_hz)
    targets, owner = expand_schedule(ticks, args.rate_hz, cap)
    peak_elbow = peak_elbow_bend_deg(plan, cal)

    print(f"plan '{plan['text']}': {len(ticks)} waypoints, solver {plan.get('summary', {}).get('solver')}, "
          f"calibration {args.calibration}")
    unverified = [j for j in JOINTS if not cal["joints"][j].get("verified")]
    if unverified:
        print(f"WARNING: tick direction/offset NOT verified on the arm for: {', '.join(unverified)} "
              f"(see how_to_verify in the calibration file)")
    print_table(plan, ticks, clamp, every=max(1, args.every))
    print("tick limits (with margin): " + ", ".join(f"{j} {lo}..{hi}" for j, (lo, hi) in limits.items()))
    if elbow_max_deg is not None:
        print(f"elbow cap: {elbow_cap_ticks(cal, elbow_max_deg)} ticks = home + {elbow_max_deg} deg mechanical stop "
              f"({args.air_config}); plan's peak elbow bend {peak_elbow:.1f} deg")
    else:
        print(f"plan's peak elbow bend {peak_elbow:.1f} deg (no elbow_max_deg in {args.air_config})")
    print("peak tick velocity if one waypoint per loop at %.0f Hz: " % args.rate_hz
          + ", ".join(f"{j} {v:.0f}" for j, v in zip(JOINTS, raw_vel))
          + f" ticks/s  (hard cap {HARD_CAP_TICKS_PER_S:.0f}, '{args.speed}' cap {cap:.0f})")
    print(f"schedule: {len(targets)} steps at {args.rate_hz:.0f} Hz = {len(targets) / args.rate_hz:.1f} s "
          f"after interpolating to <= {cap:.0f} ticks/s ({(raw_vel > cap).sum()} joints needed slowing); "
          f"approach to the first hover point at 'fine' ({fine_cap:.0f} ticks/s)")
    if problems:
        print("REFUSED: the plan needs clamping beyond the allowed tolerance:")
        for line in problems:
            print("  " + line)
        print("Move the text (distance/height), shorten the pen, or fix the calibration sign/offset.")
        return 2
    if not args.check and not args.execute:
        print("DRY RUN: nothing moved. Add --check --port PORT to read the arm, --execute --port PORT to drive it.")
        return 0
    if not args.port:
        p.error("--check / --execute need --port")

    from neural_reflex.leader_arm import make_bus
    print()
    states = read_arm_ticks(make_bus(args.port))
    refusals = arm_check(states, plan, cal, home, targets[0], fine_cap=fine_cap,
                         allow_any_start=args.allow_any_start,
                         i_verified_calibration=args.i_verified_calibration)
    if refusals:
        print("CHECK FAILED:" if not args.execute else "REFUSED to execute:")
        for line in refusals:
            print("  " + line)
        return 2
    print("CHECK PASSED.")
    if not args.execute:
        print("Nothing moved. Add --execute to drive the arm.")
        return 0

    runs = REPO / "data" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    log_path = args.log or str(runs / time.strftime("write-%Y%m%d-%H%M%S.jsonl"))
    print(f"run log: {log_path}")
    log = RunLog(log_path)
    log.write(event="plan", plan=str(args.plan), text=plan["text"], speed=args.speed, rate_hz=args.rate_hz,
              n_waypoints=int(len(ticks)), n_steps=int(len(targets)), start_positions=states,
              peak_elbow_bend_deg=round(peak_elbow, 2))
    print("\nThe arm will move to the first hover point at 'fine' speed, hold, then write. Pen fitted, surface clear.")
    if args.countdown_s:
        # Hands-free mode: every prompt becomes a printed countdown; Ctrl+C still stops the run.
        def ask(prompt: str) -> str:
            print(prompt.replace("Type MOVE and press Enter", f"starting in {args.countdown_s:.0f} s (Ctrl+C aborts)"))
            for remaining in range(int(args.countdown_s), 0, -1):
                print(f"  {remaining}...", flush=True)
                time.sleep(1.0)
            return "MOVE"
    else:
        ask = input
    if ask("Type MOVE and press Enter to power the motors: ").strip() != "MOVE":
        print("Not started.")
        return 1
    from neural_reflex.arm_web import ArmController
    controller = ArmController(make_bus(args.port), timeout=15.0)
    code = 0
    try:
        ok = drive(controller, targets, owner, plan, speed=args.speed, rate_hz=args.rate_hz,
                   start_pause_s=args.start_pause_s, log=log, ask=ask)
        code = 0 if ok else 1
    except KeyboardInterrupt:
        print("\nStopped: holding the arm where it is.")
        log.write(event="stopped by Ctrl+C")
    except (RuntimeError, ValueError) as error:   # ValueError: ArmController's out-of-bounds target
        print(f"\nStopped: {error}")
        log.write(event=f"stopped: {error}")
        code = 1
    finally:
        try:
            controller.pause(controller.motion_epoch)
        except Exception:   # noqa: BLE001
            pass
        try:
            if args.countdown_s and code == 0:
                # The run parked the arm upright, so a timed release is safe; a stopped run still waits.
                print(f"\nParked. Motors turn off in {args.countdown_s:.0f} s; support the arm if it is not upright.")
                time.sleep(float(args.countdown_s))
            else:
                input("\nSupport the arm with your hand, then press Enter to turn the motors off... ")
        finally:
            controller.close()
            log.close()
            print("Motors off.")
    return code


if __name__ == "__main__":
    sys.exit(main())
