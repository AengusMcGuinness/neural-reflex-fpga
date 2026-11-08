"""Simulate picking an item off the table with the SO-ARM101, in one vertical plane.

The base stays at home and the item sits straight in front of the robot, so only shoulder
lift, elbow, wrist flex and the gripper move. Geometry and link masses come from the official
SO-101 model (TheRobotStudio/SO-ARM100, so101_new_calib.urdf); this leader arm's handle and
lighter servos make the numbers approximate.

Angles are degrees from the straight-up home pose, positive = bending FORWARD (toward the item),
measured from vertical: shoulder 90 means the upper arm is level.

For every distance along the table it checks: can the gripper reach the item from above,
does every joint stay inside its limits, does the arm stay clear of the table, and can the
leader arm's 10 kg-cm servos hold the pose. Then it plans one pick (home -> above -> down ->
close -> lift -> home), simulates the controller's speed limits, and writes
pick_plan.json and pick_sim.html. Nothing talks to the arm.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .dashboard import load_home
from .robot_map import follow, load_map

REPO = Path(__file__).resolve().parent.parent
VIEWER = REPO / "web" / "pick_sim" / "viewer.html"

# --- Geometry (metres): config/robot_geometry.json, defaulting to the SO-101 URDF -----------
GEOMETRY_FILE = REPO / "config" / "robot_geometry.json"
_URDF = {"shoulder_height_m": 0.117,    # base plate to the shoulder-lift axis (0.0624 + 0.0542)
         "shoulder_forward_m": 0.030,   # shoulder-lift axis in front of the base-rotation axis
         "upper_arm_m": 0.116,          # shoulder -> elbow (0.1126 along, 0.028 offset)
         "forearm_m": 0.135,            # elbow -> wrist
         "hand_m": 0.159}               # wrist -> gripper tip (0.061 + 0.098)


def load_geometry(path: Path = GEOMETRY_FILE) -> dict:
    geometry = dict(_URDF)
    if path.exists():
        data = json.loads(path.read_text())
        geometry.update({k: float(data[k]) for k in _URDF if k in data})
    for key, value in geometry.items():
        if not 0.0 <= value <= 0.5:
            raise ValueError(f"{path}: {key} = {value} m is not a plausible arm dimension")
    return geometry


_G = load_geometry()
SHOULDER_HEIGHT = _G["shoulder_height_m"]
SHOULDER_FORWARD = _G["shoulder_forward_m"]
UPPER_ARM = _G["upper_arm_m"]
FOREARM = _G["forearm_m"]
HAND = _G["hand_m"]
LINKS = (UPPER_ARM, FOREARM, HAND)
GEOMETRY = {"shoulder_height": SHOULDER_HEIGHT, "shoulder_forward": SHOULDER_FORWARD, "links": list(LINKS)}

# --- Masses (kg) and centre of mass as a fraction along each link, from the URDF inertials --
MASSES = (
    (0, 0.103, 0.090 / UPPER_ARM),         # upper arm
    (1, 0.104, 0.098 / FOREARM),           # forearm
    (2, 0.079, 0.039 / HAND),              # wrist motor
    (2, 0.099, 0.086 / HAND),              # gripper + jaw
)
STALL_KG_CM = 10.0          # HX-10HM stall torque at 11.1 V
SAFE_KG_CM = 4.0            # hold under 40% of stall
MAX_KG_CM = 6.0             # never plan above 60%

TABLE_CLEARANCE = 0.015     # elbow and wrist must stay this far above the table
TICKS_PER_DEGREE = 4096 / 360
# Which tick direction bends each joint forward. Only shoulder lift is checked on the arm.
FORWARD_TICK_SIGN = {"shoulder_lift": +1, "elbow_flex": +1, "wrist_flex": +1}
DIRECTION_CHECKED = {"shoulder_lift": True, "elbow_flex": False, "wrist_flex": False}
PLANAR = ("shoulder_lift", "elbow_flex", "wrist_flex")
GRIPPER_CLOSE_TICKS = 500   # robot_map gripper gain; direction unchecked
GRIP_CLOSE_S = 3.0          # 500 ticks with a 1.875x peak stays under the 360 ticks/s cap


def joint_limits(config: dict, home: dict) -> dict:
    """Degrees from home the robot may use, keeping the map's safety margin."""
    margin = config["margin_ticks"]
    out = {}
    for joint in PLANAR:
        lo, hi = config["limits"][joint]
        a = (lo + margin - home[joint]) / TICKS_PER_DEGREE * FORWARD_TICK_SIGN[joint]
        b = (hi - margin - home[joint]) / TICKS_PER_DEGREE * FORWARD_TICK_SIGN[joint]
        out[joint] = (min(a, b), max(a, b))
    return out


def forward_kinematics(angles) -> np.ndarray:
    """Shoulder, elbow, wrist and tip positions (x forward, z up) for (lift, elbow, wrist) degrees."""
    points = [np.array([SHOULDER_FORWARD, SHOULDER_HEIGHT])]
    heading = 0.0
    for length, angle in zip(LINKS, angles):
        heading += math.radians(angle)
        points.append(points[-1] + length * np.array([math.sin(heading), math.cos(heading)]))
    return np.array(points)


def gravity_torque(angles) -> dict:
    """kg-cm each motor must hold (shoulder carries everything, elbow carries forearm onward)."""
    points = forward_kinematics(angles)
    torque = {"shoulder_lift": 0.0, "elbow_flex": 0.0}
    for link, mass, fraction in MASSES:
        com_x = points[link][0] + fraction * (points[link + 1][0] - points[link][0])
        torque["shoulder_lift"] += mass * abs(com_x - points[0][0]) * 100
        if link >= 1:
            torque["elbow_flex"] += mass * abs(com_x - points[1][0]) * 100
    return torque


def inverse_kinematics(x: float, z: float, pitch: float):
    """Joint angles putting the tip at (x, z) with the hand at `pitch` degrees from vertical
    (180 = pointing straight down). Elbow bends forward. None if out of reach."""
    p = math.radians(pitch)
    wx = x - HAND * math.sin(p) - SHOULDER_FORWARD
    wz = z - HAND * math.cos(p) - SHOULDER_HEIGHT
    d2 = wx * wx + wz * wz
    cos_elbow = (d2 - UPPER_ARM ** 2 - FOREARM ** 2) / (2 * UPPER_ARM * FOREARM)
    if not -1.0 <= cos_elbow <= 1.0:
        return None
    elbow = math.acos(cos_elbow)                                   # > 0: forward bend
    shoulder = math.atan2(wx, wz) - math.atan2(FOREARM * math.sin(elbow), UPPER_ARM + FOREARM * math.cos(elbow))
    lift, bend = math.degrees(shoulder), math.degrees(elbow)
    return np.array([lift, bend, pitch - lift - bend])


def check_pose(angles, limits, item_top: float | None = None) -> list[str]:
    """Reasons a pose is not allowed (empty list = fine)."""
    problems = []
    for joint, value in zip(PLANAR, angles):
        lo, hi = limits[joint]
        if not lo - 1e-6 <= value <= hi + 1e-6:
            problems.append(f"{joint} {value:.0f}° outside {lo:.0f}..{hi:.0f}°")
    points = forward_kinematics(angles)
    if min(points[1][1], points[2][1]) < TABLE_CLEARANCE:
        problems.append("elbow or wrist would hit the table")
    torque = gravity_torque(angles)
    if torque["shoulder_lift"] > MAX_KG_CM:
        problems.append(f"shoulder needs {torque['shoulder_lift']:.1f} kg·cm (limit {MAX_KG_CM:g})")
    return problems


def best_line(x: float, z0: float, z1: float, limits, samples: int = 9):
    """Most vertical gripper angle for which the whole straight vertical path from z0 to z1
    at distance x is allowed. Returns (pitch, grasp angles, reasons); pitch None if none works."""
    reasons = []
    for pitch in range(180, 104, -5):
        ok, grasp = True, None
        for z in np.linspace(z0, z1, samples):
            angles = inverse_kinematics(x, z, pitch)
            problems = ["out of reach"] if angles is None else check_pose(angles, limits)
            if problems:
                ok, reasons = False, problems
                break
            grasp = angles if grasp is None else grasp
        if ok:
            return float(pitch), grasp, []
    return None, None, reasons


def reach_map(limits, item_height: float, lift_height: float, step: float = 0.005) -> list[dict]:
    """For each distance from the base: can the robot grasp the item there and lift it straight up?"""
    rows = []
    grasp_z = item_height * 0.6
    for x in np.arange(0.05, 0.46, step):
        pitch, grasp, why = best_line(x, grasp_z, grasp_z + lift_height, limits)
        row = {"distance_m": round(float(x), 3), "reason": "; ".join(why)}
        if pitch is None:
            row["status"] = "unreachable"
        else:
            above = inverse_kinematics(x, grasp_z + lift_height, pitch)
            worst = max(gravity_torque(grasp)["shoulder_lift"], gravity_torque(above)["shoulder_lift"])
            row.update(status="safe" if worst <= SAFE_KG_CM else "heavy", pitch=pitch,
                       grasp=[round(float(a), 1) for a in grasp], shoulder_kg_cm=round(worst, 2))
        rows.append(row)
    return rows


def smoothstep(n: int) -> np.ndarray:
    s = np.linspace(0, 1, n)
    return s * s * s * (s * (6 * s - 15) + 10)                      # minimum-jerk profile


def plan_pick(x: float, item_height: float, lift_height: float, limits, speed: float, dt: float = 0.02):
    """Joint trajectory for one pick. Down and up moves follow a straight vertical line."""
    grasp_z = item_height * 0.6
    pitch, grasp, why = best_line(x, grasp_z, grasp_z + lift_height, limits)
    if pitch is None:
        raise ValueError(f"cannot pick at {x * 100:.0f} cm: {'; '.join(why)}")
    home = np.zeros(3)

    def joint_move(start, end, label, grip):
        # A minimum-jerk move peaks at 1.875x its average speed; `speed` is the peak.
        n = max(2, int(math.ceil(np.max(np.abs(end - start)) * 1.875 / speed / dt)) + 1)
        return [(start + (end - start) * s, grip, label) for s in smoothstep(n)]

    def vertical_move(z0, z1, label, grip):
        n = max(2, int(math.ceil(abs(z1 - z0) / 0.03 / dt)) + 1)    # 3 cm/s
        out = []
        for s in smoothstep(n):
            angles = inverse_kinematics(x, z0 + (z1 - z0) * s, pitch)
            if angles is None:
                raise ValueError("straight-down path leaves the reachable area")
            out.append((angles, grip, label))
        return out

    def hold(angles, seconds, label, grip):
        return [(angles, grip, label)] * int(seconds / dt)

    above = inverse_kinematics(x, grasp_z + lift_height, pitch)
    steps = []
    steps += hold(home, 1.0, "home", 0.0)
    steps += joint_move(home, above, "move above the item", 0.0)
    steps += vertical_move(grasp_z + lift_height, grasp_z, "lower onto the item", 0.0)
    steps += [(grasp, g, "close the gripper") for g in smoothstep(int(GRIP_CLOSE_S / dt))]
    steps += hold(grasp, 0.5, "hold", 1.0)
    steps += vertical_move(grasp_z, grasp_z + lift_height, "lift the item", 1.0)
    steps += hold(above, 0.5, "hold", 1.0)
    steps += joint_move(above, home, "return home", 1.0)
    steps += hold(home, 1.0, "home", 1.0)
    angles = np.array([s[0] for s in steps])
    grip = np.array([s[1] for s in steps])
    labels = [s[2] for s in steps]
    problems = sorted({p.split(" ")[0] + (" past its limit" if "outside" in p else " hits the table")
                       for a in angles for p in check_pose(a, limits) if "outside" in p or "table" in p})
    torque = np.array([[gravity_torque(a)[j] for j in ("shoulder_lift", "elbow_flex")] for a in angles])
    return {"angles": angles, "grip": grip, "labels": labels, "pitch": pitch, "problems": problems,
            "torque": torque, "dt": dt}


def to_ticks(plan: dict, home: dict) -> dict:
    ticks = {}
    for i, joint in enumerate(PLANAR):
        ticks[joint] = home[joint] + FORWARD_TICK_SIGN[joint] * plan["angles"][:, i] * TICKS_PER_DEGREE
    ticks["gripper"] = home["gripper"] + GRIPPER_CLOSE_TICKS * plan["grip"]
    for joint in ("shoulder_pan", "wrist_roll"):
        ticks[joint] = np.full(len(plan["grip"]), float(home[joint]))      # one plane: held at home
    return ticks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--distance", type=float, help="item distance from the base axis, metres "
                                                       "(default: middle of the safe zone)")
    parser.add_argument("--item-height", type=float, default=0.05, help="item height, metres")
    parser.add_argument("--lift", type=float, default=0.08, help="how high to lift the item, metres")
    parser.add_argument("--speed", type=float, default=24.0, help="peak joint speed for free moves, deg/s (cap 32)")
    parser.add_argument("--output", default=str(REPO / "data" / "sim"), help="folder for the plan and preview")
    args = parser.parse_args(argv)
    if not 1 <= args.speed <= 30:
        parser.error("--speed must be 1..30 deg/s (the motors cap near 32)")
    config, home = load_map(), load_home()
    limits = joint_limits(config, home)
    rows = reach_map(limits, args.item_height, args.lift)
    safe = [r["distance_m"] for r in rows if r["status"] == "safe"]
    heavy = [r["distance_m"] for r in rows if r["status"] == "heavy"]
    print("Joint limits used (deg from straight up, forward +): " +
          ", ".join(f"{j} {lo:.0f}..{hi:.0f}" for j, (lo, hi) in limits.items()))
    print(f"Item {args.item_height * 100:.0f} cm tall, lifted {args.lift * 100:.0f} cm:")
    if safe:
        print(f"  safe to pick:        {min(safe) * 100:.1f} – {max(safe) * 100:.1f} cm from the base")
    if heavy:
        print(f"  reachable but heavy: {min(heavy) * 100:.1f} – {max(heavy) * 100:.1f} cm "
              f"(shoulder over {SAFE_KG_CM:g} kg·cm)")
    if not safe:
        raise SystemExit("No safe pick distance for this item height and lift.")
    distance = args.distance if args.distance else safe[len(safe) // 2]
    plan = plan_pick(distance, args.item_height, args.lift, limits, args.speed)
    ticks = to_ticks(plan, home)
    cap = config["speed_caps_ticks_per_s"]["fast"]
    executed = {j: follow(ticks[j], home[j], cap, config["acceleration_ticks_per_s2"], plan["dt"]) for j in ticks}
    lag = {j: float(np.max(np.abs(executed[j] - ticks[j])) / TICKS_PER_DEGREE) for j in ticks}
    peak = plan["torque"].max(axis=0)
    print(f"\nPick at {distance * 100:.1f} cm, gripper {180 - plan['pitch']:.0f}° from straight down, "
          f"{len(plan['grip']) * plan['dt']:.1f} s:")
    print(f"  peak hold: shoulder {peak[0]:.1f} kg·cm, elbow {peak[1]:.1f} kg·cm (stall {STALL_KG_CM:g})")
    print("  worst lag behind plan at fast speed: " + ", ".join(f"{j} {lag[j]:.1f}°" for j in (*PLANAR, "gripper")))
    print("  problems: " + ("; ".join(plan["problems"]) if plan["problems"] else "none"))
    unchecked = [j for j, ok in DIRECTION_CHECKED.items() if not ok]
    print(f"  directions not yet checked on the arm: {', '.join(unchecked)}, gripper")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "pick_plan.json").write_text(json.dumps({
        "distance_m": distance, "item_height_m": args.item_height, "lift_m": args.lift, "dt_s": plan["dt"],
        "pitch_deg": plan["pitch"], "home": home, "labels": plan["labels"],
        "targets": {j: [int(round(v)) for v in ticks[j]] for j in ticks}}) + "\n")
    step = 2
    preview = {
        "geometry": {"shoulder_height": SHOULDER_HEIGHT, "shoulder_forward": SHOULDER_FORWARD, "links": LINKS},
        "item": {"distance": distance, "height": args.item_height, "width": 0.04},
        "dt": plan["dt"] * step, "labels": plan["labels"][::step], "grip": plan["grip"][::step].round(2).tolist(),
        "angles": plan["angles"][::step].round(1).tolist(),
        "torque": plan["torque"][::step].round(2).tolist(),
        "limits": {j: list(v) for j, v in limits.items()},
        "executed_deg": {j: ((executed[j][::step] - home[j]) / TICKS_PER_DEGREE * FORWARD_TICK_SIGN[j]).round(1).tolist()
                         for j in PLANAR},
        "reach": rows, "safe_kg_cm": SAFE_KG_CM, "max_kg_cm": MAX_KG_CM, "stall_kg_cm": STALL_KG_CM,
        "checked": DIRECTION_CHECKED, "pitch": plan["pitch"], "problems": plan["problems"]}
    page = VIEWER.read_text(encoding="utf-8").replace(
        "__PICK_DATA__", json.dumps(preview, separators=(",", ":")).replace("</", "<\\/"))
    (output / "pick_sim.html").write_text(page, encoding="utf-8")
    print(f"\nPlan: {output / 'pick_plan.json'}\nPreview: {output / 'pick_sim.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
