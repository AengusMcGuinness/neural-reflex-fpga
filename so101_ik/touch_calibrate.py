#!/usr/bin/env python3
"""Calibrate the lift/elbow/wrist home offsets from pen touches on the writing surface.

The straight-up home pose was set by eye and is uncertain by tens of degrees at the elbow, which
moves the pen tip by centimetres.  Here the operator moves the arm BY HAND (torque off) so the pen
tip rests on the paper at several different spots; each pose is read from the servos.  The tip must
be at the surface height at every touch, so the three pitch offsets (and optionally the pen length)
are fitted by least squares.  The fitted home ticks are written back to the calibration and home
files (previous values kept).

    PYTHONPATH=.local/hiwonder-SoArm-101/src .venv-ik/bin/python -m so101_ik.touch_calibrate \
        --port /dev/ttyACM0 --points data/so101_ik/touch_points.json

Read-only on the bus: never enables torque.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from so101_ik.fk import JOINTS, build_so101_model, forward_kinematics
from so101_ik.write_arm import (CALIBRATION_FILE, HOME_FILE, TICKS_PER_RAD, load_calibration,
                                ticks_to_q)

PITCH = ("shoulder_lift", "elbow_flex", "wrist_flex")
PAN_AXIS_X_M = 0.0388353          # shoulder_pan joint origin in base_link (URDF)
PITCH_IDX = [JOINTS.index(j) for j in PITCH]


def read_ticks(bus) -> dict:
    from neural_reflex.leader_arm import inspect
    states = inspect(bus)
    return {j: int(states[j].position) for j in JOINTS}


def tip_xyz(ticks, cal, model, delta=np.zeros(3)):
    q = ticks_to_q([ticks[j] for j in JOINTS], cal)
    q = q.copy()
    for k, i in enumerate(PITCH_IDX):
        q[i] += delta[k]
    return forward_kinematics(q, model)[:3, 3]


def fit(points, cal, pen_len_m, plane_z_m, fit_pen=False, iters=50):
    """Gauss-Newton on residuals z_tip - plane_z (metres).  Unknowns: 3 pitch offsets (rad)
    [+ pen length (m)]."""
    x = np.zeros(4 if fit_pen else 3)
    if fit_pen:
        x[3] = pen_len_m

    def residuals(x):
        model = build_so101_model((0.0, 0.0, float(x[3]) if fit_pen else pen_len_m))
        out = []
        for p in points:
            xyz = tip_xyz(p["ticks"], cal, model, x[:3])
            out.append(xyz[2] - plane_z_m)
            # Optional ruler measurement: horizontal distance from the pan axis to the pen tip.
            # Height alone cannot separate the offsets when the wrist angle barely varies.
            if p.get("r_m") is not None:
                out.append(math.hypot(xyz[0] - PAN_AXIS_X_M, xyz[1]) - float(p["r_m"]))
        return np.array(out)

    r = residuals(x)
    for _ in range(iters):
        J = np.empty((len(points), len(x)))
        h = 1e-6
        for k in range(len(x)):
            dx = np.zeros_like(x); dx[k] = h
            J[:, k] = (residuals(x + dx) - r) / h
        step, *_ = np.linalg.lstsq(J.T @ J + 1e-9 * np.eye(len(x)), -J.T @ r, rcond=None)
        x = x + step
        r_new = residuals(x)
        if np.linalg.norm(r_new) > np.linalg.norm(r) * 1.0001 and np.linalg.norm(step) < 1e-9:
            break
        r = r_new
        if np.linalg.norm(step) < 1e-10:
            break
    return x, r


def apply(cal, home, x, fit_pen):
    """Shift home ticks so that q_true = q_model + delta.  Returns the new per-joint home ticks."""
    tpr = float(cal.get("ticks_per_rad", TICKS_PER_RAD))
    new = {}
    for k, j in enumerate(PITCH):
        c = cal["joints"][j]
        new[j] = int(round(c["home_tick"] - c["sign"] * float(x[k]) * tpr))
    return new


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", required=True)
    p.add_argument("--points", default="data/so101_ik/touch_points.json",
                   help="JSON file of touches; existing points are reused, new ones appended")
    p.add_argument("--plane_z_m", type=float, default=0.0, help="height of the writing surface in base_link")
    p.add_argument("--pen_len_m", type=float, default=0.05, help="pen tip below the handle frame along the tool axis")
    p.add_argument("--fit_pen", action="store_true", help="also fit the pen length (needs >= 5 varied touches)")
    p.add_argument("--min_points", type=int, default=4)
    p.add_argument("--write", action="store_true", help="write the fitted home ticks into the config files")
    p.add_argument("--no_read", action="store_true", help="fit the saved points only, do not open the arm")
    args = p.parse_args(argv)

    cal = load_calibration()
    pts_path = Path(args.points)
    points = json.loads(pts_path.read_text()) if pts_path.exists() else []
    print(f"{len(points)} saved touch point(s) in {pts_path}")

    if not args.no_read:
        from neural_reflex.leader_arm import make_bus
        bus = make_bus(args.port)
        bus.connect()
        try:
            while True:
                n = len(points) + 1
                ans = input(f"Touch {n}: rest the pen tip on the paper at a NEW spot (vary near/far/left/right), "
                            f"hold still, press Enter to read (or type done): ").strip().lower()
                if ans == "done":
                    break
                ticks = read_ticks(bus)
                time.sleep(0.2)
                again = read_ticks(bus)
                if any(abs(ticks[j] - again[j]) > 3 for j in JOINTS):
                    print("  the arm moved while reading; hold still and try again")
                    continue
                model = build_so101_model((0.0, 0.0, args.pen_len_m))
                xyz = tip_xyz(ticks, cal, model)
                print(f"  ticks {ticks}")
                print(f"  model tip before calibration: x {xyz[0]*1000:7.1f} y {xyz[1]*1000:7.1f} z {xyz[2]*1000:7.1f} mm")
                points.append({"ticks": ticks, "t": time.time()})
                pts_path.parent.mkdir(parents=True, exist_ok=True)
                pts_path.write_text(json.dumps(points, indent=1))
        finally:
            bus.disconnect(disable_torque=False)

    if len(points) < args.min_points:
        print(f"need at least {args.min_points} touches, have {len(points)}; nothing fitted")
        return 1

    x, r = fit(points, cal, args.pen_len_m, args.plane_z_m, fit_pen=args.fit_pen)
    model0 = build_so101_model((0.0, 0.0, args.pen_len_m))
    r0 = np.array([tip_xyz(pt["ticks"], cal, model0)[2] - args.plane_z_m for pt in points])
    print("\nFIT")
    print("  z residual before: " + " ".join(f"{v*1000:+6.1f}" for v in r0) + " mm")
    print("  residuals after (z, and ruler distance where given): "
          + " ".join(f"{v*1000:+6.1f}" for v in r) + " mm   (rms "
          f"{float(np.sqrt(np.mean(r**2)))*1000:.1f} mm)")
    n_r = sum(1 for pt in points if pt.get("r_m") is not None)
    print(f"  {len(points)} touches, {n_r} with a ruler distance")
    for k, j in enumerate(PITCH):
        print(f"  {j:14} offset {math.degrees(x[k]):+6.1f} deg")
    pen = float(x[3]) if args.fit_pen else args.pen_len_m
    if args.fit_pen:
        print(f"  pen length fitted: {pen*1000:.1f} mm")
    model1 = build_so101_model((0.0, 0.0, pen))
    print("  fitted tip positions (x, y mm) per touch, for a sanity check against where you put them:")
    for i, pt in enumerate(points, 1):
        xyz = tip_xyz(pt["ticks"], cal, model1, x[:3])
        print(f"    touch {i}: x {xyz[0]*1000:7.1f}  y {xyz[1]*1000:7.1f}  z {xyz[2]*1000:6.1f}")

    home = json.loads(Path(HOME_FILE).read_text())
    new_home = apply(cal, home, x, args.fit_pen)
    print("  new home ticks:", new_home, " (old:", {j: cal["joints"][j]["home_tick"] for j in PITCH}, ")")
    if args.write:
        for j, t in new_home.items():
            cal["joints"][j]["home_tick"] = t
            cal["joints"][j]["note"] = cal["joints"][j].get("note", "") + \
                f" Home tick refitted from {len(points)} pen touches on {time.strftime('%Y-%m-%d')} (so101_ik.touch_calibrate)."
        home.setdefault("previous_by_eye", dict(home["positions"]))
        for j, t in new_home.items():
            home["positions"][j] = t
        home["description"] += f" Lift/elbow/wrist refitted from {len(points)} pen touches ({time.strftime('%Y-%m-%d')})."
        Path(CALIBRATION_FILE).write_text(json.dumps(cal, indent=1) + "\n")
        Path(HOME_FILE).write_text(json.dumps(home, indent=2) + "\n")
        print(f"  written to {CALIBRATION_FILE} and {HOME_FILE}")
    else:
        print("  (dry run: add --write to store these)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
