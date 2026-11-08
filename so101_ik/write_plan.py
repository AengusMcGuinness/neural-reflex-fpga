"""Plan a handwriting path for the SO-101 (dry run only: this never touches the arm).

Frame conventions (URDF base_link, metres):
  +x  the arm's forward direction (at q = 0 the forearm and gripper point along +x)
  +z  up; the bottom of the base is z = 0, so a surface the base is clamped to is plane_z = 0
Text is centred at (distance_m, 0, plane_z_m), runs left to right as seen by someone standing
behind the robot looking forward (left = +y), with the cap height pointing away from the base (+x).

For every stroke the pen hovers above its first point, descends, traces it and lifts.  Each
waypoint's orientation target is "pen pointing straight down" (tool z = -z of the base); wrist
roll is held at START_Q's value (``task="pen"`` zeroes its Jacobian column, see newton.py).
Waypoints whose elbow bend would exceed config/air_arm.json's elbow_max_deg mechanical stop are
rejected at planning time.  Waypoints are solved one after another, each seeded with the
previous solution, by IKSolver.solve_refined (DNN seed + Newton polish) when --model_dir holds
a trained model, otherwise by the damped Newton polish alone.

Usage:
  python -m so101_ik.write_plan --text "hello Neo" --distance_m 0.20 --plane_z_m 0.0 \
      --model_dir so101_ik/results --out plan.json --preview plan.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from so101_ik.fk import JOINT_LIMITS, JOINTS, N_JOINTS, build_so101_model, forward_kinematics
from so101_ik.text_strokes import (DEFAULT_FONT, DEFAULT_HEIGHT_M, MAX_SPACING_M, strokes_bbox,
                                   strokes_length, text_to_strokes)

TABLE_Z_M = 0.0                        # bottom of base_link, where the base is clamped
DEFAULT_PEN_TIP_OFFSET_M = (0.0, 0.0, 0.05)
# Wrist roll is held at its seed for the whole plan (newton.py zeroes its Jacobian column for the
# pen task), so seed it in the middle of the asymmetric URDF roll range, as far from both stops as
# possible.
ROLL_MID_RAD = float((JOINT_LIMITS[4][0] + JOINT_LIMITS[4][1]) / 2.0)
# Arm folded forward with the pen pointing down-ish: a sensible seed for the first waypoint.
START_Q = np.array([0.0, 0.0, 0.5, 1.3, ROLL_MID_RAD])
ELBOW_INDEX = JOINTS.index("elbow_flex")


class ElbowStop:
    """The elbow's mechanical stop (config/air_arm.json elbow_max_deg) in URDF radians, through the
    tick calibration's sign/offset, so plans that write_arm would refuse are rejected here."""

    def __init__(self, max_deg=None, cal=None):
        from so101_ik.write_arm import elbow_bend_deg, load_air_config, load_calibration
        self.max_deg = load_air_config().get("elbow_max_deg") if max_deg is None else max_deg
        self.cal = load_calibration() if cal is None else cal
        self._bend = elbow_bend_deg

    def bend_deg(self, q) -> float:
        return float(self._bend(np.asarray(q, dtype=np.float64).flatten()[ELBOW_INDEX], self.cal))

    def ok(self, q) -> bool:
        return self.max_deg is None or self.bend_deg(q) <= float(self.max_deg)


def calibrated_limits(cal=None, elbow_max_deg=None):
    """Joint limits in URDF radians that the physical arm can actually reach: the servo tick limits
    (minus write_arm's margin, elbow capped at the mechanical stop) mapped through the calibration,
    intersected with the URDF limits.  Returns (lo, hi) arrays."""
    from so101_ik.write_arm import load_calibration, tick_limits, ticks_to_q
    cal = load_calibration() if cal is None else cal
    limits = tick_limits(cal, elbow_max_deg)
    a = ticks_to_q([limits[j][0] for j in JOINTS], cal)
    b = ticks_to_q([limits[j][1] for j in JOINTS], cal)
    lo = np.maximum(np.minimum(a, b), _LIMITS_LO)      # a negative sign swaps the ends
    hi = np.minimum(np.maximum(a, b), _LIMITS_HI)
    return lo, hi
# Tool z straight down (-z of base); the x/y columns only fix the free roll's reference.
PEN_DOWN_ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
PEN_DOWN_SPEED_MPS = 0.03              # nominal, only used for the waypoint timestamps
PEN_UP_SPEED_MPS = 0.06
_LIMITS_LO = np.array([lo for lo, _ in JOINT_LIMITS])
_LIMITS_HI = np.array([hi for _, hi in JOINT_LIMITS])


# ---------- Cartesian path -----------------------------------------------------------------------
def layout_strokes(strokes, distance_m, plane_z_m, hover_m, mirror=False, lateral_m=0.0,
                   yaw_deg=0.0, plane_slope=(0.0, 0.0)):
    """Strokes (page x right, y up, metres) -> list of (xyz, pen_down) waypoints in base_link.
    The text is centred at (distance_m, lateral_m) on the plane; +lateral is the robot's left (+y).
    yaw_deg rotates the text line about the vertical (0 = along the robot's y axis).  plane_slope
    (dz/dx, dz/dy) lets the surface be tilted in the model's frame: z = plane_z_m + slope . (xy - centre),
    which absorbs a calibration bias measured with pen touches across the writing area."""
    xmin, ymin, xmax, ymax = strokes_bbox(strokes)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    sign = 1.0 if mirror else -1.0
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))

    def to_base(p):
        dx, dy = (p[1] - cy), sign * (p[0] - cx)
        x = distance_m + c * dx - s * dy
        y = lateral_m + s * dx + c * dy
        z = plane_z_m + plane_slope[0] * (x - distance_m) + plane_slope[1] * (y - lateral_m)
        return np.array([x, y, z])

    up = np.array([0.0, 0.0, hover_m])
    path = []
    for stroke in strokes:
        pts = [to_base(p) for p in np.asarray(stroke)]
        path.append((pts[0] + up, False))
        for p in pts:
            path.append((p, True))
        path.append((pts[-1] + up, False))
    return path


def timestamps(path, down_speed=PEN_DOWN_SPEED_MPS, up_speed=PEN_UP_SPEED_MPS):
    t, out = 0.0, [0.0]
    for (a, _), (b, down) in zip(path[:-1], path[1:]):
        t += float(np.linalg.norm(b - a)) / (down_speed if down else up_speed)
        out.append(t)
    return out


def target_pose(xyz, tilt_rad=0.0):
    """Pen-tip pose: tool z-axis pointing down, optionally tilted ``tilt_rad`` away from vertical in the
    arm's own (radial) plane so the pen leans back toward the base like a hand-held pen.  A tilt keeps
    the wrist lower and closer to the shoulder, which this leader arm's small servos can hold."""
    T = np.eye(4)
    if abs(tilt_rad) < 1e-9:
        T[:3, :3] = PEN_DOWN_ROTATION
    else:
        radial = np.array([xyz[0], xyz[1], 0.0])
        n = float(np.linalg.norm(radial))
        radial = radial / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        z_axis = math.sin(tilt_rad) * radial + np.array([0.0, 0.0, -math.cos(tilt_rad)])
        y_axis = np.cross(z_axis, radial)          # horizontal, perpendicular to the arm's plane
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        T[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    T[:3, 3] = xyz
    return T


# ---------- solvers -------------------------------------------------------------------------------
class NewtonOnlySolver:
    """Fallback when there is no trained model: damped Newton from the previous solution."""

    name = "newton-only"

    def __init__(self, model, **newton_kwargs):
        from so101_ik.newton import ik_newton_refine
        self.model, self.kw, self._refine = model, newton_kwargs, ik_newton_refine

    def solve(self, T, q_seed):
        q, iters, pos_err_mm, rot_err_deg, converged = self._refine(T, q_seed, self.model, task="pen", **self.kw)
        return q, {"converged": bool(converged), "iters": int(iters), "pos_err_mm": float(pos_err_mm),
                   "rot_err_deg": float(rot_err_deg)}


class DnnSolver:
    """IKSolver.solve_refined(task="pen"): DNN seed, then the same Newton polish."""

    name = "dnn+newton"

    def __init__(self, model_dir, tool_offset_m, elbow_stop=None, **newton_kwargs):
        from so101_ik.infer import IKSolver
        self.ik = IKSolver(str(model_dir), device="cpu", tool_offset_m=tuple(tool_offset_m))
        self.kw = newton_kwargs
        self.elbow_stop = elbow_stop
        self.fallback = NewtonOnlySolver(self.ik.robot, **newton_kwargs)

    def solve(self, T, q_seed):
        lo, hi = self.kw.get("limits") or (_LIMITS_LO, _LIMITS_HI)
        q_dnn = np.clip(self.ik.solve(T, q_seed), lo, hi)
        # Wrist roll is the free DOF of the pen task: the DNN's roll is arbitrary (it was trained on
        # uniformly random q), and feeding it back as the next seed makes roll random-walk into its
        # limit over a few hundred waypoints.  Keep the previous roll; Newton polishes the rest.
        q_dnn[4] = q_seed[4]
        q, iters, pos_err_mm, rot_err_deg, converged = self.fallback._refine(
            T, q_dnn, self.ik.robot, task="pen", **self.kw)
        out = {"converged": bool(converged), "iters": int(iters),
               "pos_err_mm": float(pos_err_mm), "rot_err_deg": float(rot_err_deg),
               "q_dnn_rad": [float(x) for x in q_dnn]}
        if (not out["converged"] or not within_limits(np.asarray(q), 1e-6, self.kw.get("limits"))
                or (self.elbow_stop is not None and not self.elbow_stop.ok(q))):
            # The demo must not stall on one bad DNN seed: polish from the previous waypoint instead.
            q, out2 = self.fallback.solve(T, q_seed)
            out = {**out2, "q_dnn_rad": out.get("q_dnn_rad"), "fallback": "newton-from-previous"}
        return np.asarray(q, dtype=np.float64).flatten(), out


def make_solver(model_dir, tool_offset_m, newton_kwargs, log=print, elbow_stop=None):
    model = build_so101_model(tuple(tool_offset_m))
    if model_dir:
        weights = Path(model_dir) / "model_best.pt"
        if weights.exists():
            try:
                solver = DnnSolver(model_dir, tool_offset_m, elbow_stop=elbow_stop, **newton_kwargs)
                log(f"solver: DNN seed from {weights} + Newton polish (task=pen)")
                return solver, model
            except Exception as error:   # noqa: BLE001
                log(f"could not load the model from {model_dir} ({error}); falling back to Newton only")
        else:
            log(f"no model_best.pt in {model_dir}; falling back to Newton only")
    else:
        log("no --model_dir given: Newton only, each waypoint seeded by the previous solution")
    return NewtonOnlySolver(model, **newton_kwargs), model


def within_limits(q, margin_rad=0.0, limits=None):
    lo, hi = limits or (_LIMITS_LO, _LIMITS_HI)
    return bool(np.all(q >= lo + margin_rad) and np.all(q <= hi - margin_rad))


def solve_path(path, solver, q_start=START_Q, limit_margin_rad=1e-6, log=print, elbow_stop=None,
               limits=None, tilt_rad=0.0):
    """Solve every waypoint in order. Returns (accepted waypoints, rejected waypoints).

    A waypoint is rejected when Newton did not converge, when it leaves the URDF joint limits, or
    when its elbow bend exceeds the mechanical stop (``elbow_stop``, an ElbowStop)."""
    times = timestamps(path)
    q_prev = np.asarray(q_start, dtype=np.float64)
    accepted, rejected = [], []
    t0 = time.perf_counter()
    for i, ((xyz, pen_down), t) in enumerate(zip(path, times)):
        q, info = solver.solve(target_pose(xyz, tilt_rad), q_prev)
        q = np.asarray(q, dtype=np.float64).flatten()
        row = {"index": i, "t": round(float(t), 3), "xyz": [round(float(v), 5) for v in xyz],
               "pen_down": bool(pen_down), "q_rad": [round(float(v), 6) for v in q], **info}
        row["pos_err_mm"] = round(row["pos_err_mm"], 4)
        row["rot_err_deg"] = round(row["rot_err_deg"], 4)
        reason = None
        if not info["converged"]:
            reason = "did not converge"
        elif not within_limits(q, limit_margin_rad, limits):
            reason = "outside joint limits"
        elif elbow_stop is not None and not elbow_stop.ok(q):
            reason = (f"elbow bend {elbow_stop.bend_deg(q):.1f} deg exceeds the {elbow_stop.max_deg} deg "
                      f"mechanical stop")
        if reason:
            row["reason"] = reason
            rejected.append(row)
            continue
        accepted.append(row)
        q_prev = q
    log(f"solved {len(path)} waypoints in {time.perf_counter() - t0:.1f} s: "
        f"{len(accepted)} accepted, {len(rejected)} rejected")
    return accepted, rejected


# ---------- preview -------------------------------------------------------------------------------
def link_points(q, model):
    """Positions of base, the five joint origins and the pen tip for one configuration."""
    from so101_ik import fk as _fk
    origins = getattr(_fk, "_URDF_JOINT_ORIGINS", None)
    pts = [np.zeros(3)]
    if origins is not None:
        T = np.eye(4)
        for (xyz, rpy), qi in zip(origins, q):
            T = T @ _fk._origin_transform(xyz, rpy) @ _fk._rz4(qi)
            pts.append(T[:3, 3].copy())
    else:   # closest point to the origin on each axis, transported by the preceding joints
        T = np.eye(4)
        for i in range(N_JOINTS):
            p0 = np.cross(model["w"][i], model["v"][i])
            pts.append((T @ np.append(p0, 1.0))[:3])
            T = T @ _fk._exp_se3(model["w"][i], model["v"][i], q[i])
    pts.append(forward_kinematics(q, model)[:3, 3])
    return np.asarray(pts)


def save_preview(path_png, text, strokes, waypoints, rejected, model, plane_z_m, distance_m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # top view: strokes on the plane, in base coordinates (x forward, y left)
    ax = axes[0]
    down = [w for w in waypoints if w["pen_down"]]
    if down:
        xy = np.array([w["xyz"] for w in down])
        ax.plot(xy[:, 1], xy[:, 0], ".", ms=2, color="tab:blue", label="pen down")
    if rejected:
        xy = np.array([w["xyz"] for w in rejected])
        ax.plot(xy[:, 1], xy[:, 0], "x", ms=5, color="tab:red", label="rejected")
    ax.plot(0, 0, "ks", label="base")
    ax.set_aspect("equal")
    ax.invert_xaxis()
    ax.set_xlabel("base y [m]  (left is +y)")
    ax.set_ylabel("base x [m]  (forward)")
    ax.set_title(f"top view: '{text}' at {distance_m:.2f} m")
    ax.legend(loc="lower right", fontsize=8)
    # side view: arm at a few waypoints
    ax = axes[1]
    if waypoints:
        picks = sorted({0, len(waypoints) // 4, len(waypoints) // 2, 3 * len(waypoints) // 4, len(waypoints) - 1})
        for k, i in enumerate(picks):
            pts = link_points(np.array(waypoints[i]["q_rad"]), model)
            ax.plot(pts[:, 0], pts[:, 2], "o-", ms=3, alpha=0.8, label=f"t={waypoints[i]['t']:.1f}s")
    ax.axhline(plane_z_m, color="gray", ls="--", lw=1)
    ax.set_aspect("equal")
    ax.set_xlabel("base x [m]")
    ax.set_ylabel("base z [m]")
    ax.set_title("side view (links from FK)")
    if waypoints:
        ax.legend(fontsize=7)
    # joint angles vs time
    ax = axes[2]
    if waypoints:
        t = np.array([w["t"] for w in waypoints])
        q = np.degrees(np.array([w["q_rad"] for w in waypoints]))
        for j, name in enumerate(JOINTS):
            ax.plot(t, q[:, j], lw=1, label=name)
        pen = np.array([w["pen_down"] for w in waypoints])
        ax.fill_between(t, ax.get_ylim()[0], ax.get_ylim()[1], where=pen, color="tab:blue", alpha=0.06)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("joint angle [deg]")
    ax.set_title("joint angles vs time (shaded = pen down)")
    if waypoints:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path_png, dpi=110)
    plt.close(fig)


# ---------- CLI -----------------------------------------------------------------------------------
def build_plan(args, log=print):
    strokes = text_to_strokes(args.text, args.text_height_m, args.font, args.max_spacing_m)
    if not strokes:
        raise ValueError(f"no strokes for text {args.text!r}")
    bbox = strokes_bbox(strokes)
    path = layout_strokes(strokes, args.distance_m, args.plane_z_m, args.hover_m, args.mirror,
                          getattr(args, "lateral_m", 0.0) or 0.0, getattr(args, "yaw_deg", 0.0) or 0.0,
                          tuple(getattr(args, "plane_slope", None) or (0.0, 0.0)))
    elbow_stop = ElbowStop(getattr(args, "elbow_max_deg", None))
    # Solve inside the range the calibrated servos can reach, not just the URDF's.
    limits = calibrated_limits(elbow_stop.cal, elbow_stop.max_deg)
    log("joint limits used (deg, URDF frame, from the servo tick limits): "
        + ", ".join(f"{j} {math.degrees(lo):.0f}..{math.degrees(hi):.0f}"
                    for j, lo, hi in zip(JOINTS, limits[0], limits[1])))
    newton_kwargs = {"max_iter": args.max_iter, "eps_v_mm": args.eps_v_mm, "eps_w_rad": args.eps_w_rad,
                     "limits": limits}
    solver, model = make_solver(args.model_dir, args.pen_tip_offset_m, newton_kwargs, log=log,
                                elbow_stop=elbow_stop)
    q_start = np.clip(START_Q, limits[0], limits[1])
    tilt_rad = math.radians(getattr(args, "pen_tilt_deg", 0.0) or 0.0)
    accepted, rejected = solve_path(path, solver, q_start=q_start, log=log, elbow_stop=elbow_stop,
                                    limits=limits, tilt_rad=tilt_rad)
    radial = [float(math.hypot(p[0], p[1])) for p, _ in path]
    roll = [w["q_rad"][4] for w in accepted]
    summary = {
        "peak_elbow_bend_deg": round(max(elbow_stop.bend_deg(w["q_rad"]) for w in accepted), 2) if accepted else None,
        "elbow_max_deg": elbow_stop.max_deg,
        "wrist_roll_range_rad": [round(min(roll), 6), round(max(roll), 6)] if roll else None,
        "solver": solver.name,
        "n_strokes": len(strokes),
        "n_waypoints": len(path),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "convergence_pct": round(100.0 * len(accepted) / len(path), 2),
        "rejected": [{k: r[k] for k in ("index", "xyz", "pen_down", "reason", "pos_err_mm", "rot_err_deg")}
                     for r in rejected],
        "pen_down_length_m": round(strokes_length(strokes), 4),
        "duration_s": round(accepted[-1]["t"], 2) if accepted else 0.0,
        "mean_iters": round(float(np.mean([w["iters"] for w in accepted])), 2) if accepted else None,
        "n_newton_fallback": sum(1 for w in accepted if w.get("fallback")),
        "max_pos_err_mm": round(max(w["pos_err_mm"] for w in accepted), 4) if accepted else None,
        "max_rot_err_deg": round(max(w["rot_err_deg"] for w in accepted), 4) if accepted else None,
        "radial_reach_m": [round(min(radial), 4), round(max(radial), 4)],
        "text_bbox_page_m": [round(v, 4) for v in bbox],
        "max_joint_step_rad": round(float(np.max(np.abs(np.diff(
            np.array([w["q_rad"] for w in accepted]), axis=0)))), 5) if len(accepted) > 1 else 0.0,
    }
    plan = {
        "text": args.text,
        "params": {
            "distance_m": args.distance_m, "plane_z_m": args.plane_z_m, "text_height_m": args.text_height_m,
            "pen_tilt_deg": getattr(args, "pen_tilt_deg", 0.0) or 0.0,
            "hover_m": args.hover_m, "pen_tip_offset_m": [float(v) for v in args.pen_tip_offset_m],
            "font": args.font, "max_spacing_m": args.max_spacing_m, "mirror": bool(args.mirror),
            "model_dir": args.model_dir, "start_q_rad": [float(v) for v in START_Q],
            "newton": {k: v for k, v in newton_kwargs.items() if k != "limits"},
            "joint_limits_deg": {j: [round(math.degrees(lo), 2), round(math.degrees(hi), 2)]
                                 for j, lo, hi in zip(JOINTS, limits[0], limits[1])},
            "joints": list(JOINTS), "frame": "URDF base_link, metres, radians",
            "orientation": "tool z = -z of base (pen straight down), wrist roll held at start_q_rad[4]",
        },
        "waypoints": [{k: w[k] for k in ("t", "xyz", "pen_down", "q_rad", "converged", "iters", "pos_err_mm")}
                      for w in accepted],
        "summary": summary,
    }
    return plan, strokes, accepted, rejected, model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--text", default="hello Neo")
    p.add_argument("--distance_m", type=float, default=0.20,
                   help="centre of the text, measured from the base axis along the arm's forward (+x) direction")
    p.add_argument("--plane_z_m", type=float, default=TABLE_Z_M,
                   help="height of the writing surface in the URDF base frame (bottom of the base = 0 = the table)")
    p.add_argument("--text_height_m", type=float, default=DEFAULT_HEIGHT_M, help="cap height")
    p.add_argument("--hover_m", type=float, default=0.01, help="pen lift between strokes")
    p.add_argument("--lateral_m", type=float, default=0.0,
                   help="sideways offset of the text centre in base_link y (+ = robot's left)")
    p.add_argument("--yaw_deg", type=float, default=0.0, help="rotate the text line about vertical")
    p.add_argument("--plane_slope", type=float, nargs=2, default=None, metavar=("DZDX", "DZDY"),
                   help="tilt of the writing plane in the model frame, from pen touches")
    p.add_argument("--pen_tilt_deg", type=float, default=0.0,
                   help="lean the pen back toward the base by this angle from vertical (0 = vertical pen); "
                        "30-45 keeps the arm folded so the small shoulder servo can hold it")
    p.add_argument("--pen_tip_offset_m", type=float, nargs=3, default=list(DEFAULT_PEN_TIP_OFFSET_M),
                   metavar=("X", "Y", "Z"),
                   help="pen tip in the gripper_frame_link frame; default 0 0 0.05 = a 5 cm pen protruding along the tool axis")
    p.add_argument("--model_dir", default=None, help="directory with model_best.pt + scalers (omit: Newton only)")
    p.add_argument("--out", default="plan.json")
    p.add_argument("--preview", default="plan.png")
    p.add_argument("--font", default=DEFAULT_FONT)
    p.add_argument("--max_spacing_m", type=float, default=MAX_SPACING_M)
    p.add_argument("--mirror", action="store_true", help="readable from in front of the robot instead of behind")
    p.add_argument("--max_iter", type=int, default=200)
    p.add_argument("--eps_v_mm", type=float, default=1.0)
    p.add_argument("--eps_w_rad", type=float, default=0.01)
    p.add_argument("--elbow_max_deg", type=float, default=None,
                   help="elbow mechanical stop in degrees from straight up (default: config/air_arm.json)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    plan, strokes, accepted, rejected, model = build_plan(args)
    s = plan["summary"]
    print(f"text '{args.text}': {s['n_strokes']} strokes, {s['n_waypoints']} waypoints, "
          f"pen-down path {s['pen_down_length_m'] * 100:.1f} cm, radial reach "
          f"{s['radial_reach_m'][0]:.3f}..{s['radial_reach_m'][1]:.3f} m")
    print(f"converged {s['n_accepted']}/{s['n_waypoints']} ({s['convergence_pct']:.1f}%), "
          f"{s['n_newton_fallback']} via Newton-from-previous fallback, "
          f"mean iters {s['mean_iters']}, max pos err {s['max_pos_err_mm']} mm, "
          f"max axis err {s['max_rot_err_deg']} deg, max joint step {s['max_joint_step_rad']} rad")
    roll = s["wrist_roll_range_rad"]
    print(f"peak elbow bend {s['peak_elbow_bend_deg']} deg (mechanical stop {s['elbow_max_deg']} deg); "
          f"wrist roll {roll[0]:.4f}..{roll[1]:.4f} rad" if roll else "no accepted waypoints")
    if rejected:
        print(f"REJECTED {len(rejected)} waypoints:")
        for r in rejected[:20]:
            print(f"  #{r['index']:4d} xyz={r['xyz']} pen_down={r['pen_down']} {r['reason']} "
                  f"(pos {r['pos_err_mm']:.2f} mm, axis {r['rot_err_deg']:.2f} deg)")
        if len(rejected) > 20:
            print(f"  ... and {len(rejected) - 20} more")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=1))
    print(f"wrote {out}")
    if args.preview:
        Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
        save_preview(args.preview, args.text, strokes, accepted, rejected, model, args.plane_z_m, args.distance_m)
        print(f"wrote {args.preview}")
    return 0 if not rejected else 2


if __name__ == "__main__":
    sys.exit(main())
