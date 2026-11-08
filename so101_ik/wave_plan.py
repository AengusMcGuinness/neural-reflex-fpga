#!/usr/bin/env python3
"""Plan a wave (and optionally a circle) in the air with the SO-101, solved waypoint by waypoint
with the DNN-seeded Newton-Raphson IK.

The hand (pen/handle tip) sweeps side to side along a horizontal arc above the base, like a
"hi / bye" wave, then traces a circle.  The tool axis points UP, so the arm stays folded and
nearly vertical where the small leader-arm servos have the least gravity load.  The output is a
plan.json in the same shape as write_plan's, so so101_ik.write_arm executes it unchanged
(all waypoints are "pen up").

    .venv-ik/bin/python -m so101_ik.wave_plan --model_dir so101_ik/results/so101_1m \
        --out data/so101_ik/wave_plan.json --preview data/so101_ik/wave_plan.png
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from so101_ik.fk import JOINTS
from so101_ik.write_plan import (ElbowStop, calibrated_limits, make_solver, solve_path, START_Q,
                                 PEN_UP_SPEED_MPS, DEFAULT_PEN_TIP_OFFSET_M)

# target_pose(xyz, pi) points the tool z-axis straight up; smaller tilts lean the hand forward.
def hand_tilt_rad(hand_tilt_deg: float) -> float:
    """hand_tilt_deg = 0 hand straight up, 90 hand horizontal pointing forward."""
    return math.pi - math.radians(hand_tilt_deg)


def wave_path(centre_xyz, half_width_m, cycles, bob_m, spacing_m):
    """Side-to-side arc through the centre: y sweeps +-half_width, with a small vertical bob at
    the ends so it reads as a wave rather than a slide."""
    cx, cy, cz = centre_xyz
    n_per_cycle = max(8, int(4 * half_width_m / spacing_m))
    pts = []
    for i in range(cycles * n_per_cycle + 1):
        phase = 2 * math.pi * i / n_per_cycle
        y = cy + half_width_m * math.sin(phase)
        z = cz + bob_m * (1 - math.cos(2 * phase)) / 2
        pts.append(np.array([cx, y, z]))
    return pts


def circle_path(centre_xyz, radius_m, turns, spacing_m, plane="yz"):
    cx, cy, cz = centre_xyz
    n = max(12, int(2 * math.pi * radius_m / spacing_m))
    pts = []
    for i in range(turns * n + 1):
        a = 2 * math.pi * i / n
        if plane == "yz":
            pts.append(np.array([cx, cy + radius_m * math.cos(a), cz + radius_m * math.sin(a)]))
        else:                                    # horizontal circle
            pts.append(np.array([cx + radius_m * math.cos(a), cy + radius_m * math.sin(a), cz]))
    return pts


def build(args, log=print):
    centre = np.array([args.x_m, args.y_m, args.z_m])
    pts = wave_path(centre, args.half_width_m, args.cycles, args.bob_m, args.spacing_m)
    if args.circle_radius_m > 0:
        pts += circle_path(centre, args.circle_radius_m, args.circle_turns, args.spacing_m, args.circle_plane)
        pts.append(centre.copy())
    path = [(p, False) for p in pts]        # everything is "pen up": nothing touches anything

    elbow_stop = ElbowStop(args.elbow_max_deg)
    limits = calibrated_limits(elbow_stop.cal, elbow_stop.max_deg)
    log("joint limits used (deg): " + ", ".join(f"{j} {math.degrees(lo):.0f}..{math.degrees(hi):.0f}"
                                              for j, lo, hi in zip(JOINTS, limits[0], limits[1])))
    newton_kwargs = {"max_iter": args.max_iter, "eps_v_mm": args.eps_v_mm, "eps_w_rad": args.eps_w_rad,
                     "limits": limits, "lock_roll": True}
    solver, model = make_solver(args.model_dir, args.pen_tip_offset_m, newton_kwargs, log=log,
                                elbow_stop=elbow_stop)
    # Seed: arm nearly straight up with a slight forward bend, hand up.
    seed = np.radians(args.seed_deg) if args.seed_deg else np.array([0.0, -0.1, -0.9, -0.3, START_Q[4]])
    q_start = np.clip(seed, limits[0], limits[1])
    t0 = time.perf_counter()
    accepted, rejected = solve_path(path, solver, q_start=q_start, log=log, elbow_stop=elbow_stop,
                                    limits=limits, tilt_rad=hand_tilt_rad(args.hand_tilt_deg))
    log(f"solved {len(path)} waypoints in {time.perf_counter() - t0:.1f} s: {len(accepted)} accepted, "
        f"{len(rejected)} rejected")
    # Re-time: write_plan's timestamps assume pen speeds; use the wave speed for everything.
    t = 0.0
    for prev, w in zip([None] + accepted[:-1], accepted):
        if prev is not None:
            t += float(np.linalg.norm(np.array(w["xyz"]) - np.array(prev["xyz"]))) / args.speed_mps
        w["t"] = round(t, 3)
    summary = {
        "solver": solver.name, "n_waypoints": len(path), "n_accepted": len(accepted), "n_rejected": len(rejected),
        "convergence_pct": round(100.0 * len(accepted) / len(path), 2),
        "rejected": [{k: r[k] for k in ("index", "xyz", "reason")} for r in rejected][:20],
        "mean_iters": round(float(np.mean([w["iters"] for w in accepted])), 2) if accepted else None,
        "max_pos_err_mm": round(max(w["pos_err_mm"] for w in accepted), 4) if accepted else None,
        "max_rot_err_deg": round(max(w["rot_err_deg"] for w in accepted), 4) if accepted else None,
        "peak_elbow_bend_deg": round(max(elbow_stop.bend_deg(w["q_rad"]) for w in accepted), 2) if accepted else None,
        "elbow_max_deg": elbow_stop.max_deg,
        "duration_s": accepted[-1]["t"] if accepted else 0.0,
        "max_joint_step_rad": round(float(np.max(np.abs(np.diff(np.array([w["q_rad"] for w in accepted]), axis=0)))), 5)
        if len(accepted) > 1 else 0.0,
    }
    log(f"converged {summary['convergence_pct']}%, mean iters {summary['mean_iters']}, "
        f"max pos err {summary['max_pos_err_mm']} mm, peak elbow bend {summary['peak_elbow_bend_deg']} deg, "
        f"duration {summary['duration_s']} s at {args.speed_mps} m/s")
    plan = {
        "text": f"wave x{args.cycles}" + (f" + circle r{args.circle_radius_m}" if args.circle_radius_m > 0 else ""),
        "params": {"x_m": args.x_m, "y_m": args.y_m, "z_m": args.z_m, "half_width_m": args.half_width_m,
                   "cycles": args.cycles, "bob_m": args.bob_m, "circle_radius_m": args.circle_radius_m,
                   "pen_tip_offset_m": list(args.pen_tip_offset_m), "plane_z_m": 0.0,
                   "orientation": f"hand {args.hand_tilt_deg} deg from straight up toward forward", "hand_tilt_deg": args.hand_tilt_deg, "model_dir": args.model_dir,
                   "joints": list(JOINTS), "frame": "URDF base_link, metres, radians"},
        "waypoints": accepted, "summary": summary,
    }
    return plan, rejected


def preview(plan, path_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = np.array([w["xyz"] for w in plan["waypoints"]])
    Q = np.degrees(np.array([w["q_rad"] for w in plan["waypoints"]]))
    T = np.array([w["t"] for w in plan["waypoints"]])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(P[:, 1] * 1000, P[:, 2] * 1000, ".-", ms=2)
    ax[0].set_xlabel("y [mm] (robot's left +)"); ax[0].set_ylabel("z [mm]"); ax[0].set_title("hand path, front view")
    ax[0].set_aspect("equal"); ax[0].invert_xaxis()
    for i, j in enumerate(JOINTS):
        ax[1].plot(T, Q[:, i], label=j)
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("deg"); ax[1].legend(fontsize=7); ax[1].set_title("joint angles")
    fig.tight_layout(); fig.savefig(path_png, dpi=110); plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--x_m", type=float, default=0.10, help="hand centre forward of base_link")
    p.add_argument("--y_m", type=float, default=0.0)
    p.add_argument("--z_m", type=float, default=0.40, help="hand centre height")
    p.add_argument("--half_width_m", type=float, default=0.10)
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--bob_m", type=float, default=0.02)
    p.add_argument("--circle_radius_m", type=float, default=0.06, help="0 disables the circle")
    p.add_argument("--circle_turns", type=int, default=1)
    p.add_argument("--circle_plane", choices=("yz", "xy"), default="yz")
    p.add_argument("--hand_tilt_deg", type=float, default=45.0,
                   help="hand direction: 0 = straight up, 90 = horizontal forward (45 keeps the wrist in range)")
    p.add_argument("--seed_deg", type=float, nargs=5, default=None, help="URDF joint seed for the first waypoint, degrees")
    p.add_argument("--spacing_m", type=float, default=0.005)
    p.add_argument("--speed_mps", type=float, default=0.08)
    p.add_argument("--pen_tip_offset_m", type=float, nargs=3, default=list(DEFAULT_PEN_TIP_OFFSET_M))
    p.add_argument("--model_dir", default=None)
    p.add_argument("--out", default="wave_plan.json")
    p.add_argument("--preview", default="wave_plan.png")
    p.add_argument("--max_iter", type=int, default=200)
    p.add_argument("--eps_v_mm", type=float, default=1.0)
    p.add_argument("--eps_w_rad", type=float, default=0.01)
    p.add_argument("--elbow_max_deg", type=float, default=None)
    args = p.parse_args(argv)
    plan, rejected = build(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(plan, indent=1))
    print(f"wrote {args.out}")
    if plan["waypoints"]:
        preview(plan, args.preview)
        print(f"preview {args.preview}")
    return 0 if not rejected else 2


if __name__ == "__main__":
    raise SystemExit(main())
