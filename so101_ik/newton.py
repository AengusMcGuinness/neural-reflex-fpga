"""Damped Newton-Raphson IK polish for the SO-101 arm.

Port of ``ik_newton_refine`` from the ES 259 project (``app.py`` /
``benchmark_ik2.py``), generalised to the 5-DOF SO-101:

* damped least squares  ``dq = solve(J^T J + lam I, J^T e)``, ``lam = 0.01 ||e||``
* per-step clamp ``||dq|| <= 0.5 rad``
* joints clipped to ``JOINT_LIMITS`` after every step
* convergence when ``||pos_err|| < eps_v_mm`` AND ``rot_err < eps_w_rad``
* max 200 iterations

Two task modes:

``task="se3"``
    Full 6D twist error ``[dp; dw]`` exactly as in the reference, where ``dw``
    is the matrix-log of ``R_target R_cur^T``.  The SO-101 has only 5 DOF, so a
    generic SE(3) target is not reachable; this mode is meant for targets that
    were produced by FK (the benchmark) and for the "polish" step after the DNN.

``task="pen"``
    Writing task: ``[dp; theta * unit(cross(z_cur, z_des))]`` where ``z`` is the
    third column of the tool rotation and ``theta`` the angle between the two
    axes (the geodesic error, so the pull does not vanish for antiparallel axes
    and Newton converges from inverted starts; when the axes are exactly
    antiparallel an arbitrary axis perpendicular to ``z_cur`` is used).  Rotation
    about the pen axis is unobservable in this error, so the angular Jacobian
    rows are projected onto the plane orthogonal to ``z_cur``.  The wrist_roll
    column of the Jacobian is zeroed as well: the pen tip sits ~8 mm off the roll
    axis, so roll is not truly free (it would be used to fix sub-millimetre
    position error and drift through its whole range over a long plan); with the
    column zeroed ``dq[4] == 0`` exactly and roll stays at its seed value.
    Convergence: ``||dp|| < eps_v_mm`` AND ``angle(z_cur, z_des) < eps_w_rad``.

Positions are in METRES internally; ``eps_v_mm`` is converted on entry and the
reported ``pos_err_mm`` is converted back.

Jacobian frame
--------------
``fk.space_jacobian`` returns the space-frame Jacobian ``[J_v; J_w]`` whose
linear rows give the velocity of the body point that currently sits at the
space origin, not the tool point.  The reference code pairs that Jacobian
directly with the tool position error.  Here the default ``jacobian="tool"``
converts the linear rows to the tool-point velocity Jacobian
``J_p = J_v - [p_tool]x J_w`` so that the position error is driven to zero
consistently.  Pass ``jacobian="space"`` to reproduce the reference pairing
verbatim.
"""

from __future__ import annotations

import numpy as np

from so101_ik.fk import JOINT_LIMITS, N_JOINTS, forward_kinematics, skew, space_jacobian

_LIMITS_LO = np.array([lo for lo, _ in JOINT_LIMITS], dtype=np.float64)
_LIMITS_HI = np.array([hi for _, hi in JOINT_LIMITS], dtype=np.float64)


def rotation_log(R_err: np.ndarray) -> tuple[np.ndarray, float]:
    """Matrix log of a rotation -> (axis*angle 3-vector, angle). Same as reference."""
    cos_a = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_a))
    if angle < 1e-10:
        return np.zeros(3), 0.0
    s = np.sin(angle)
    if abs(s) < 1e-8:
        # angle ~ pi: axis from the symmetric part (R + I)/2 = w w^T
        A = (R_err + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(A)))
        w = A[:, k] / max(np.sqrt(A[k, k]), 1e-12)
        return w * angle, angle
    dw = (angle / (2.0 * s)) * np.array(
        [R_err[2, 1] - R_err[1, 2], R_err[0, 2] - R_err[2, 0], R_err[1, 0] - R_err[0, 1]]
    )
    return dw, angle


def pose_errors(T_target: np.ndarray, T_cur: np.ndarray) -> tuple[float, float]:
    """(position error [m], rotation error [rad]) between two transforms."""
    dp = T_target[:3, 3] - T_cur[:3, 3]
    _, angle = rotation_log(T_target[:3, :3] @ T_cur[:3, :3].T)
    return float(np.linalg.norm(dp)), angle


def axis_angle_error(z_cur: np.ndarray, z_des: np.ndarray) -> float:
    """Angle [rad] between two unit vectors."""
    c = float(np.clip(np.dot(z_cur, z_des), -1.0, 1.0))
    return float(np.arccos(c))


def _task_error(T_target: np.ndarray, T_cur: np.ndarray, task: str):
    """Return (error 6-vector, pos_err_m, rot_err_rad) for the given task."""
    dp = T_target[:3, 3] - T_cur[:3, 3]
    pos_err = float(np.linalg.norm(dp))
    if task == "se3":
        dw, angle = rotation_log(T_target[:3, :3] @ T_cur[:3, :3].T)
        return np.concatenate([dp, dw]), pos_err, angle
    if task == "pen":
        z_cur = T_cur[:3, 2]
        z_des = T_target[:3, 2]
        z_des = z_des / max(np.linalg.norm(z_des), 1e-12)
        angle = axis_angle_error(z_cur, z_des)
        return np.concatenate([dp, axis_error(z_cur, z_des, angle)]), pos_err, angle
    raise ValueError(f"unknown task {task!r}; expected 'se3' or 'pen'")


WRIST_ROLL = N_JOINTS - 1


def axis_error(z_cur: np.ndarray, z_des: np.ndarray, angle: float | None = None) -> np.ndarray:
    """Geodesic axis error ``theta * unit(cross(z_cur, z_des))`` (3-vector, perpendicular to z_cur).

    Unlike the raw cross product (``sin(theta)``) this does not vanish as the axes become
    antiparallel.  Exactly antiparallel axes have no preferred rotation axis, so any direction
    perpendicular to ``z_cur`` is used (the least-aligned base axis, projected)."""
    if angle is None:
        angle = axis_angle_error(z_cur, z_des)
    axis = np.cross(z_cur, z_des)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        if angle < 1e-9:
            return np.zeros(3)
        helper = np.eye(3)[int(np.argmin(np.abs(z_cur)))]
        axis = helper - np.dot(helper, z_cur) * z_cur
        norm = float(np.linalg.norm(axis))
    return angle * axis / norm


def _task_jacobian(q: np.ndarray, T_cur: np.ndarray, model: dict, task: str, jacobian: str,
                   lock_roll: bool = True) -> np.ndarray:
    J = space_jacobian(q, model)  # (6, n) rows [linear; angular], space frame
    J_v = J[:3]
    J_w = J[3:]
    if jacobian == "tool":
        p = T_cur[:3, 3]
        J_v = J_v - skew(p) @ J_w
    elif jacobian != "space":
        raise ValueError(f"unknown jacobian {jacobian!r}; expected 'tool' or 'space'")
    if task == "pen":
        z = T_cur[:3, 2]
        J_w = (np.eye(3) - np.outer(z, z)) @ J_w  # roll about the pen axis is unobservable
        J = np.vstack([J_v, J_w])
        # The pen tip is off the roll axis, so the roll column has a small linear part that the
        # damped solve would use to trim position error; zeroing the column keeps dq[roll] == 0 and
        # the roll exactly at its seed value for the whole plan.
        if lock_roll:
            J[:, WRIST_ROLL] = 0.0
        return J
    return np.vstack([J_v, J_w])


def ik_newton_refine(
    T_target: np.ndarray,
    q_init: np.ndarray,
    model: dict,
    task: str = "se3",
    max_iter: int = 200,
    eps_v_mm: float = 1.0,
    eps_w_rad: float = 0.01,
    max_step: float = 0.5,
    jacobian: str = "tool",
    limits=None,
    lock_roll: bool = True,
) -> tuple[np.ndarray, int, float, float, bool]:
    """Damped Newton-Raphson IK.

    Returns ``(q, iters, pos_err_mm, rot_err_deg, converged)``.  For
    ``task="pen"`` the reported rotation error is the angle between the current
    and desired tool z-axes.  ``limits`` is an optional ``(lo, hi)`` pair of
    per-joint arrays used for clipping instead of the URDF ``JOINT_LIMITS``
    (e.g. the physical servo range of a calibrated arm).  ``lock_roll=False`` lets the
    pen task move wrist_roll too (needed when the frozen-roll 4-DOF chain cannot meet a
    5-constraint target exactly, e.g. a hand held sideways in the air).
    """
    T_target = np.asarray(T_target, dtype=np.float64)
    q = np.asarray(q_init, dtype=np.float64).flatten().copy()
    if q.shape[0] != N_JOINTS:
        raise ValueError(f"q_init must have {N_JOINTS} entries, got {q.shape[0]}")
    if limits is None:
        lo, hi = _LIMITS_LO, _LIMITS_HI
    else:
        lo = np.asarray(limits[0], dtype=np.float64)
        hi = np.asarray(limits[1], dtype=np.float64)
    q = np.clip(q, lo, hi)
    eps_v = eps_v_mm * 1e-3
    n = N_JOINTS
    eye_n = np.eye(n)

    for it in range(1, max_iter + 1):
        T_cur = forward_kinematics(q, model)
        err, pos_err, rot_err = _task_error(T_target, T_cur, task)
        if pos_err < eps_v and rot_err < eps_w_rad:
            return q, it, pos_err * 1e3, float(np.degrees(rot_err)), True

        J = _task_jacobian(q, T_cur, model, task, jacobian, lock_roll)
        lam = 0.01 * float(np.linalg.norm(err))
        JtJ = J.T @ J
        try:
            dq = np.linalg.solve(JtJ + lam * eye_n, J.T @ err)
        except np.linalg.LinAlgError:
            dq = np.linalg.lstsq(JtJ + (lam + 1e-9) * eye_n, J.T @ err, rcond=None)[0]

        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > max_step:
            dq = dq * (max_step / dq_norm)

        q = np.clip(q + dq, lo, hi)

    T_cur = forward_kinematics(q, model)
    _, pos_err, rot_err = _task_error(T_target, T_cur, task)
    return q, max_iter, pos_err * 1e3, float(np.degrees(rot_err)), False


__all__ = [
    "ik_newton_refine",
    "rotation_log",
    "pose_errors",
    "axis_angle_error",
    "axis_error",
]
