"""Forward kinematics for the SO-101 arm (TheRobotStudio SO-ARM100,
Simulation/SO101/so101_new_calib.urdf) in the Product-of-Exponentials form
used by the ES 259 UR5e project.

Only the 5 kinematic joints are modelled; the gripper jaw is excluded.
Units: metres and radians, float64 throughout.
"""
from __future__ import annotations

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
N_JOINTS = 5
JOINT_LIMITS = [
    (-1.91986, 1.91986),
    (-1.74533, 1.74533),
    (-1.69, 1.69),
    (-1.65806, 1.65806),
    (-2.74385, 2.84121),
]

# URDF joint origins (xyz, rpy) expressed in the parent link frame, in chain order.
# Every joint rotates about its local z axis (axis 0 0 1).
_URDF_JOINT_ORIGINS = [
    ((0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159)),      # shoulder_pan
    ((-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0)),                # shoulder_lift
    ((-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708)),      # elbow_flex
    ((-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708)),       # wrist_flex
    ((5.55112e-17, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159)),              # wrist_roll
]
# Fixed gripper_frame_joint: gripper_link -> gripper_frame_link (tool frame).
_URDF_TOOL_ORIGIN = ((-0.0079, -0.000218121, -0.0981274), (0.0, 3.14159, 0.0))


# --------------------------------------------------------------------------- #
# Small SO(3)/SE(3) helpers
# --------------------------------------------------------------------------- #
def skew(v) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix of a 3-vector."""
    v = np.asarray(v, dtype=np.float64).flatten()
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rpy_to_rotmat(rpy) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = (float(x) for x in rpy)
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def _origin_transform(xyz, rpy) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_rotmat(rpy)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def _trans(xyz) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64).flatten()
    return T


def _rz4(theta: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rot_z(theta)
    return T


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def build_so101_model(tool_offset_m=(0.0, 0.0, 0.0)) -> dict:
    """Build the PoE model {w, v, M, limits, names} for the SO-101.

    At q = 0 every joint frame is chained from the URDF origins; joint i's
    space-frame axis is w_i = R_i[:, 2] and v_i = -w_i x p_i.  M is the tool
    (gripper_frame_link) pose at q = 0, followed by an extra translation
    ``tool_offset_m`` expressed in the gripper_frame_link frame (pen tip).
    """
    w = np.zeros((N_JOINTS, 3), dtype=np.float64)
    v = np.zeros((N_JOINTS, 3), dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    for i, (xyz, rpy) in enumerate(_URDF_JOINT_ORIGINS):
        T = T @ _origin_transform(xyz, rpy)
        w_i = T[:3, 2].copy()
        p_i = T[:3, 3].copy()
        w[i] = w_i
        v[i] = -np.cross(w_i, p_i)
    M = T @ _origin_transform(*_URDF_TOOL_ORIGIN) @ _trans(tool_offset_m)
    return {
        "w": w,
        "v": v,
        "M": M,
        "limits": [tuple(l) for l in JOINT_LIMITS],
        "names": tuple(JOINTS),
        "tool_offset_m": tuple(float(x) for x in tool_offset_m),
    }


def _exp_se3(w: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    wh = skew(w)
    wh2 = wh @ wh
    ct, st = np.cos(theta), np.sin(theta)
    R = np.eye(3) + st * wh + (1.0 - ct) * wh2
    G = np.eye(3) * theta + (1.0 - ct) * wh + (theta - st) * wh2
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = G @ v
    return T


# --------------------------------------------------------------------------- #
# Forward kinematics
# --------------------------------------------------------------------------- #
def forward_kinematics(q, model: dict) -> np.ndarray:
    """PoE FK: (5,) joint angles -> (4,4) tool pose in base_link."""
    q = np.asarray(q, dtype=np.float64).flatten()
    T = np.eye(4, dtype=np.float64)
    for i in range(N_JOINTS):
        T = T @ _exp_se3(model["w"][i], model["v"][i], q[i])
    return T @ model["M"]


def fk_batch(q, model: dict) -> np.ndarray:
    """Vectorised PoE FK: (N,5) -> (N,4,4). Loops over joints, not samples."""
    q = np.asarray(q, dtype=np.float64).reshape(-1, N_JOINTS)
    N = q.shape[0]
    w = model["w"]
    v = model["v"]
    T = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
    for j in range(N_JOINTS):
        th = q[:, j]
        wh = skew(w[j])
        wh2 = wh @ wh
        st = np.sin(th)[:, None, None]
        ct = np.cos(th)[:, None, None]
        R = np.eye(3)[None] + st * wh[None] + (1.0 - ct) * wh2[None]
        G = (th[:, None, None] * np.eye(3)[None]
             + (1.0 - ct) * wh[None]
             + (th[:, None, None] - np.sin(th)[:, None, None]) * wh2[None])
        p = G @ v[j]
        Tj = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
        Tj[:, :3, :3] = R
        Tj[:, :3, 3] = p
        T = T @ Tj
    return T @ model["M"][None]


def space_jacobian(q, model: dict) -> np.ndarray:
    """Space-frame Jacobian (6,5), rows [linear(3); angular(3)].

    Same construction as the ES 259 app.py space_jacobian: column i is the
    screw axis of joint i transported by exp of joints 0..i-1.
    """
    w = model["w"]
    v = model["v"]
    q = np.asarray(q, dtype=np.float64).flatten()
    J = np.zeros((6, N_JOINTS), dtype=np.float64)
    T_acc = np.eye(4, dtype=np.float64)
    for i in range(N_JOINTS):
        if i == 0:
            J[:3, 0] = v[0]
            J[3:, 0] = w[0]
        else:
            R = T_acc[:3, :3]
            p = T_acc[:3, 3]
            w_new = R @ w[i]
            v_new = R @ v[i] + np.cross(p, w_new)
            J[:3, i] = v_new
            J[3:, i] = w_new
        T_acc = T_acc @ _exp_se3(w[i], v[i], q[i])
    return J


def urdf_chain_fk(q, tool_offset_m=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Naive reference FK: chain URDF origins and Rz(q_i). Used by tests only."""
    q = np.asarray(q, dtype=np.float64).flatten()
    T = np.eye(4, dtype=np.float64)
    for i, (xyz, rpy) in enumerate(_URDF_JOINT_ORIGINS):
        T = T @ _origin_transform(xyz, rpy) @ _rz4(q[i])
    return T @ _origin_transform(*_URDF_TOOL_ORIGIN) @ _trans(tool_offset_m)


__all__ = [
    "JOINTS", "N_JOINTS", "JOINT_LIMITS", "skew", "rpy_to_rotmat",
    "build_so101_model", "forward_kinematics", "fk_batch", "space_jacobian",
    "urdf_chain_fk",
]
