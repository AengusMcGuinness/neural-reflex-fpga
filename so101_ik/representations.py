"""
representations.py

Utility functions for the SO-101 seed-conditioned IK pipeline:
  - 6D Continuous Rotation Representation (Zhou et al., 2019)
  - Sin/cos encoding and decoding of joint angles (5 kinematic joints)
  - Pose extraction from 4x4 homogeneous transforms

Port of ES_259_final_project/ik_v3/representations.py generalised from the
6-DOF UR5e to the 5-DOF SO-101 (gripper jaw excluded).
"""

import numpy as np
import torch

N_JOINTS = 5


# ============================================================================
# 6D Continuous Rotation Representation
# ============================================================================

def rotmat_to_6d(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to 6D continuous representation.

    Drops the third column and flattens the first two columns.

    Parameters
    ----------
    R : (..., 3, 3) rotation matrix

    Returns
    -------
    r6d : (..., 6) -- [r11, r21, r31, r12, r22, r32]
    """
    R = np.asarray(R)
    col1 = R[..., :3, 0]  # (..., 3)
    col2 = R[..., :3, 1]  # (..., 3)
    return np.concatenate([col1, col2], axis=-1)


def r6d_to_rotmat(r6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D continuous representation back to 3x3 rotation matrix.

    Uses Gram-Schmidt orthogonalization to recover a valid SO(3) matrix.

    Parameters
    ----------
    r6d : (N, 6) -- [a1, a2, a3, b1, b2, b3]

    Returns
    -------
    R : (N, 3, 3) rotation matrix
    """
    a1 = r6d[:, 0:3]  # (N, 3)
    a2 = r6d[:, 3:6]  # (N, 3)

    # Normalize first column
    e1 = torch.nn.functional.normalize(a1, dim=-1)

    # Orthogonalize second column (remove component along e1)
    u2 = a2 - (e1 * a2).sum(dim=-1, keepdim=True) * e1
    e2 = torch.nn.functional.normalize(u2, dim=-1)

    # Third column via cross product
    e3 = torch.cross(e1, e2, dim=-1)

    R = torch.stack([e1, e2, e3], dim=-1)  # (N, 3, 3)
    return R


# ============================================================================
# Pose extraction from 4x4 transform
# ============================================================================

def extract_pose_9d(T: np.ndarray) -> np.ndarray:
    """Extract 9D pose vector from a 4x4 homogeneous transform.

    Returns [x, y, z, r11, r21, r31, r12, r22, r32]
    (position + 6D rotation = 9D)

    Parameters
    ----------
    T : (4, 4) or (N, 4, 4) homogeneous transform

    Returns
    -------
    pose : (9,) or (N, 9)
    """
    T = np.asarray(T)
    if T.ndim == 2:
        pos = T[:3, 3]
        rot6d = rotmat_to_6d(T[:3, :3])
        return np.concatenate([pos, rot6d])
    else:
        pos = T[:, :3, 3]                   # (N, 3)
        rot6d = rotmat_to_6d(T[:, :3, :3])  # (N, 6)
        return np.concatenate([pos, rot6d], axis=-1)  # (N, 9)


def extract_pose_9d_batch(T: np.ndarray) -> np.ndarray:
    """Extract 9D pose from (N, 4, 4) transforms."""
    return extract_pose_9d(T)


# ============================================================================
# Sin/cos encoding and decoding
# ============================================================================

def encode_sincos(joints: np.ndarray) -> np.ndarray:
    """Encode joint angles to interleaved sin/cos pairs.

    Parameters
    ----------
    joints : (..., 5) joint angles in radians

    Returns
    -------
    sincos : (..., 10) -- [sin(q1), cos(q1), ..., sin(q5), cos(q5)]
    """
    joints = np.asarray(joints, dtype=np.float32)
    out = np.empty(joints.shape[:-1] + (2 * N_JOINTS,), dtype=np.float32)
    for i in range(N_JOINTS):
        out[..., 2 * i] = np.sin(joints[..., i])
        out[..., 2 * i + 1] = np.cos(joints[..., i])
    return out


def decode_sincos(sincos: np.ndarray) -> np.ndarray:
    """Decode sin/cos pairs to joint angles using atan2.

    Parameters
    ----------
    sincos : (..., 10) -- [sin(q1), cos(q1), ..., sin(q5), cos(q5)]

    Returns
    -------
    joints : (..., 5) joint angles in radians
    """
    sincos = np.asarray(sincos)
    joints = np.empty(sincos.shape[:-1] + (N_JOINTS,), dtype=sincos.dtype)
    for i in range(N_JOINTS):
        joints[..., i] = np.arctan2(sincos[..., 2 * i], sincos[..., 2 * i + 1])
    return joints


def decode_sincos_torch(sincos: torch.Tensor) -> torch.Tensor:
    """Decode (N, 10) sin/cos pairs to (N, 5) joint angles via atan2."""
    return torch.stack([
        torch.atan2(sincos[:, 2 * i], sincos[:, 2 * i + 1])
        for i in range(N_JOINTS)
    ], dim=-1)


def normalize_sincos(sincos: torch.Tensor) -> torch.Tensor:
    """L2-normalize each (sin, cos) pair so that sin^2+cos^2=1.

    Parameters
    ----------
    sincos : (N, 10) -- [sin(q1), cos(q1), ..., sin(q5), cos(q5)]

    Returns
    -------
    sincos_normalized : (N, 10)
    """
    pairs = sincos.reshape(-1, N_JOINTS, 2)              # (N, 5, 2)
    pairs = torch.nn.functional.normalize(pairs, dim=-1)  # L2 per pair
    return pairs.reshape(-1, 2 * N_JOINTS)


# ============================================================================
# Column name helpers
# ============================================================================

POSE9D_COLS = ["x", "y", "z", "r6d_0", "r6d_1", "r6d_2",
               "r6d_3", "r6d_4", "r6d_5"]
JOINT_COLS = [f"q{i+1}" for i in range(N_JOINTS)]
SEED_COLS = [f"seed_q{i+1}" for i in range(N_JOINTS)]
SINCOS_COLS = [f"sin_q{i}" if j == 0 else f"cos_q{i}"
               for i in range(1, N_JOINTS + 1) for j in range(2)]
