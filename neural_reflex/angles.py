"""Joint angles in degrees from 21 MediaPipe world landmarks."""

import numpy as np

from .config import JOINT_NAMES

_FINGERS = {
    "index": (0, 5, 6, 7),
    "middle": (0, 9, 10, 11),
    "ring": (0, 13, 14, 15),
    "pinky": (0, 17, 18, 19),
}


def _unit_angle_degrees(first, second):
    first_length = np.linalg.norm(first)
    second_length = np.linalg.norm(second)
    if first_length < 1e-9 or second_length < 1e-9:
        raise ValueError("landmark bones must have length")
    cosine = np.clip(np.dot(first, second) / (first_length * second_length), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _flexion(proximal, joint, distal):
    return 180.0 - _unit_angle_degrees(proximal - joint, distal - joint)


def joint_angles(world_landmarks):
    points = np.asarray(world_landmarks, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError("expected 21 world landmarks with x, y, z")
    angles = {}
    for name, (wrist, knuckle, middle, farther) in _FINGERS.items():
        angles[f"{name}_mcp"] = _flexion(points[wrist], points[knuckle], points[middle])
        angles[f"{name}_pip"] = _flexion(points[knuckle], points[middle], points[farther])
    angles["thumb_mcp"] = _flexion(points[1], points[2], points[3])
    angles["thumb_ip"] = _flexion(points[2], points[3], points[4])
    wrist = points[0]
    angles["thumb_opposition"] = _unit_angle_degrees(points[2] - wrist, points[5] - wrist)
    angles["finger_spread"] = _unit_angle_degrees(points[5] - wrist, points[17] - wrist)
    forward = points[9] - wrist
    horizontal = np.hypot(forward[0], forward[2])
    angles["palm_pitch"] = float(np.degrees(np.arctan2(forward[1], horizontal)))
    angles["palm_yaw"] = float(np.degrees(np.arctan2(forward[0], forward[2])))
    return {name: angles[name] for name in JOINT_NAMES}
