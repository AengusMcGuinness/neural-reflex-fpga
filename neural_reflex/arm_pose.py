"""Webcam arm pose: MediaPipe Pose landmarks turned into robot-like arm angles.

Angles are measured in a torso frame, so leaning or turning toward the camera
changes them much less than raw coordinates would:

  shoulder_lift   0° arm hanging down, 90° raised forward or sideways, 180° straight up
  shoulder_pan    0° upper arm pointing forward, +90° pointing out to your side
                  (None when the upper arm is near vertical, where pan is undefined)
  elbow_flex      0° straight arm, increases as the elbow bends
  wrist_flex      approximate: forearm against the hand's direction (index/pinky points)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

POSE_LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                       "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
ARM_JOINTS = ("shoulder_lift", "shoulder_pan", "elbow_flex", "wrist_flex")
# MediaPipe Pose indices, person's own left/right.
SIDES = {
    "right": {"shoulder": 12, "elbow": 14, "wrist": 16, "pinky": 18, "index": 20},
    "left": {"shoulder": 11, "elbow": 13, "wrist": 15, "pinky": 17, "index": 19},
}
NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 0, 11, 12, 23, 24
MIN_VISIBILITY = 0.5
MIN_PAN_HORIZONTAL = 0.25   # fraction of the upper arm that must be horizontal to define pan


def _unit(vector):
    length = np.linalg.norm(vector)
    if length < 1e-9:
        raise ValueError("zero-length body segment")
    return vector / length


def _angle(first, second):
    return float(np.degrees(np.arccos(np.clip(np.dot(_unit(first), _unit(second)), -1.0, 1.0))))


def torso_frame(points):
    """Unit up, outward-right and forward vectors for the person's torso."""
    shoulders = (points[LEFT_SHOULDER] + points[RIGHT_SHOULDER]) / 2
    hips = (points[LEFT_HIP] + points[RIGHT_HIP]) / 2
    up = _unit(shoulders - hips)
    right = points[RIGHT_SHOULDER] - points[LEFT_SHOULDER]
    right = _unit(right - np.dot(right, up) * up)
    forward = np.cross(up, right)
    # The nose is in front of the shoulders; use it to fix forward's sign.
    if np.dot(points[NOSE] - shoulders, forward) < 0:
        forward = -forward
    return up, right, forward


def arm_angles(world_landmarks, side="right"):
    points = np.asarray(world_landmarks, dtype=np.float64)
    if points.shape != (33, 3):
        raise ValueError("expected 33 pose world landmarks with x, y, z")
    if side not in SIDES:
        raise ValueError("side must be left or right")
    index = SIDES[side]
    up, right, forward = torso_frame(points)
    outward = right if side == "right" else -right
    shoulder, elbow, wrist = (points[index[k]] for k in ("shoulder", "elbow", "wrist"))
    upper, fore = elbow - shoulder, wrist - elbow
    hand = (points[index["index"]] + points[index["pinky"]]) / 2 - wrist
    horizontal = upper - np.dot(upper, up) * up
    pan = None
    if np.linalg.norm(horizontal) >= MIN_PAN_HORIZONTAL * np.linalg.norm(upper):
        pan = float(np.degrees(np.arctan2(np.dot(horizontal, outward), np.dot(horizontal, forward))))
    return {
        "shoulder_lift": _angle(-up, upper),
        "shoulder_pan": pan,
        "elbow_flex": _angle(upper, fore),
        "wrist_flex": _angle(fore, hand),
    }


def arm_row(host_time_s, block, side, world_landmarks, visibility):
    """One recorded frame; angles only when the shoulder, elbow and wrist are clearly visible."""
    row = {"host_time_s": host_time_s, "block": block, "side": side,
           "world_landmarks": world_landmarks, "visibility": visibility, "angles": None}
    if world_landmarks is None:
        return row
    index = SIDES[side]
    needed = [index["shoulder"], index["elbow"], index["wrist"], LEFT_SHOULDER, RIGHT_SHOULDER,
              LEFT_HIP, RIGHT_HIP]
    if min(visibility[i] for i in needed) >= MIN_VISIBILITY:
        try:
            row["angles"] = arm_angles(world_landmarks, side)
        except ValueError:
            pass
    return row


def ensure_pose_model(path):
    from .pose import ensure_hand_model
    return ensure_hand_model(path, url=POSE_LANDMARKER_URL)


def record_arm_pose(output, camera_index, block, model_path, side):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    model = ensure_pose_model(model_path)
    landmarker = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)), running_mode=RunningMode.VIDEO,
        num_poses=1, min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5))
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"could not open camera {camera_index}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    started, last_ms, count = time.monotonic(), -1, 0
    index = SIDES[side]
    chain = [index["shoulder"], index["elbow"], index["wrist"], index["index"]]
    try:
        with destination.open("w", encoding="utf-8") as sink:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                host_time_s = time.monotonic()
                timestamp_ms = max(int((host_time_s - started) * 1000), last_ms + 1)
                last_ms = timestamp_ms
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = landmarker.detect_for_video(image, timestamp_ms)
                world, visibility = None, None
                if result.pose_world_landmarks:
                    world = [[p.x, p.y, p.z] for p in result.pose_world_landmarks[0]]
                    visibility = [float(p.visibility) for p in result.pose_landmarks[0]]
                row = arm_row(host_time_s, block, side, world, visibility)
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                count += 1
                if result.pose_landmarks:
                    h, w = frame.shape[:2]
                    pts = [(int(result.pose_landmarks[0][i].x * w), int(result.pose_landmarks[0][i].y * h))
                           for i in chain]
                    for a, b in zip(pts, pts[1:]):
                        cv2.line(frame, a, b, (80, 220, 60), 4)
                if row["angles"]:
                    a = row["angles"]
                    pan = "--" if a["shoulder_pan"] is None else f"{a['shoulder_pan']:.0f}"
                    label = (f"lift {a['shoulder_lift']:.0f}  pan {pan}  elbow {a['elbow_flex']:.0f}"
                             f"  wrist {a['wrist_flex']:.0f}")
                else:
                    label = f"{side} arm not fully visible: show shoulders, hips and the whole arm"
                cv2.putText(frame, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 220, 60), 2)
                cv2.imshow("neural reflex arm pose (q to stop)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        landmarker.close()
    return count
