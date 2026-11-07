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

# The guided session shown at the bottom of the camera window: (step id, instruction, seconds).
# Every recorded frame stores the step id as "cue", so it doubles as a coarse label.
PROTOCOL = (
    ("ready", "Get ready: stand back so shoulders, hips and your whole arm are in view", 5),
    ("sync", "SYNC: make 4 sharp fist clenches", 6),
    ("rest", "Rest: arm hanging down, relaxed", 4),
    ("lift_forward", "Raise your arm forward to shoulder height and back down, slowly", 20),
    ("sweep_side", "Sweep your arm out to the side and back", 20),
    ("elbow", "Bend and straighten your elbow", 20),
    ("wrist_flex", "Bend your wrist up and down", 15),
    ("wrist_roll", "Rotate your forearm: palm up, palm down", 15),
    ("free", "Free movement: reach around naturally", 20),
    ("home", "Hold your arm straight up (robot home pose)", 5),
    ("rest_end", "Rest: arm down. Recording stops automatically", 3),
)


def protocol_step(elapsed_s, protocol=PROTOCOL):
    """Return (index, step id, instruction, seconds left in step, seconds left overall) or None when done."""
    total = sum(step[2] for step in protocol)
    start = 0.0
    for index, (key, text, seconds) in enumerate(protocol):
        if elapsed_s < start + seconds:
            return index, key, text, start + seconds - elapsed_s, total - elapsed_s
        start += seconds
    return None


def _clock(seconds):
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


class StreamWatch:
    """Tell whether the sibling EMG/motion files are still growing, i.e. those recorders are live."""

    def __init__(self, folder: Path, names=("emg.jsonl", "motion.jsonl")):
        self.paths = {name.split(".")[0]: folder / name for name in names}
        self.sizes = {key: -1 for key in self.paths}
        self.live = {key: False for key in self.paths}
        self.checked = 0.0

    def update(self, now):
        if now - self.checked < 1.0:
            return self.live
        self.checked = now
        for key, path in self.paths.items():
            size = path.stat().st_size if path.exists() else -1
            self.live[key] = size > max(self.sizes[key], 0)
            self.sizes[key] = size
        return self.live


def draw_guide(frame, step, streams, pose_ok, block):
    """Bottom bar: instruction, time left, progress, and which data streams are recording."""
    import cv2
    h, w = frame.shape[:2]
    bar = 118
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    green, grey, red, white = (90, 220, 90), (170, 170, 170), (70, 70, 240), (255, 255, 255)
    if step is None:
        cv2.putText(frame, "Done. Saving...", (20, h - 70), font, 0.9, white, 2, cv2.LINE_AA)
        return
    index, _, text, step_left, total_left = step
    # Shrink long instructions to fit narrower cameras (some ignore the requested 960 px width).
    scale = min(0.78, 0.78 * (w - 40) / max(1, cv2.getTextSize(text, font, 0.78, 2)[0][0]))
    cv2.putText(frame, text, (20, h - 78), font, scale, white, 2, cv2.LINE_AA)
    cv2.putText(frame, f"step {index + 1}/{len(PROTOCOL)}   {int(step_left) + 1} s left in step   "
                f"session {_clock(total_left)} left", (20, h - 46), font, 0.55, grey, 1, cv2.LINE_AA)
    total = sum(s[2] for s in PROTOCOL)
    done = (total - total_left) / total
    cv2.rectangle(frame, (20, h - 28), (w - 20, h - 20), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, h - 28), (20 + int((w - 40) * done), h - 20), green, -1)
    status = [("EMG", streams.get("emg")), ("motion", streams.get("motion")), ("arm pose", pose_ok)]
    x = w - 20
    for name, ok in reversed(status):
        label = f"{name} {'ok' if ok else '--'}"
        size = cv2.getTextSize(label, font, 0.5, 1)[0][0]
        x -= size
        cv2.putText(frame, label, (x, h - 98), font, 0.5, green if ok else red, 1, cv2.LINE_AA)
        x -= 18
    cv2.circle(frame, (24, h - 103), 6, red, -1)
    cv2.putText(frame, f"REC  {block}", (38, h - 98), font, 0.5, white, 1, cv2.LINE_AA)


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


def side_angles(world_landmarks, visibility, side):
    """Angles for one arm, or None unless its shoulder, elbow, wrist and the torso are visible."""
    index = SIDES[side]
    needed = [index["shoulder"], index["elbow"], index["wrist"], LEFT_SHOULDER, RIGHT_SHOULDER,
              LEFT_HIP, RIGHT_HIP]
    if min(visibility[i] for i in needed) < MIN_VISIBILITY:
        return None
    try:
        return arm_angles(world_landmarks, side)
    except ValueError:
        return None


def arm_row(host_time_s, block, side, world_landmarks, visibility, cue=None):
    """One recorded frame. Both arms are stored because a mirrored camera swaps MediaPipe's
    left/right labels; "angles" is the requested side, "angles_by_side" has both."""
    row = {"host_time_s": host_time_s, "block": block, "side": side, "cue": cue,
           "world_landmarks": world_landmarks, "visibility": visibility, "angles": None,
           "angles_by_side": {"left": None, "right": None}}
    if world_landmarks is None:
        return row
    for name in SIDES:
        row["angles_by_side"][name] = side_angles(world_landmarks, visibility, name)
    row["angles"] = row["angles_by_side"][side]
    return row


def ensure_pose_model(path):
    from .pose import ensure_hand_model
    return ensure_hand_model(path, url=POSE_LANDMARKER_URL)


def record_arm_pose(output, camera_index, block, model_path, side, guided=True):
    """Record until q is pressed, or until the guided protocol finishes."""
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
    streams = StreamWatch(destination.parent)
    guide_started = None
    try:
        with destination.open("w", encoding="utf-8") as sink:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                host_time_s = time.monotonic()
                if guide_started is None:
                    guide_started = host_time_s
                step = protocol_step(host_time_s - guide_started) if guided else None
                if guided and step is None:
                    break
                timestamp_ms = max(int((host_time_s - started) * 1000), last_ms + 1)
                last_ms = timestamp_ms
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = landmarker.detect_for_video(image, timestamp_ms)
                world, visibility = None, None
                if result.pose_world_landmarks:
                    world = [[p.x, p.y, p.z] for p in result.pose_world_landmarks[0]]
                    visibility = [float(p.visibility) for p in result.pose_landmarks[0]]
                row = arm_row(host_time_s, block, side, world, visibility,
                              cue=step[1] if step else None)
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                count += 1
                if result.pose_landmarks:
                    h, w = frame.shape[:2]
                    pts = [(int(result.pose_landmarks[0][i].x * w), int(result.pose_landmarks[0][i].y * h))
                           for i in chain]
                    for a, b in zip(pts, pts[1:]):
                        cv2.line(frame, a, b, (80, 220, 60), 4)
                other = "left" if side == "right" else "right"
                shown = row["angles"] or row["angles_by_side"][other]
                if shown:
                    a = shown
                    pan = "--" if a["shoulder_pan"] is None else f"{a['shoulder_pan']:.0f}"
                    label = (f"lift {a['shoulder_lift']:.0f}  pan {pan}  elbow {a['elbow_flex']:.0f}"
                             f"  wrist {a['wrist_flex']:.0f}")
                    if not row["angles"]:
                        label += f"  (seen as {other} arm: mirrored camera? both arms are saved)"
                else:
                    label = "arm not fully visible: show shoulders, hips and the whole arm"
                cv2.putText(frame, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 220, 60), 2)
                if guided:
                    draw_guide(frame, step, streams.update(host_time_s), shown is not None, block)
                cv2.imshow("neural reflex arm pose (q to stop)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        landmarker.close()
    return count
