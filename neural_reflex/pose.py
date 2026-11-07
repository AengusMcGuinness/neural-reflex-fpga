"""Webcam hand pose rows. Angles are computed only when the hand is tracked."""

import json
import ssl
import subprocess
import time
import urllib.request
from pathlib import Path

from .angles import joint_angles
from .config import HAND_LANDMARKER_URL, MIN_CONFIDENCE


def ensure_hand_model(path, url=HAND_LANDMARKER_URL):
    destination = Path(path)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(url, context=context) as response:
            destination.write_bytes(response.read())
    except OSError:
        subprocess.run(["curl", "-fsSL", "--output", str(destination), url],
                       check=True)
    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(f"model download failed: {destination}")
    return destination


def pose_row(host_time_s, block, world_landmarks, confidence):
    row = {"host_time_s": host_time_s, "block": block, "confidence": confidence,
           "world_landmarks": None, "angles": None}
    if world_landmarks is not None and confidence >= MIN_CONFIDENCE:
        row["world_landmarks"] = world_landmarks
        row["angles"] = joint_angles(world_landmarks)
    return row


def record_pose(output, camera_index, block, model_path):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    model = ensure_hand_model(model_path)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=MIN_CONFIDENCE,
        min_hand_presence_confidence=MIN_CONFIDENCE,
        min_tracking_confidence=MIN_CONFIDENCE,
    )
    landmarker = HandLandmarker.create_from_options(options)
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"could not open camera {camera_index}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_ms = -1
    count = 0
    try:
        with destination.open("w", encoding="utf-8") as sink:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                host_time_s = time.monotonic()
                timestamp_ms = int((host_time_s - started) * 1000)
                if timestamp_ms <= last_ms:
                    timestamp_ms = last_ms + 1
                last_ms = timestamp_ms
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(image, timestamp_ms)
                landmarks = None
                confidence = 0.0
                if result.hand_world_landmarks:
                    landmarks = [[point.x, point.y, point.z]
                                 for point in result.hand_world_landmarks[0]]
                    if result.handedness:
                        confidence = float(result.handedness[0][0].score)
                row = pose_row(host_time_s, block, landmarks, confidence)
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                count += 1
                if row["angles"] is not None:
                    label = f"index PIP {row['angles']['index_pip']:.0f}"
                else:
                    label = "no hand"
                cv2.putText(frame, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 220, 80), 2)
                cv2.imshow("neural reflex pose", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        landmarker.close()
    return count
