"""Run the real arm-recording loop with a fake camera and fake pose model, for every protocol."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from neural_reflex import arm_pose

UP, RIGHT, FORWARD = np.array([0, -1.0, 0]), np.array([-1.0, 0, 0]), np.array([0, 0, -1.0])


def fake_body():
    p = np.zeros((33, 3))
    p[11], p[12] = -0.18 * RIGHT, 0.18 * RIGHT
    p[23], p[24] = -0.1 * RIGHT - 0.5 * UP, 0.1 * RIGHT - 0.5 * UP
    p[0] = 0.25 * UP + 0.08 * FORWARD
    p[14] = p[12] - 0.3 * UP
    p[16] = p[14] + 0.25 * FORWARD
    p[18] = p[20] = p[14] + 0.3 * FORWARD
    world = [SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in p]
    image = [SimpleNamespace(x=0.5, y=0.5, visibility=0.95) for _ in range(33)]
    return SimpleNamespace(pose_world_landmarks=[world], pose_landmarks=[image])


class FakeCapture:
    def __init__(self, index, frames=6):
        self.left = frames

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def read(self):
        self.left -= 1
        if self.left < 0:
            return False, None
        return True, np.zeros((540, 960, 3), dtype=np.uint8)

    def release(self):
        pass


class FakeLandmarker:
    @classmethod
    def create_from_options(cls, options):
        return cls()

    def detect_for_video(self, image, timestamp_ms):
        return fake_body()

    def close(self):
        pass


class RecordLoopTests(unittest.TestCase):
    def run_protocol(self, protocol):
        import importlib
        import cv2
        vision = importlib.import_module("mediapipe.tasks.python.vision")
        tasks = importlib.import_module("mediapipe.tasks.python")
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(cv2, "VideoCapture", FakeCapture), \
                mock.patch.object(cv2, "imshow"), mock.patch.object(cv2, "destroyAllWindows"), \
                mock.patch.object(cv2, "waitKey", return_value=-1), \
                mock.patch.object(vision, "PoseLandmarker", FakeLandmarker), \
                mock.patch.object(arm_pose, "ensure_pose_model", return_value=Path(tmp) / "model.task"), \
                mock.patch.object(tasks, "BaseOptions"), \
                mock.patch.object(vision, "PoseLandmarkerOptions"):
            out = Path(tmp) / "blocks" / "b" / "arm.jsonl"
            count = arm_pose.record_arm_pose(str(out), 0, "b", "model.task", "right", guided=True, protocol=protocol)
            rows = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(count, 6)
        self.assertEqual(len(rows), 6)
        return rows

    def test_follow_protocol_saves_target_labels(self):
        rows = self.run_protocol("follow")
        self.assertTrue(all("target" in r and r["target"]["step"] == "ready" for r in rows))
        self.assertTrue(all(r["followed"] is not None for r in rows))
        self.assertAlmostEqual(rows[0]["followed"]["elbow_flex"], 90, delta=2)

    def test_robot_and_full_protocols_still_record(self):
        for protocol in ("robot", "full"):
            rows = self.run_protocol(protocol)
            self.assertEqual(rows[0]["cue"], "ready")
            self.assertNotIn("target", rows[0])
            self.assertIsNotNone(rows[0]["angles"])


if __name__ == "__main__":
    unittest.main()
