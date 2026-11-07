"""Arm angles from synthetic MediaPipe Pose world landmarks. No camera or model needed."""

import json
import unittest

import numpy as np

from neural_reflex.arm_pose import arm_angles, arm_row
from neural_reflex.kinesis import iter_emg_records

# MediaPipe-like world frame: x to the camera's right, y down, z away from the camera.
# The person faces the camera, so their forward is -z and their right is -x.
UP, RIGHT, FORWARD = np.array([0, -1.0, 0]), np.array([-1.0, 0, 0]), np.array([0, 0, -1.0])


def body(upper_dir, fore_dir, hand_dir=None):
    points = np.zeros((33, 3))
    points[11], points[12] = -0.18 * RIGHT, 0.18 * RIGHT           # shoulders
    points[23], points[24] = -0.1 * RIGHT - 0.5 * UP, 0.1 * RIGHT - 0.5 * UP
    points[0] = 0.25 * UP + 0.08 * FORWARD                          # nose
    elbow = points[12] + 0.3 * np.asarray(upper_dir, dtype=float)
    wrist = elbow + 0.27 * np.asarray(fore_dir, dtype=float)
    hand = wrist + 0.08 * np.asarray(hand_dir if hand_dir is not None else fore_dir, dtype=float)
    points[14], points[16], points[18], points[20] = elbow, wrist, hand, hand
    return points.tolist()


class ArmAngleTests(unittest.TestCase):
    def test_arm_hanging_straight_down(self):
        a = arm_angles(body(-UP, -UP))
        self.assertAlmostEqual(a["shoulder_lift"], 0, delta=0.5)
        self.assertAlmostEqual(a["elbow_flex"], 0, delta=0.5)
        self.assertIsNone(a["shoulder_pan"])           # vertical arm has no pan

    def test_arm_straight_up_like_robot_home(self):
        a = arm_angles(body(UP, UP))
        self.assertAlmostEqual(a["shoulder_lift"], 180, delta=0.5)
        self.assertAlmostEqual(a["elbow_flex"], 0, delta=0.5)

    def test_reach_forward_and_to_the_side(self):
        forward = arm_angles(body(FORWARD, FORWARD))
        self.assertAlmostEqual(forward["shoulder_lift"], 90, delta=0.5)
        self.assertAlmostEqual(forward["shoulder_pan"], 0, delta=0.5)
        side = arm_angles(body(RIGHT, RIGHT))
        self.assertAlmostEqual(side["shoulder_pan"], 90, delta=0.5)

    def test_elbow_and_wrist_bend(self):
        a = arm_angles(body(-UP, FORWARD, hand_dir=UP))
        self.assertAlmostEqual(a["elbow_flex"], 90, delta=0.5)
        self.assertAlmostEqual(a["wrist_flex"], 90, delta=0.5)

    def test_row_skips_angles_when_arm_hidden(self):
        landmarks = body(-UP, -UP)
        visible = [0.9] * 33
        self.assertIsNotNone(arm_row(1.0, "b", "right", landmarks, visible)["angles"])
        visible[14] = 0.1                                 # elbow out of view
        self.assertIsNone(arm_row(1.0, "b", "right", landmarks, visible)["angles"])
        self.assertIsNone(arm_row(1.0, "b", "right", None, None)["angles"])


class GuideTests(unittest.TestCase):
    def test_protocol_steps_and_time_left(self):
        from neural_reflex.arm_pose import PROTOCOL, protocol_step
        total = sum(step[2] for step in PROTOCOL)
        first = protocol_step(0.0)
        self.assertEqual(first[1], "ready")
        self.assertAlmostEqual(first[4], total)
        sync = protocol_step(PROTOCOL[0][2] + 1.0)
        self.assertEqual(sync[1], "sync")
        self.assertAlmostEqual(sync[3], PROTOCOL[1][2] - 1.0)
        self.assertIsNone(protocol_step(total + 0.01))

    def test_rows_carry_the_cue(self):
        self.assertEqual(arm_row(1.0, "b", "right", None, None, cue="elbow")["cue"], "elbow")

    def test_stream_watch_sees_growing_files(self):
        import tempfile
        from pathlib import Path
        from neural_reflex.arm_pose import StreamWatch
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            watch = StreamWatch(folder)
            self.assertFalse(watch.update(10.0)["emg"])
            (folder / "emg.jsonl").write_text("a\n")
            self.assertTrue(watch.update(11.0)["emg"])
            self.assertFalse(watch.update(12.0)["emg"])     # stopped growing
            self.assertFalse(watch.update(13.0)["motion"])

    def test_guide_draws_on_a_frame(self):
        from neural_reflex.arm_pose import draw_guide, protocol_step
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        draw_guide(frame, protocol_step(30.0), {"emg": True, "motion": False}, True, "arm-reach")
        self.assertGreater(frame[-60:].sum(), 0)
        draw_guide(frame, None, {}, False, "arm-reach")


class MotionRecordingTests(unittest.TestCase):
    def test_emg_reader_keeps_connection_and_passes_motion(self):
        hello = {"type": "hello", "payload": {"status": {"band_connection_id": "abc"}}}
        gyro = {"type": "gyro", "payload": {"values": [1, 2, 3]}}
        emg = {"type": "emg", "host_time_s": 1.0, "band_timestamp_us": 5, "event_sequence": 3,
               "stream_session_id": "s", "payload": {"sequence": 7, "channels": 8, "samples_per_batch": 16,
                                                     "layout": "sample_major", "sample_interval_us": 488.28125,
                                                     "sample_rate_hz": 2048, "values": [[0] * 8] * 16}}
        lines = []
        for event in (hello, gyro, emg):
            lines += [f"event: {event['type']}", "data: " + json.dumps(event), ""]
        motion = []
        records = list(iter_emg_records(lines, motion.append))
        self.assertEqual(records[0]["band_connection_id"], "abc")
        self.assertEqual(motion[0]["type"], "gyro")
        self.assertEqual(motion[0]["band_connection_id"], "abc")


if __name__ == "__main__":
    unittest.main()
