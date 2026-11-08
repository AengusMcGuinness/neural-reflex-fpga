"""Tests use synthetic pose metadata, never private recordings."""

import json
from pathlib import Path
import tempfile
import unittest

from neural_reflex.label_audit import audit_labels


def _frame(t, target, followed, rnd=None, cue="reach", landmarks=None):
    return {"host_time_s": t, "target": target, "followed": followed,
            "round": rnd, "cue": cue, "world_landmarks": landmarks,
            "visibility": [1.0] * 33 if landmarks is not None else None}


def _target(shoulder, elbow, forward=.4, up=-.3):
    return {"shoulder_lift": shoulder, "elbow_flex": elbow,
            "forward": forward, "up": up}


def _write(folder, rows):
    with (folder / "arm.jsonl").open("w") as sink:
        for row in rows:
            sink.write(json.dumps(row) + "\n")


class LabelAuditTests(unittest.TestCase):
    def test_same_frame_error_round_coverage_and_private_cue_redaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rows = [_frame(1.0, _target(10, 20), _target(12, 18), 1),
                    _frame(1.05, _target(20, 30), None, 1),
                    _frame(1.10, _target(30, 40), _target(28, 44), 2, "private label"),
                    _frame(1.30, _target(40, 50), _target(44, 46), 2)]
            _write(folder, rows)
            report, data = audit_labels(folder)
            self.assertEqual(report["camera_frames"], 4)
            self.assertEqual(report["followed_frames"], 3)
            self.assertEqual(report["camera_gaps_over_150ms"], 1)
            self.assertAlmostEqual(report["comparison"]["shoulder_lift"]["mae"], 8 / 3, places=3)
            self.assertAlmostEqual(report["comparison"]["shoulder_lift"]["bias_measured_minus_target"], 4 / 3, places=3)
            self.assertEqual(report["rounds"]["1"]["followed_fraction"], .5)
            self.assertEqual(report["rounds"]["2"]["followed_fraction"], 1.0)
            self.assertEqual(report["benchmark_label_validity"]["shoulder_lift"]["valid_frames"], 0)
            self.assertEqual(data["phases"].tolist()[2], "other")
            self.assertNotIn("private label", json.dumps(report))

    def test_original_benchmark_label_coverage_is_independently_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            points = [[0.0, 0.0, 0.0] for _ in range(33)]
            points[11], points[12] = [-.3, 1, 0], [.3, 1, 0]
            points[23], points[24] = [-.3, 0, 0], [.3, 0, 0]
            points[14], points[16] = [.6, .6, 0], [.9, .6, 0]
            points[18], points[20] = [1, .6, -.1], [1, .6, .1]
            rows = [_frame(1.0, _target(10, 20), None, 1, landmarks=points),
                    _frame(1.05, _target(20, 30), _target(22, 28), 1)]
            _write(folder, rows)
            report, _ = audit_labels(folder)
            self.assertEqual(report["landmark_frames"], 1)
            self.assertEqual(report["followed_frames"], 1)
            self.assertEqual(report["benchmark_label_validity"]["shoulder_lift"]["valid_frames"], 1)
            self.assertEqual(report["benchmark_label_validity"]["elbow_flex"]["valid_frames"], 1)

    def test_missing_target_or_backward_clock_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write(folder, [_frame(1.0, _target(10, 20), None),
                            _frame(1.0, _target(20, 30), None)])
            with self.assertRaisesRegex(ValueError, "increasing"):
                audit_labels(folder)
            _write(folder, [_frame(1.0, _target(10, 20), None),
                            _frame(1.1, None, None)])
            with self.assertRaisesRegex(ValueError, "target"):
                audit_labels(folder)


if __name__ == "__main__":
    unittest.main()
