"""Aggregate recording audit tests use small, synthetic JSONL blocks only."""

import json
from pathlib import Path
import tempfile
import unittest

from neural_reflex.recording_audit import audit_block


def _write(folder, name, rows):
    with (folder / name).open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row) + "\n")


def _batch(sequence, time_s, *, stream="secret-stream", connection="secret-band"):
    return {"host_time_s": time_s, "band_timestamp_us": int(time_s * 1_000_000),
            "stream_session_id": stream, "band_connection_id": connection,
            "sequence": sequence, "sample_rate_hz": 2048,
            "values": [[0] * 8 for _ in range(16)]}


def _frame(time_s, angles=None, cue="rest"):
    return {"host_time_s": time_s, "cue": cue,
            "world_landmarks": [[0.0, 0.0, 0.0]], "angles": angles,
            "angles_by_side": {"right": angles}}


class RecordingAuditTests(unittest.TestCase):
    def test_counts_wrap_gap_and_keeps_stream_ids_out_of_report(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            _write(folder, "emg.jsonl", [_batch(2**32 - 2, 1.0),
                                         _batch(2**32 - 1, 1.008),
                                         _batch(0, 1.016), _batch(2, 1.032)])
            _write(folder, "arm.jsonl", [_frame(1.0, {"shoulder_lift": 0.0,
                                                        "shoulder_pan": None}),
                                        _frame(1.033, {"shoulder_lift": 90.0,
                                                         "shoulder_pan": 20.0}, "lift_forward")])
            _write(folder, "motion.jsonl", [{"host_time_s": 1.01, "type": "gyro"}])
            report = audit_block(folder)
            self.assertEqual((report["emg_batches"], report["emg_samples"], report["pose_frames"]),
                             (4, 64, 2))
            self.assertEqual(report["source_sample_rate_hz"], 2048)
            self.assertEqual(report["sequence_anomalies"]["wraps"], 1)
            self.assertEqual(report["sequence_anomalies"]["gap_boundaries"], 1)
            self.assertEqual(report["sequence_anomalies"]["missing_batches_estimate"], 1)
            self.assertEqual(report["sequence_anomalies"]["backsteps_or_resets"], 0)
            self.assertEqual(report["per_metric_validity"]["shoulder_pan"]["valid_frames"], 1)
            self.assertEqual(report["angle_distributions"]["shoulder_lift"]["max"], 90.0)
            self.assertEqual(report["motion_events"], {"gyro": 1})
            self.assertNotIn("secret-stream", json.dumps(report))
            self.assertNotIn("secret-band", json.dumps(report))

    def test_epoch_change_does_not_create_false_sequence_gap(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            _write(folder, "emg.jsonl", [_batch(90, 1.0, connection="first"),
                                         _batch(91, 1.008, connection="first"),
                                         _batch(1, 1.016, connection="second")])
            report = audit_block(folder)
            self.assertEqual(report["epochs"]["band_connections"], 2)
            self.assertEqual(report["sequence_anomalies"]["epoch_transitions"], 1)
            self.assertEqual(report["sequence_anomalies"]["gap_boundaries"], 0)

    def test_duplicate_and_backstep_are_counted(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            _write(folder, "emg.jsonl", [_batch(10, 1.0), _batch(10, 1.008),
                                         _batch(9, 1.016)])
            report = audit_block(folder)
            self.assertEqual(report["sequence_anomalies"]["duplicates"], 1)
            self.assertEqual(report["sequence_anomalies"]["backsteps_or_resets"], 1)

    def test_camera_only_and_missing_angles_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            _write(folder, "arm.jsonl", [_frame(1.0),
                                        _frame(1.033, {"elbow_flex": 30.0,
                                                         "shoulder_pan": None}, "other private cue")])
            report = audit_block(folder)
            self.assertTrue(report["camera_only"])
            self.assertEqual(report["emg_samples"], 0)
            self.assertIsNone(report["source_sample_rate_hz"])
            self.assertEqual(report["per_metric_validity"]["elbow_flex"]["fraction"], 0.5)
            self.assertEqual(report["per_metric_validity"]["shoulder_pan"]["valid_frames"], 0)
            self.assertEqual(report["cues"]["other"]["frames"], 1)
            self.assertNotIn("private cue", json.dumps(report))

    def test_all_missing_pose_labels_still_report_each_arm_metric(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            _write(folder, "arm.jsonl", [_frame(1.0), _frame(1.033)])
            report = audit_block(folder)
            self.assertEqual(set(report["per_metric_validity"]),
                             {"shoulder_lift", "shoulder_pan", "elbow_flex", "wrist_flex"})
            self.assertTrue(all(item["valid_frames"] == 0
                                for item in report["per_metric_validity"].values()))

    def test_unknown_angle_names_are_not_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            private_name = "private subject label"
            _write(folder, "arm.jsonl", [_frame(1.0, {
                "elbow_flex": 30.0, private_name: 123.0,
            })])
            report = audit_block(folder)
            self.assertEqual(report["per_metric_validity"]["elbow_flex"]["valid_frames"], 1)
            self.assertNotIn(private_name, json.dumps(report))

    def test_malformed_emg_shape_or_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            bad = _batch(1, 1.0)
            bad["values"] = [[0] * 8 for _ in range(15)]
            _write(folder, "emg.jsonl", [bad])
            with self.assertRaisesRegex(ValueError, "16 sample rows"):
                audit_block(folder)
            bad["values"] = [[0] * 8 for _ in range(16)]
            bad["values"][0][0] = float("nan")
            _write(folder, "emg.jsonl", [bad])
            with self.assertRaisesRegex(ValueError, "ADC count"):
                audit_block(folder)


if __name__ == "__main__":
    unittest.main()
