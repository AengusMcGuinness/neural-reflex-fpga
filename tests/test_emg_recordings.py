"""Segmented recording loader preserves samples and prevents gap-spanning input."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from neural_reflex.emg_benchmark import extract_features, load_recording
from neural_reflex.emg_recordings import load_segments


RATE = 2048
BATCH_SECONDS = 16 / RATE
VALUES = [[32768 + channel for channel in range(8)] for _ in range(16)]


def batches(count, *, start_sequence=0, start_band_us=1_000_000,
            start_host_s=100.0, connection="first"):
    out = []
    for index in range(count):
        out.append({"host_time_s": start_host_s + index * BATCH_SECONDS,
                    "band_timestamp_us": start_band_us + round(index * BATCH_SECONDS * 1e6),
                    "stream_session_id": "stream", "band_connection_id": connection,
                    "sequence": start_sequence + index,
                    "sample_interval_us": 1e6 / RATE, "sample_rate_hz": RATE,
                    "values": VALUES})
    return out


def pose_rows(start_s, end_s):
    p = np.zeros((33, 3))
    p[11], p[12] = [-.2, 0, 0], [.2, 0, 0]
    p[23], p[24] = [-.2, -1, 0], [.2, -1, 0]
    p[14], p[16] = [.2, -.5, 0], [.2, -1, 0]
    p[18], p[20] = [.2, -1.1, 0], [.2, -1.1, 0]
    return [{"host_time_s": float(t), "world_landmarks": p.tolist(),
             "visibility": [1.0] * 33, "cue": "reach", "round": 1,
             "target": {"step": "reach", "spot": 2}}
            for t in np.arange(start_s, end_s, 1 / 30)]


def write_block(folder, emg, pose):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in (("emg.jsonl", emg), ("arm.jsonl", pose)):
        with (folder / name).open("w") as out:
            for row in rows:
                out.write(json.dumps(row) + "\n")


class SegmentedRecordingTests(unittest.TestCase):
    def test_clean_block_matches_original_loader_sample_for_sample(self):
        emg = batches(120)
        pose = pose_rows(100, 101.2)
        with tempfile.TemporaryDirectory() as folder:
            write_block(folder, emg, pose)
            old = load_recording(folder, "right")
            segments, metadata = load_segments(folder, "right")
        self.assertEqual(metadata["segment_count"], 1)
        self.assertEqual(metadata["clock_epoch_count"], 1)
        self.assertEqual(metadata["sequence_missing_batches"], 0)
        np.testing.assert_array_equal(segments[0]["raw"], old["raw"])
        np.testing.assert_allclose(segments[0]["times"], old["times"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(segments[0]["pose_targets"], old["pose_targets"],
                                   atol=0, rtol=0, equal_nan=True)
        np.testing.assert_array_equal(segments[0]["cues"], old["cues"])
        self.assertEqual(segments[0]["rounds"][0], 1)
        self.assertEqual(segments[0]["steps"][0], "reach")
        self.assertEqual(segments[0]["spots"][0], 2)

    def test_missing_batches_split_features_without_fabricating_samples(self):
        first = batches(500)
        second = batches(500, start_sequence=503,
                         start_band_us=1_000_000 + round(503 * BATCH_SECONDS * 1e6),
                         start_host_s=100 + 503 * BATCH_SECONDS)
        emg = first + second
        pose = pose_rows(100, 100 + 1003 * BATCH_SECONDS)
        with tempfile.TemporaryDirectory() as folder:
            write_block(folder, emg, pose)
            segments, metadata = load_segments(folder)
        self.assertEqual(metadata["sequence_missing_batches"], 3)
        self.assertEqual(metadata["segment_count"], 2)
        self.assertEqual(metadata["clock_epoch_count"], 1)
        self.assertEqual(metadata["sample_count"], 1000 * 16)
        self.assertEqual(metadata["boundaries"][0]["reasons"], ["sequence_gap"])
        self.assertAlmostEqual(metadata["sequence_missing_duration_s"], 3 * BATCH_SECONDS)
        self.assertAlmostEqual(metadata["retained_sample_duration_s"], 1000 * BATCH_SECONDS)
        for segment in segments:
            self.assertEqual(segment["raw"].shape, (500 * 16, 8))
            x, ends, starts = extract_features(segment["raw"])
            self.assertGreater(len(x), 0)
            self.assertGreaterEqual(starts.min(), 0)
            self.assertLess(ends.max(), len(segment["raw"]))
            self.assertTrue(np.all(np.diff(segment["times"]) > 0))
        self.assertGreater(segments[1]["times"][0] - segments[0]["times"][-1],
                           3 * BATCH_SECONDS)

    def test_connection_change_gets_separate_clock_and_records_fallback(self):
        first = batches(25)
        # A short, bursty first epoch has an implausible unconstrained slope.
        for index, row in enumerate(first):
            row["host_time_s"] = 100 + index * BATCH_SECONDS * .86
        second = batches(120, start_sequence=0, start_band_us=9_000_000,
                         start_host_s=110, connection="second")
        with tempfile.TemporaryDirectory() as folder:
            write_block(folder, first + second, pose_rows(100, 111))
            segments, metadata = load_segments(folder)
        self.assertEqual(metadata["connection_epoch_count"], 2)
        self.assertEqual(metadata["clock_epoch_count"], 2)
        self.assertEqual(metadata["segment_count"], 2)
        self.assertEqual(metadata["sequence_missing_batches"], 0)
        self.assertEqual(metadata["clocks"][0]["slope"], 1.0)
        self.assertIn("fallback", metadata["clocks"][0]["fit"])
        self.assertEqual(metadata["boundaries"][0]["reasons"], ["connection_transition"])
        self.assertGreater(segments[1]["times"][0] - segments[0]["times"][-1], 9)

    def test_single_batch_connection_is_retained_without_fitted_drift(self):
        first = batches(1)
        second = batches(120, start_sequence=0, start_band_us=9_000_000,
                         start_host_s=110, connection="second")
        with tempfile.TemporaryDirectory() as folder:
            write_block(folder, first + second, pose_rows(100, 111))
            segments, metadata = load_segments(folder)
        self.assertEqual(len(segments[0]["raw"]), 16)
        self.assertEqual(metadata["clocks"][0]["slope"], 1.0)
        self.assertIn("single-batch", metadata["clocks"][0]["fit"])

    def test_unit_mode_keeps_unit_slope_despite_startup_receipt_drift(self):
        emg = batches(1000)
        for index, row in enumerate(emg):
            t = index * BATCH_SECONDS
            row["host_time_s"] += .05 * np.exp(-t)
        pose = pose_rows(100, 108)
        with tempfile.TemporaryDirectory() as folder:
            write_block(folder, emg, pose)
            ols_segments, ols_meta = load_segments(folder, clock_mode="ols")
            unit_segments, unit_meta = load_segments(folder, clock_mode="unit")
        self.assertEqual(unit_meta["clock_mode"], "unit")
        self.assertAlmostEqual(unit_meta["clocks"][0]["slope"], 1.0)
        self.assertLess(ols_meta["clocks"][0]["slope"], .995)
        self.assertAlmostEqual(unit_meta["clocks"][0]["candidate_ols_slope"],
                               ols_meta["clocks"][0]["slope"])
        self.assertIn("unit slope", unit_meta["clocks"][0]["fit"])
        self.assertEqual(len(ols_segments[0]["raw"]), len(unit_segments[0]["raw"]))

    def test_invalid_clock_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "clock_mode"):
            load_segments("unused", clock_mode="label_tuned")

    def test_device_timestamp_discontinuity_splits_clock_without_sequence_loss(self):
        first = batches(120)
        second = batches(120, start_sequence=120,
                         start_band_us=1_000_000 + round(120 * BATCH_SECONDS * 1e6) + 1_000_000,
                         start_host_s=100 + 120 * BATCH_SECONDS + 1)
        with tempfile.TemporaryDirectory() as folder:
            write_block(folder, first + second, pose_rows(100, 103))
            segments, metadata = load_segments(folder)
        self.assertEqual(metadata["segment_count"], 2)
        self.assertEqual(metadata["clock_epoch_count"], 2)
        self.assertEqual(metadata["connection_epoch_count"], 1)
        self.assertEqual(metadata["sequence_missing_batches"], 0)
        self.assertEqual(metadata["boundaries"][0]["reasons"],
                         ["device_clock_discontinuity"])
        self.assertEqual(len(segments[0]["raw"]) + len(segments[1]["raw"]), 240 * 16)

    def test_duplicate_sequence_and_malformed_raw_are_rejected(self):
        original = batches(3)
        with tempfile.TemporaryDirectory() as folder:
            rows = [dict(row) for row in original]
            rows[1]["sequence"] = rows[0]["sequence"]
            write_block(folder, rows, pose_rows(100, 100.2))
            with self.assertRaisesRegex(ValueError, "duplicate or backward"):
                load_segments(folder)
            rows = [dict(row) for row in original]
            rows[1]["values"] = [[0] * 7] * 16
            write_block(folder, rows, pose_rows(100, 100.2))
            with self.assertRaises(ValueError):
                load_segments(folder)


if __name__ == "__main__":
    unittest.main()
