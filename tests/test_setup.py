import json
import unittest
from pathlib import Path

import numpy as np

from neural_reflex.align import (estimate_pose_lag_s, expand_batch, fit_band_clock,
                                 mark_sequence_gaps)
from neural_reflex.angles import joint_angles
from neural_reflex.kinesis import (assert_local_stream_url, emg_records_from_events,
                                   iter_emg_records, parse_sse)
from neural_reflex.prepare import build_examples
from neural_reflex.ridge_decoder import fit_ridge, predict_angles, split_by_block
from neural_reflex.session import create_session
from neural_reflex.signal import condition_emg, lagged_features, window_features
from neural_reflex.tcn import AngleTCN


def _landmarks():
    points = np.zeros((21, 3))
    points[1] = (0.5, 0, 0)
    points[2] = (1, 0, 0)
    points[3] = (1.5, 0, 0)
    points[4] = (2, 0, 0)
    points[5] = (0, 1, 0)
    points[6] = (0, 2, 0)
    points[7] = (1, 2, 0)
    points[8] = (1, 3, 0)
    points[9] = (0, 0, 1)
    points[10] = (0, 0, 2)
    points[11] = (0, 0, 3)
    points[12] = (0, 0, 4)
    points[13] = (0.3, 1, 0)
    points[14] = (0.3, 2, 0)
    points[15] = (0.3, 3, 0)
    points[16] = (0.3, 4, 0)
    points[17] = (1, 0, 0)
    points[18] = (1, 1, 0)
    points[19] = (1, 2, 0)
    points[20] = (1, 3, 0)
    return points


class SetupTests(unittest.TestCase):
    def test_sse_parser_keeps_raw_emg_batch_and_connection(self):
        values = [[float(sample), 0, 0, 0, 0, 0, 0, 0] for sample in range(16)]
        status = {"schema_version": 1, "type": "status", "stream_session_id": "s",
                  "event_sequence": 1, "host_time_s": 10.0,
                  "payload": {"band_connection_id": "c1"}}
        emg = {"schema_version": 1, "type": "emg", "stream_session_id": "s",
               "event_sequence": 2, "host_time_s": 10.01, "band_timestamp_us": 1000000,
               "payload": {"sequence": 7, "values": values, "sample_rate_hz": 2048,
                           "sample_interval_us": 488.28125, "channels": 8,
                           "samples_per_batch": 16, "layout": "sample_major"}}
        text = "\n".join(("event: status", "data: " + json.dumps(status), "",
                          "event: emg", "data: " + json.dumps(emg), ""))
        records = emg_records_from_events(parse_sse(text))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["band_connection_id"], "c1")
        self.assertEqual(records[0]["values"][0][0], 0.0)
        self.assertEqual(records[0]["values"][15][0], 15.0)
        self.assertEqual(records[0]["sequence"], 7)

    def test_line_stream_yields_an_emg_batch_before_the_feed_ends(self):
        values = [[0.0] * 8 for _ in range(16)]
        status = {"type": "status", "stream_session_id": "s", "event_sequence": 1,
                  "host_time_s": 10.0, "payload": {"band_connection_id": "c1"}}
        emg = {"type": "emg", "stream_session_id": "s", "event_sequence": 2,
               "host_time_s": 10.01, "band_timestamp_us": 1000,
               "payload": {"sequence": 1, "values": values, "sample_rate_hz": 2048,
                           "sample_interval_us": 488.28125, "channels": 8,
                           "samples_per_batch": 16, "layout": "sample_major"}}
        lines = ["event: status\n", "data: " + json.dumps(status) + "\n", "\n",
                 "event: emg\n", "data: " + json.dumps(emg) + "\n", "\n"]

        def source():
            yield from lines

        records = list(iter_emg_records(source()))
        self.assertEqual(records[0]["band_connection_id"], "c1")
        self.assertEqual(len(records[0]["values"]), 16)

    def test_remote_stream_url_is_refused(self):
        with self.assertRaises(ValueError):
            assert_local_stream_url("http://example.com/events?token=abc")
        assert_local_stream_url("http://127.0.0.1:8766/events?token=abc")

    def test_band_clock_line_stamps_each_sample(self):
        slope, intercept = fit_band_clock([0, 1_000_000], [5.0, 6.0])
        record = {"band_timestamp_us": 1_000_000, "sample_interval_us": 488.28125,
                  "sequence": 7, "gap_before": False,
                  "values": [[float(i)] + [0.0] * 7 for i in range(16)]}
        samples = expand_batch(record, slope, intercept)
        self.assertAlmostEqual(samples[0]["host_time_s"], 6.0)
        self.assertAlmostEqual(samples[0]["band_timestamp_us"], 1_000_000)
        self.assertAlmostEqual(samples[15]["band_timestamp_us"], 1_000_000 + 15 * 488.28125)
        self.assertEqual(samples[15]["emg"][0], 15.0)

    def test_sequence_gap_is_marked_without_crossing_reconnects(self):
        records = [
            {"band_connection_id": "c1", "sequence": 4},
            {"band_connection_id": "c1", "sequence": 5},
            {"band_connection_id": "c1", "sequence": 8},
            {"band_connection_id": "c2", "sequence": 1},
        ]
        marked = mark_sequence_gaps(records)
        self.assertEqual([row["gap_before"] for row in marked], [False, False, True, False])

    def test_pose_lag_finds_a_later_video_pulse(self):
        times = np.linspace(0, 2, 400)
        envelope = ((times > 1.0) & (times < 1.05)).astype(float)
        speed = ((times > 1.10) & (times < 1.15)).astype(float)
        lag = estimate_pose_lag_s(times, envelope, times, speed, search_s=0.3, step_s=0.01)
        self.assertAlmostEqual(lag, 0.10, delta=0.011)

    def test_straight_and_bent_fingers_have_expected_angles(self):
        angles = joint_angles(_landmarks())
        self.assertAlmostEqual(angles["index_mcp"], 0.0, places=5)
        self.assertAlmostEqual(angles["index_pip"], 90.0, places=5)
        self.assertAlmostEqual(angles["thumb_opposition"], 90.0, places=5)
        self.assertAlmostEqual(angles["finger_spread"], 90.0, places=5)
        self.assertAlmostEqual(angles["palm_pitch"], 0.0, places=5)
        self.assertAlmostEqual(angles["palm_yaw"], 0.0, places=5)
        self.assertEqual(len(angles), 14)

    def test_filter_is_causal_and_rejects_dc_and_mains(self):
        rate = 2048
        seconds = np.arange(rate * 2) / rate
        prefix = 1000
        original = np.sin(2 * np.pi * 100 * seconds)
        changed = original.copy()
        changed[prefix:] += 3
        original = np.column_stack([original] * 8)
        changed = np.column_stack([changed] * 8)
        self.assertTrue(np.allclose(condition_emg(original, rate)[:prefix],
                                    condition_emg(changed, rate)[:prefix]))
        steady = slice(rate, None)
        mains = np.sin(2 * np.pi * 60 * seconds)
        tone = np.sin(2 * np.pi * 100 * seconds)
        offset = np.full_like(seconds, 30000.0)
        mains_rms = np.sqrt(np.mean(condition_emg(np.column_stack([mains] * 8), rate)[steady, 0] ** 2))
        tone_rms = np.sqrt(np.mean(condition_emg(np.column_stack([tone] * 8), rate)[steady, 0] ** 2))
        offset_mean = np.mean(np.abs(condition_emg(np.column_stack([offset] * 8), rate)[steady, 0]))
        self.assertLess(mains_rms, 0.05)
        self.assertGreater(tone_rms, 0.3)
        self.assertLess(offset_mean, 1.0)

    def test_window_features_match_hand_computed_values(self):
        window = np.zeros((4, 8))
        window[:, 0] = [0, 1, 2, 3]
        features = window_features(window)
        self.assertEqual(features.shape, (40,))
        self.assertAlmostEqual(features[0], 1.5)
        self.assertAlmostEqual(features[8], 1.0)
        self.assertAlmostEqual(features[16], np.sqrt(3.5))
        crossing = np.zeros((4, 8))
        crossing[:, 0] = [1, -1, 1, -1]
        crossed = window_features(crossing)
        self.assertAlmostEqual(crossed[24], 1.0)
        self.assertAlmostEqual(crossed[32], 1.0)

    def test_lag_stack_uses_older_feature_rows(self):
        rows = np.arange(10, dtype=float)[:, None] * np.ones((1, 40))
        stacked = lagged_features(rows, index=5, lags_hops=(0, 2))
        self.assertEqual(stacked[0], 5.0)
        self.assertEqual(stacked[40], 3.0)
        self.assertIsNone(lagged_features(rows, index=1, lags_hops=(0, 2)))

    def test_held_out_block_does_not_train_the_ridge_decoder(self):
        features = []
        targets = []
        blocks = []
        for block, flip in ((0, False), (1, True)):
            for value, angle in ((0.0, 0.0), (10.0, 90.0)):
                for _ in range(12):
                    features.append([value, 0.0])
                    targets.append([90.0 - angle if flip else angle])
                    blocks.append(block)
        features = np.asarray(features)
        targets = np.asarray(targets)
        train, _, test = split_by_block(blocks, test_blocks={1}, validation_blocks=set())
        model = fit_ridge(features, targets, train, regularization=0.01)
        predictions = predict_angles(model, features[test])
        mae = np.mean(np.abs(predictions - targets[test]))
        self.assertGreater(mae, 45)

    def test_label_delay_pairs_emg_with_later_pose(self):
        times = np.arange(0, 2.01, 0.1)
        emg = np.zeros((len(times), 8))
        samples = [{"host_time_s": float(t), "emg": row.tolist(), "valid": True}
                   for t, row in zip(times, emg)]
        poses = [{"host_time_s": 0.5, "angles": {"index_mcp": 10.0}, "confidence": 1.0},
                 {"host_time_s": 1.0, "angles": {"index_mcp": 20.0}, "confidence": 1.0}]
        examples = build_examples(samples, poses, window_samples=5, hop_samples=5,
                                  lags_hops=(0,), label_delay_s=0.1, min_confidence=0.5,
                                  max_pose_gap_s=0.2, joint_names=("index_mcp",))
        by_time = {round(row["window_end_s"], 2): row["target"][0] for row in examples}
        self.assertEqual(by_time[0.4], 10.0)
        self.assertEqual(by_time[0.9], 20.0)

    def test_low_confidence_pose_does_not_invent_angles(self):
        from neural_reflex.pose import pose_row
        missing = pose_row(1.0, "rest", _landmarks().tolist(), 0.1)
        tracked = pose_row(1.0, "rest", _landmarks().tolist(), 0.9)
        self.assertIsNone(missing["angles"])
        self.assertAlmostEqual(tracked["angles"]["index_pip"], 90.0, places=5)

    def test_session_manifest_records_wrist_and_joints(self):
        root = Path(self.id().replace(".", "_"))
        try:
            path = create_session(root, subject="wearer", wrist="right")
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["wrist"], "right")
            self.assertIn("index_mcp", manifest["joints"])
            self.assertTrue((path / "blocks").is_dir())
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)

    def test_tcn_output_ignores_future_samples(self):
        import torch
        model = AngleTCN(n_joints=14)
        model.eval()
        source = torch.randn(1, 8, 50)
        changed = source.clone()
        changed[:, :, 30:] = torch.randn(1, 8, 20)
        with torch.no_grad():
            original = model.forward_sequence(source)
            future = model.forward_sequence(changed)
            last = model(source)
        self.assertEqual(tuple(original.shape), (1, 50, 14))
        self.assertTrue(torch.allclose(original[:, :30], future[:, :30]))
        self.assertTrue(torch.allclose(last, original[:, -1]))


if __name__ == "__main__":
    unittest.main()
