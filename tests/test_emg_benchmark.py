"""Causality, split isolation, and missing-label behavior for EMG-only decoding."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from neural_reflex.emg_benchmark import (
    Decoder,
    TCNDecoder,
    extract_features,
    fit_scaler,
    label_values,
    make_tcn,
    pair_segment,
    score,
    split_intervals,
    transform_features,
)


class BenchmarkFeatureTests(unittest.TestCase):
    def test_future_raw_change_cannot_change_past_features(self):
        raw = np.full((6000, 8), 32768.0)
        raw[:, 0] += 100 * np.sin(np.arange(len(raw)) / 6)
        baseline, ends, starts = extract_features(raw)
        changed = raw.copy()
        changed[5000:5100, 0] += 1000
        replay, replay_ends, replay_starts = extract_features(changed)

        np.testing.assert_array_equal(ends, replay_ends)
        np.testing.assert_array_equal(starts, replay_starts)
        np.testing.assert_allclose(baseline[ends < 5000], replay[ends < 5000], atol=0, rtol=0)
        self.assertTrue(np.any(np.abs(baseline[ends >= 5000] - replay[ends >= 5000]) > 1e-6))

    def test_dependency_indices_cover_all_history_windows(self):
        history, hop, window, warmup = 31, 41, 205, 2048
        x, ends, starts = extract_features(np.full((5000, 8), 32768),
                                           history=history, hop_samples=hop,
                                           window_samples=window, warmup_samples=warmup)
        self.assertEqual(x.shape, (len(ends), history, 40))
        np.testing.assert_array_equal(starts, ends - (history - 1) * hop - window + 1)
        self.assertGreaterEqual(int(starts.min()), warmup)
        self.assertLess(int(ends.max()), 5000)
        np.testing.assert_allclose(extract_features(np.full((1000, 8), 32768))[0].shape,
                                   (0, history, 40))

    def test_scaler_uses_only_training_frames(self):
        train = np.stack((np.zeros((31, 40)), np.full((31, 40), 2.0)))
        held_out = np.full((1, 31, 40), 10.0)
        scaler = fit_scaler(train)
        np.testing.assert_allclose(scaler["mean"], 1.0)
        np.testing.assert_allclose(scaler["scale"], 1.0)
        np.testing.assert_allclose(transform_features(train, scaler).mean(axis=(0, 1)), 0.0)
        np.testing.assert_allclose(transform_features(held_out, scaler), 6.0)
        np.testing.assert_allclose(scaler["mean"], 1.0)  # held-out data did not update it


class BenchmarkSplitTests(unittest.TestCase):
    def test_chronological_intervals_leave_guard_on_both_sides(self):
        splits = split_intervals(10.0, 110.0, guard_s=0.8)
        train, validation, test = (splits[name] for name in ("train", "validation", "test"))
        self.assertEqual(train[0], 10.0)
        self.assertEqual(test[1], 110.0)
        self.assertAlmostEqual(validation[0] - train[1], 1.6)
        self.assertAlmostEqual(test[0] - validation[1], 1.6)
        self.assertLess(train[1], validation[0])
        self.assertLess(validation[1], test[0])

    def test_pairing_keeps_every_raw_and_camera_dependency_inside_its_split(self):
        rate, duration = 2048, 30
        times = np.arange(rate * duration) / rate
        pose_times = np.arange(0, duration, 1 / 30)
        recording = {
            "times": times,
            "raw": np.full((len(times), 8), 32768.0),
            "pose_times": pose_times,
            "pose_targets": np.column_stack((30 + np.sin(pose_times),
                                                60 + np.cos(pose_times),
                                                np.full(len(pose_times), np.nan))),
            "cues": np.array(["synthetic"] * len(pose_times)),
        }
        intervals = split_intervals(0.0, float(duration))
        parts = {name: pair_segment(recording, interval)
                 for name, interval in intervals.items()}
        for name, part in parts.items():
            lo, hi = intervals[name]
            self.assertGreater(len(part["y"]), 0)
            self.assertTrue(np.all(part["input_start"] >= lo))
            self.assertTrue(np.all(part["label_end"] < hi))
            self.assertTrue(np.all(part["input_start"] <= part["times"]))
            self.assertTrue(np.all(part["times"] <= part["label_end"]))
        self.assertLess(parts["train"]["label_end"].max(),
                        parts["validation"]["input_start"].min())
        self.assertLess(parts["validation"]["label_end"].max(),
                        parts["test"]["input_start"].min())


class BenchmarkLabelTests(unittest.TestCase):
    @staticmethod
    def row():
        p = np.zeros((33, 3))
        p[11], p[12] = [-0.2, 0, 0], [0.2, 0, 0]
        p[23], p[24] = [-0.2, -1, 0], [0.2, -1, 0]
        p[14], p[16] = [0.2, -0.5, 0], [0.2, -1, 0]
        p[18], p[20] = [0.2, -1.1, 0], [0.2, -1.1, 0]
        return {"world_landmarks": p.tolist(), "visibility": [1.0] * 33}

    def test_missing_hand_points_mask_wrist_without_masking_shoulder_or_elbow(self):
        row = self.row()
        np.testing.assert_allclose(label_values(row, "right"), [0, 0, 0], atol=1e-8)
        row["visibility"][18] = 0.2
        label = label_values(row, "right")
        np.testing.assert_allclose(label[:2], [0, 0], atol=1e-8)
        self.assertTrue(np.isnan(label[2]))
        row["visibility"][16] = 0.2
        label = label_values(row, "right")
        self.assertTrue(np.isfinite(label[0]))
        self.assertTrue(np.isnan(label[1]))
        self.assertTrue(np.isnan(label[2]))

    def test_missing_video_and_constant_targets_do_not_create_false_scores(self):
        self.assertTrue(np.isnan(label_values({"world_landmarks": None}, "right")).all())
        actual = np.array([[0, 10, np.nan], [2, 10, 5], [4, 10, np.nan]], float)
        predicted = np.array([[0, 10, 999], [2, 11, 5], [4, 9, 999]], float)
        report = score(actual, predicted)
        self.assertEqual(report["count"], [3, 3, 1])
        self.assertEqual(report["r2"][0], 1.0)
        self.assertIsNone(report["r2"][1])  # zero target variance
        self.assertIsNone(report["r2"][2])  # one observed wrist label
        self.assertEqual(report["mae_deg"][2], 0.0)


class DecoderArtifactTests(unittest.TestCase):
    def test_ridge_and_rbf_reload_predict_identically_without_pickle(self):
        rng = np.random.default_rng(123)
        features = rng.normal(size=(4, 31, 40)).astype(np.float32)
        scaler = {"mean": np.zeros(40), "scale": np.ones(40)}
        width = 2.5
        decoders = (
            Decoder("ridge", scaler, rng.normal(scale=.01, size=(280, 3)),
                    np.array([20.0, 30.0, 40.0])),
            Decoder("rbf", scaler, rng.normal(scale=.01, size=(6, 3)),
                    np.array([20.0, 30.0, 40.0]),
                    anchors=rng.normal(size=(6, 280)), width=width),
        )
        with tempfile.TemporaryDirectory() as folder:
            for decoder in decoders:
                path = Path(folder) / f"{decoder.kind}.npz"
                expected = decoder.predict(features)
                decoder.save(path)
                actual = Decoder.load(path).predict(features)
                np.testing.assert_allclose(actual, expected, atol=0, rtol=0)

    def test_tcn_reload_preserves_predictions_without_pickle(self):
        import torch

        torch.manual_seed(12)
        rng = np.random.default_rng(12)
        features = rng.normal(size=(3, 31, 40)).astype(np.float32)
        scaler = {"mean": rng.normal(size=40), "scale": np.ones(40)}
        decoder = TCNDecoder(make_tcn(), scaler,
                             np.array([45.0, 90.0, 25.0]),
                             np.array([15.0, 20.0, 10.0]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tcn.npz"
            expected = decoder.predict(features)
            decoder.save(path)
            actual = Decoder.load(path).predict(features)
            np.testing.assert_allclose(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
