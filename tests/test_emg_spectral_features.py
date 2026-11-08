"""Spectral normalization, gain invariance, and causal segment boundaries."""

import unittest

import numpy as np

from neural_reflex.emg_spectral_features import (
    BANDS, FEATURE_NAMES, FS, features_from_windows, paired_spectral_segments,
    periodograms,
)


def segment(start, duration, segment_id, pose_times, pose_targets):
    samples = int(duration * FS)
    times = start + np.arange(samples) / FS
    phase = np.arange(samples) / FS
    raw = np.full((samples, 8), 32768.0)
    raw += 250 * np.sin(2 * np.pi * 100 * phase)[:, None]
    return {"times": times, "raw": raw, "pose_times": pose_times,
            "pose_targets": pose_targets,
            "cues": np.full(len(pose_times), "reach"),
            "rounds": np.full(len(pose_times), 1), "segment_id": segment_id}


class PeriodogramTests(unittest.TestCase):
    def test_sine_peak_and_nonoverlapping_band_power(self):
        t = np.arange(512) / FS
        windows = np.zeros((1, 512, 8))
        windows[0, :, 0] = 2 * np.sin(2 * np.pi * 100 * t)
        frequencies, psd = periodograms(windows)
        self.assertEqual(frequencies[np.argmax(psd[0, :, 0])], 100)
        self.assertEqual(BANDS, ((20, 80), (80, 150), (150, 250), (250, 450)))
        self.assertEqual(len(FEATURE_NAMES), 12)
        features = features_from_windows(windows)
        self.assertEqual(features.shape, (1, 8, 12))
        self.assertGreater(features[0, 0, 6], .99)  # 80–150 Hz relative power
        np.testing.assert_allclose(features[0, 0, 5:9].sum(), 1, atol=1e-12)
        self.assertAlmostEqual(features[0, 0, 9], 100, places=7)
        self.assertEqual(features[0, 0, 10], 100)
        self.assertTrue(np.isnan(features[0, 1, 5:]).all())
        np.testing.assert_array_equal(features[0, 1, :5], 0)

    def test_one_sided_psd_integrates_to_hann_weighted_demeaned_power(self):
        rng = np.random.default_rng(7)
        windows = rng.normal(size=(2, 512, 8))
        frequencies, psd = periodograms(windows)
        periodic_hann = np.hanning(513)[:-1]
        demeaned = windows - windows.mean(axis=1, keepdims=True)
        weighted_power = np.sum((demeaned * periodic_hann[None, :, None])**2, axis=1) / np.sum(periodic_hann**2)
        integrated = psd.sum(axis=1) * (frequencies[1] - frequencies[0])
        np.testing.assert_allclose(integrated, weighted_power, rtol=1e-12, atol=1e-12)

    def test_gain_changes_absolute_features_but_not_relative_spectral_shape(self):
        rng = np.random.default_rng(12)
        windows = rng.normal(size=(3, 512, 8))
        original = features_from_windows(windows)
        amplified = features_from_windows(3 * windows)
        self.assertTrue(np.all(amplified[:, :, :5] > original[:, :, :5]))
        np.testing.assert_allclose(amplified[:, :, 5:], original[:, :, 5:], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(original[:, :, 5:9].sum(axis=2), 1, atol=1e-12)


class SpectralPairingTests(unittest.TestCase):
    def test_future_raw_perturbation_does_not_change_earlier_features(self):
        pose_times = np.arange(0, 6.1, 1 / 30)
        targets = np.tile([30.0, 60.0, np.nan], (len(pose_times), 1))
        original = segment(0, 6, 0, pose_times, targets)
        changed = {**original, "raw": original["raw"].copy()}
        changed["raw"][10000:, 0] += 500
        first = paired_spectral_segments([original])
        second = paired_spectral_segments([changed])
        np.testing.assert_array_equal(first["times"], second["times"])
        past = first["times"] < 10000 / FS
        self.assertGreater(past.sum(), 10)
        np.testing.assert_allclose(first["features"][past], second["features"][past], atol=0, rtol=0,
                                   equal_nan=True)
        self.assertTrue(np.any(np.abs(first["features"][~past] - second["features"][~past]) > 1e-6))
        self.assertTrue(np.isnan(first["y"][:, 2]).all())

    def test_acquisition_and_camera_gaps_never_bridge_input_or_label(self):
        camera = np.arange(0, 9.2, 1 / 30)
        camera = camera[(camera < 2) | (camera >= 2.5)]
        targets = np.tile([40.0, 80.0, np.nan], (len(camera), 1))
        pieces = [segment(0, 4, 0, camera, targets),
                  segment(5, 4, 1, camera, targets)]
        paired = paired_spectral_segments(pieces)
        self.assertEqual(paired["features"].shape[1:], (8, 12))
        self.assertEqual(len(paired["segment_coverage"]), 2)
        self.assertGreater(paired["segment_coverage"][0]["label_excluded_windows"], 0)
        for part, piece in enumerate(pieces):
            here = paired["segments"] == part
            self.assertGreater(here.sum(), 0)
            self.assertTrue(np.all(paired["input_start"][here] >= piece["times"][0]))
            self.assertTrue(np.all(paired["times"][here] <= piece["times"][-1]))
            self.assertTrue(np.all(paired["label_end"][here] <= piece["pose_times"][-1]))
        self.assertFalse(np.any((paired["times"] + .05 >= 2) & (paired["times"] + .05 < 2.5)))
        self.assertTrue(np.all(paired["input_start"] <= paired["times"]))
        self.assertTrue(np.all(paired["times"] <= paired["label_end"]))


if __name__ == "__main__":
    unittest.main()
