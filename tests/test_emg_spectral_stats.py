"""Synthetic tests for spectral association and whole-block uncertainty."""

import unittest

import numpy as np

from neural_reflex.emg_spectral_stats import (
    _bootstrap_pair_coefficients, _pearson, _within_cue,
    correlation_summary, selected_intervals,
)


def pair_with(features, y, cues):
    n = len(features)
    return {"features": features, "y": y, "cues": np.asarray(cues),
            "times": np.arange(n, dtype=float) * .5,
            "segments": np.zeros(n, int), "rounds": np.ones(n, int)}


class SpectralCorrelationTests(unittest.TestCase):
    def test_shape_pairwise_masks_constant_and_spearman(self):
        n = 40
        t = np.arange(n, dtype=float)
        features = np.zeros((n, 8, 12))
        features[:, 0, 0] = t * t
        features[:, 0, 1] = 0.1
        features[:, 1, 0] = -t
        features[::4, 1, 0] = np.nan
        y = np.stack((t, t * 3, np.full(n, 8.0)), axis=1)
        y[::5, 0] = np.nan
        report = correlation_summary(pair_with(features, y, ["move"] * n))
        self.assertEqual(report["windows"], n)
        self.assertEqual(report["pearson_r"].shape, (8, 12, 3))
        self.assertEqual(report["pair_counts"][1, 0, 0], 24)
        self.assertAlmostEqual(report["pearson_r"][1, 0, 1], -1)
        self.assertAlmostEqual(report["spearman_rho"][0, 0, 1], 1)
        self.assertLess(report["pearson_r"][0, 0, 1], 1)
        self.assertTrue(np.isnan(report["pearson_r"][0, 1, 1]))
        self.assertTrue(np.isnan(report["spearman_rho"][0, 1, 1]))
        self.assertTrue(np.isnan(report["cue_demeaned_r"][0, 0, 2]))
        self.assertEqual(report["target_std_deg"][2], 0)

    def test_cue_demeaning_uses_each_pairs_valid_subset(self):
        # One missing feature row changes the cue mean. Residualizing first,
        # then masking, would produce a different within-cue coefficient.
        features = np.zeros((8, 8, 12))
        x = np.array([0, 1, 2, 7, 20, 21, 22, 29.], float)
        y = np.array([3, 1, 4, 1, 7, 5, 8, 5.], float)
        cues = np.array(["a"] * 4 + ["b"] * 4)
        features[:, 0, 0] = x
        features[1, 0, 0] = np.nan
        target = np.column_stack((y, y, y))
        report = correlation_summary(pair_with(features, target, cues))
        valid = np.isfinite(features[:, 0, 0])
        expected = _within_cue(x[valid], y[valid], cues[valid])
        self.assertEqual(report["pair_counts"][0, 0, 0], 7)
        self.assertAlmostEqual(report["cue_demeaned_r"][0, 0, 0], expected)

    def test_cue_means_can_explain_overall_association(self):
        rng = np.random.default_rng(3)
        n = 400
        cues = np.where(np.arange(n) < n // 2, "rest", "reach")
        step = (cues == "reach").astype(float)
        x = 10 * step + rng.normal(size=n)
        y = 10 * step + rng.normal(size=n)
        features = np.zeros((n, 8, 12))
        features[:, 0, 0] = x
        target = np.tile(y[:, None], (1, 3))
        report = correlation_summary(pair_with(features, target, cues))
        self.assertGreater(report["pearson_r"][0, 0, 0], .95)
        self.assertLess(abs(report["cue_demeaned_r"][0, 0, 0]), .15)

    def test_weighted_within_cue_bootstrap_recomputes_cue_means(self):
        x = np.array([0, 2, 5, 7, 10, 10, 13, 18, 22, 20, 24, 28.], float)
        y = np.array([4, 1, 5, 3, 8, 2, 7, 6, 2, 9, 4, 10.], float)
        cues = np.array(["a", "b", "a", "b", "a", "b", "a", "b",
                         "a", "b", "a", "b"])
        blocks = [np.arange(i, i + 3) for i in range(0, 12, 3)]
        weights = np.array([[1, 1, 1, 1], [2, 0, 1, 1], [0, 2, 0, 2],
                            [1, 0, 3, 0]], dtype=int)
        overall, within = _bootstrap_pair_coefficients(x, y, cues, blocks, weights)
        for draw, row in enumerate(weights):
            indices = np.concatenate([np.tile(block, count) for block, count in zip(blocks, row)])
            self.assertAlmostEqual(overall[draw], _pearson(x[indices], y[indices]))
            self.assertAlmostEqual(within[draw], _within_cue(x[indices], y[indices], cues[indices]))
        # Fixed original residuals are not equivalent to weighted re-centering.
        original_x = x.copy()
        original_y = y.copy()
        for cue in np.unique(cues):
            at = cues == cue
            original_x[at] -= x[at].mean()
            original_y[at] -= y[at].mean()
        resampled = np.concatenate([np.tile(block, count) for block, count in zip(blocks, weights[1])])
        self.assertGreater(abs(within[1] - _pearson(original_x[resampled], original_y[resampled])), .005)

        shifted_x, shifted_y = 1e9 + x, 1e8 + y
        shifted_overall, shifted_within = _bootstrap_pair_coefficients(
            shifted_x, shifted_y, cues, blocks, weights)
        np.testing.assert_allclose(shifted_overall, overall, atol=1e-7)
        np.testing.assert_allclose(shifted_within, within, atol=1e-7)

    def test_selected_intervals_contract_and_validation(self):
        n = 32
        features = np.zeros((n, 8, 12))
        target = np.zeros((n, 3))
        target[:, 0] = np.arange(n)
        features[:, 2, 5] = target[:, 0] * 2
        pair = pair_with(features, target, ["a"] * 16 + ["b"] * 16)
        pair["segments"][20:] = 1
        pair["rounds"][12:] = 2
        selection = {"label": "chosen", "channel": 2, "feature_index": 5, "target_index": 0}
        report = selected_intervals(pair, [selection], replicates=50, seconds=(2, 5))
        self.assertEqual(set(report), {"2", "5"})
        self.assertEqual(report["2"]["pairs"][0]["label"], "chosen")
        self.assertAlmostEqual(report["2"]["pairs"][0]["pearson_r"], 1)
        self.assertAlmostEqual(report["2"]["pairs"][0]["cue_demeaned_r"], 1)
        self.assertEqual(report["2"]["pairs"][0]["pearson_ci95"], [1, 1])
        self.assertGreaterEqual(report["5"]["blocks"]["strata"], 2)
        with self.assertRaises(ValueError):
            selected_intervals(pair, [{**selection, "feature_index": 12}], replicates=2)


if __name__ == "__main__":
    unittest.main()
