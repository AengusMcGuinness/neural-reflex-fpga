"""Invariant tests for gap-aware, paired EMG transfer statistics."""

import json
import unittest

import numpy as np

from neural_reflex.emg_transfer_stats import (bootstrap_weights, nullsafe,
                                              paired_error_intervals, paired_segments,
                                              pearson_interval, temporal_blocks)


class TransferStatisticsTests(unittest.TestCase):
    def test_temporal_blocks_preserve_wall_time_and_segment_round_boundaries(self):
        times = np.array([0.0, .1, 4.9, 5.1, 9.9, 10.0, 10.2, 10.4, 10.6])
        segments = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2])
        rounds = np.array([1, 1, 1, 1, 1, 2, 2, 1, 1])
        blocks, strata = temporal_blocks(times, segments, rounds, seconds=5)
        self.assertEqual([b.tolist() for b in blocks],
                         [[0, 1, 2], [3, 4], [5, 6], [7, 8]])
        self.assertEqual(strata, [(1, 1), (1, 1), (1, 2), (2, 1)])
        for block, (segment, round_) in zip(blocks, strata):
            self.assertTrue(np.all(segments[block] == segment))
            self.assertTrue(np.all(rounds[block] == round_))
        # The 4.8 s camera gap remains in the first block's wall-clock span.
        self.assertAlmostEqual(times[blocks[0][-1]] - times[blocks[0][0]], 4.9)

    def test_bootstrap_resamples_within_strata_and_fixes_singleton(self):
        strata = [(1, 1), (1, 1), (1, 1), (1, 2), (2, 1), (2, 1)]
        weights = bootstrap_weights(strata, replicates=40, seed=23)
        self.assertEqual(weights.shape, (40, len(strata)))
        np.testing.assert_array_equal(weights[:, :3].sum(axis=1), np.full(40, 3))
        np.testing.assert_array_equal(weights[:, 3], np.ones(40, dtype=int))
        np.testing.assert_array_equal(weights[:, 4:].sum(axis=1), np.full(40, 2))
        np.testing.assert_array_equal(weights, bootstrap_weights(strata, replicates=40, seed=23))

    def test_pearson_interval_uses_exact_block_sufficient_statistics(self):
        x = np.arange(12, dtype=float)
        y = 3 * x + 7
        y[4] = np.nan
        blocks = [np.arange(0, 4), np.arange(4, 8), np.arange(8, 12)]
        weights = np.array([[1, 1, 1], [2, 0, 1], [0, 2, 1], [0, 0, 3]])
        np.testing.assert_allclose(pearson_interval(x, y, blocks, weights), [1, 1], atol=1e-12)
        np.testing.assert_allclose(pearson_interval(x, -y, blocks, weights), [-1, -1], atol=1e-12)

    def test_paired_error_bootstrap_preserves_benefit_sign_and_pairing(self):
        actual = np.zeros((4, 1))
        predicted = np.zeros((4, 1))
        baseline = np.full((4, 1), 10.0)
        blocks = [np.array([0, 1]), np.array([2, 3])]
        weights = np.array([[2, 0], [0, 2], [1, 1]])
        row = paired_error_intervals(actual, predicted, baseline, blocks, weights)[0]
        self.assertEqual(row["mae_ci95_deg"], [0, 0])
        self.assertEqual(row["baseline_mae_ci95_deg"], [10, 10])
        self.assertEqual(row["mae_improvement_ci95_deg"], [10, 10])

        # Resampling must use the same block selection for the two errors.
        predicted[:, 0] = [5, 5, 0, 0]
        baseline[:, 0] = [10, 10, 4, 4]
        row = paired_error_intervals(actual, predicted, baseline, blocks, weights)[0]
        expected_improvements = [5.0, 4.0, 4.5]
        np.testing.assert_allclose(row["mae_improvement_ci95_deg"],
                                   np.percentile(expected_improvements, [2.5, 97.5]))

    def test_nullsafe_serializes_nested_nonfinite_statistics_as_null(self):
        value = {"matrix": np.array([[1.0, np.nan], [np.inf, -np.inf]]),
                 "nested": {"count": np.int64(3), "score": np.float64(np.nan)}}
        text = json.dumps(nullsafe(value), allow_nan=False)
        self.assertEqual(json.loads(text),
                         {"matrix": [[1.0, None], [None, None]],
                          "nested": {"count": 3, "score": None}})

    def test_short_acquisition_segment_is_skipped_and_features_stay_bounded(self):
        def segment(segment_id, start, samples):
            times = start + np.arange(samples) / 2048
            pose_times = np.arange(start, times[-1], 1 / 30)
            return {"segment_id": segment_id, "times": times,
                    "raw": np.tile(np.arange(8, dtype=float), (samples, 1)),
                    "pose_times": pose_times,
                    "pose_targets": np.tile([20.0, 60.0, 10.0], (len(pose_times), 1)),
                    "cues": np.array(["reach"] * len(pose_times)),
                    "rounds": np.array([1] * len(pose_times))}

        short = segment(1, 0.0, 1000)
        long = segment(2, 3.0, 8000)
        pair = paired_segments([short, long], history=3)
        self.assertEqual(pair["segment_coverage"][0]["paired_windows"], 0)
        self.assertGreater(pair["segment_coverage"][1]["paired_windows"], 0)
        self.assertTrue(np.all(pair["segments"] == 2))
        self.assertTrue(np.all(pair["input_start"] >= long["times"][0]))
        self.assertTrue(np.all(pair["label_end"] < long["pose_times"][-1]))
        self.assertTrue(np.all(pair["times"] < pair["label_end"]))


if __name__ == "__main__":
    unittest.main()
