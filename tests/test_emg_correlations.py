"""Synthetic correlation tests; no recording data or devices are used."""

import unittest

import numpy as np

from neural_reflex.emg_correlations import (analyze_pairing, block_shuffle,
                                            correlation_matrices)


class EmgCorrelationTests(unittest.TestCase):
    def test_cue_demeaning_exposes_common_task_effect(self):
        rng = np.random.default_rng(4)
        n = 2000
        cues = np.where(np.arange(n) < n // 2, "rest", "lift_forward")
        task = (cues == "lift_forward").astype(float)
        x = rng.normal(0, .2, (n, 8)) + task[:, None] * 4
        y = rng.normal(0, .2, (n, 3)) + task[:, None] * 4
        overall, within, counts = correlation_matrices(x, y, cues)
        self.assertGreater(overall[0, 0], .98)
        self.assertLess(abs(within[0, 0]), .08)
        self.assertEqual(counts[0, 0], n)

    def test_block_shuffle_is_reproducible_and_detects_planted_pair(self):
        rng = np.random.default_rng(9)
        n = 3000
        times = np.arange(n) * .02
        cues = np.where(times < 30, "rest", "free")
        x = rng.normal(size=(n, 8))
        y = rng.normal(size=(n, 3))
        y[:, 0] = x[:, 2] + rng.normal(0, .2, n)
        y[::3, 2] = np.nan
        result = block_shuffle(x, y, cues, times, seconds=5, permutations=99)
        again = block_shuffle(x, y, cues, times, seconds=5, permutations=99)
        self.assertEqual(result, again)
        self.assertGreater(result["observed_overall_r"][2][0], .9)
        self.assertEqual(result["familywise_p_overall"][2][0], .01)
        self.assertEqual(result["joint_pair_counts"], [3000, 3000, 2000])
        self.assertEqual(result["complete_blocks"], 12)

    def test_report_contains_only_aggregate_statistics(self):
        rng = np.random.default_rng(12)
        n = 2500
        x = rng.normal(size=(n, 1, 40))
        y = rng.normal(size=(n, 3))
        y[:, 0] = x[:, -1, 16] + rng.normal(size=n)
        pair = {"x": x, "y": y, "times": 10 + np.arange(n) * .02,
                "cues": np.array(["private subject cue"] * n)}
        report = analyze_pairing(pair, permutations=3, block_seconds=(2, 5, 10))
        self.assertEqual(report["paired_windows"], n)
        self.assertEqual(len(report["overall_r"]), 8)
        self.assertEqual(len(report["overall_r"][0]), 3)
        self.assertNotIn("private subject cue", str(report))
        self.assertNotIn("private subject cue", str(report["top_pairs"]))


if __name__ == "__main__":
    unittest.main()
