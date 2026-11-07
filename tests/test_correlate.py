"""Correlation helpers: lag finding, safe Pearson r, and block discovery."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from neural_reflex.correlate import best_lag, blocks_from, pearson, summarize


class CorrelateTests(unittest.TestCase):
    def test_best_lag_finds_a_known_delay(self):
        rng = np.random.default_rng(0)
        reference = np.convolve(rng.standard_normal(2000), np.ones(10) / 10, mode="same")
        delayed = np.roll(reference, 4)              # trails by 4 steps (80 ms)
        lag, at_lag, at_zero = best_lag(reference, delayed, 20)
        self.assertEqual(lag, 4)
        self.assertGreater(at_lag, at_zero)

    def test_pearson_handles_flat_signals(self):
        self.assertIsNone(pearson(np.ones(50), np.arange(50)))
        self.assertAlmostEqual(pearson(np.arange(50), 2 * np.arange(50) + 1), 1.0)

    def test_blocks_from_session_or_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("arm-2", "arm-1"):
                block = Path(tmp) / "blocks" / name
                block.mkdir(parents=True)
                (block / "arm.jsonl").write_text("")
            (Path(tmp) / "blocks" / "hand-only").mkdir()
            self.assertEqual([b.name for b in blocks_from([tmp])], ["arm-1", "arm-2"])
            self.assertEqual(len(blocks_from([str(Path(tmp) / "blocks" / "arm-1")])), 1)

    def test_summary_averages_blocks_and_flags_grip(self):
        def report(r):
            return {"emg": {s: [r] + [0.0] * 7 for s in ("shoulder_lift", "elbow_flex", "wrist_flex",
                                                           "shoulder_pan", "wrist_roll", "gripper")},
                    "motion": {s: {"best_abs_r": 0.5, "axis": 0} for s in ("shoulder_lift", "elbow_flex",
                               "wrist_flex", "shoulder_pan", "wrist_roll", "gripper")}}
        summary = summarize([report(0.2), report(-0.4)])
        self.assertEqual(summary["elbow_flex"]["best_emg_channel"], 1)
        self.assertAlmostEqual(summary["elbow_flex"]["best_emg_abs_r"], 0.3)
        self.assertTrue(summary["gripper"]["circular"])


if __name__ == "__main__":
    unittest.main()
