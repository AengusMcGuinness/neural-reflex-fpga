"""Guided fit: follow delay, labels, cross-round fitting, reconnect handling and R² units."""

import unittest

import numpy as np

from neural_reflex import guided
from neural_reflex.guidedfit import fit_rounds, follow_delay, label_windows
from neural_reflex.handfit import longest_connection, r2


def synthetic_track(delay=0.6, dt=1 / 30):
    t = np.arange(0, guided.TOTAL_S, dt)
    shown = [guided.target_at(x) for x in t]
    did = [guided.target_at(max(0.0, x - delay)) for x in t]           # you move `delay` later
    return {"time": t + 1000.0,
            "target": {k: np.array([s[k] for s in shown]) for k in ("forward", "up", "grip", "shoulder_lift", "elbow_flex")},
            "followed": {k: np.array([d[k] for d in did]) for k in ("forward", "up", "shoulder_lift", "elbow_flex")},
            "step": [s["step"] for s in shown], "round": [s["round"] for s in shown], "spot": [s["spot"] for s in shown]}


class GuidedFitTests(unittest.TestCase):
    def test_follow_delay_is_recovered(self):
        delay, err_at, err_zero = follow_delay(synthetic_track(0.6))
        self.assertAlmostEqual(delay, 0.6, delta=0.03)
        self.assertLess(err_at, err_zero)

    def test_labels_skip_setup_and_shift_by_delay(self):
        track = synthetic_track(0.0)
        times = track["time"][::3]
        y, steps, rounds, spots, keep = label_windows(track, times, 0.5)
        self.assertFalse(any(k and s in ("ready", "sync", "rest") for k, s in zip(keep, steps)))
        i = len(times) // 2
        expected = guided.target_at(times[i] - 1000.0 - 0.5)["forward"]
        self.assertAlmostEqual(y[i, 0], expected, places=3)

    def test_cross_round_fit_learns_a_real_relationship(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((800, 10))
        y = np.column_stack([x[:, 0] * 0.1, rng.standard_normal(800) * 0.1])     # 2nd output is noise
        groups = [1] * 400 + [2] * 400
        pred, tested, regs = fit_rounds(x, y, groups)
        s = r2(y[tested], pred[tested], 1e-4)
        self.assertGreater(s[0], 0.9)
        self.assertLess(s[1], 0.05)              # no signal -> no fake skill
        self.assertEqual(len(regs), 2)

    def test_longest_connection_and_unit_aware_r2(self):
        recs = [{"band_connection_id": "a"}] * 3 + [{"band_connection_id": "b"}] * 7
        self.assertEqual(len(longest_connection(recs)), 7)
        small = np.linspace(0, 0.1, 50)[:, None]
        self.assertTrue(np.isnan(r2(small, small)[0]))                 # degrees threshold: "didn't move"
        self.assertAlmostEqual(r2(small, small, 1e-6)[0], 1.0)          # arm-length units: real score


if __name__ == "__main__":
    unittest.main()
