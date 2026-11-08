"""Band check scoring with synthetic EMG batches."""

import unittest

import numpy as np

from neural_reflex.fist_check import PHASES, evaluate


def synth(fist_gain=10.0, reconnect=False, rail_channel=None, seed=0):
    rng = np.random.default_rng(seed)
    phases, records, t = [], [], 100.0
    for kind, _, seconds in PHASES:
        phases.append((kind, t, t + seconds))
        for b in range(int(seconds * 128)):
            scale = np.full(8, 30.0)
            if kind == "fist":
                scale[0] *= fist_gain                 # ch1 over the hand-closing muscles
                scale[2] *= fist_gain / 2
            values = 32768 + rng.standard_normal((16, 8)) * scale
            if rail_channel is not None:
                values[:, rail_channel] = 0
            records.append({"host_time_s": t + b / 128, "values": values.round().astype(int).tolist(),
                            "band_connection_id": "b" if reconnect and len(records) > 300 else "a"})
        t += seconds
    return records, phases


class FistCheckTests(unittest.TestCase):
    def test_good_band_passes(self):
        result = evaluate(*synth(10.0))
        self.assertTrue(result["ok"])
        self.assertEqual(result["verdict"], "good")
        self.assertEqual(result["best_channels"][0], 1)
        self.assertGreater(result["best_ratio"], 5)

    def test_no_fist_response_fails(self):
        result = evaluate(*synth(1.0))
        self.assertFalse(result["ok"])
        self.assertTrue(any("no clear fist response" in p for p in result["problems"]))

    def test_reconnect_and_saturation_fail(self):
        result = evaluate(*synth(10.0, reconnect=True))
        self.assertFalse(result["ok"])
        self.assertTrue(any("reconnected" in p for p in result["problems"]))
        result = evaluate(*synth(10.0, rail_channel=5))
        self.assertFalse(result["ok"])
        self.assertTrue(any("saturated" in p for p in result["problems"]))

    def test_no_data(self):
        self.assertFalse(evaluate([], [])["ok"])


if __name__ == "__main__":
    unittest.main()
