"""Arm-fit helpers: held-out split, band-arm detection, overwrite protection."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from neural_reflex.armfit import band_arm, chunk_split, quaternion_matrix
from neural_reflex.cli import refuse_overwrite


class ArmfitTests(unittest.TestCase):
    def test_chunk_split_keeps_test_apart_from_training(self):
        times = np.arange(0, 60, 0.02)
        train, test, inner = chunk_split(times)
        self.assertFalse((train & test).any())
        # 2 of 12 five-second chunks are test, each trimmed by the guard at both ends.
        self.assertAlmostEqual(test.mean(), 2 * (5 - 2 * 0.6) / 60, delta=0.01)
        # No training window within the guard of any test window.
        gap = np.min(np.abs(times[train][:, None] - times[test][None, :]))
        self.assertGreaterEqual(gap, 0.5)
        self.assertFalse((inner & ~train).any())

    def test_band_arm_follows_the_gyro_not_the_label(self):
        t = np.arange(0, 30, 1 / 30)
        swing = np.radians(40) * np.sin(2 * np.pi * 0.4 * t)
        moving = np.stack([np.sin(swing), -np.cos(swing), np.zeros_like(t)], axis=1)
        still = np.tile([0.0, -1.0, 0.0], (len(t), 1)) + 0.002 * np.random.default_rng(1).standard_normal((len(t), 3))
        still /= np.linalg.norm(still, axis=1, keepdims=True)
        tracks = {"left": (t, np.zeros((len(t), 3)), moving), "right": (t, np.zeros((len(t), 3)), still)}
        gyro_t = np.arange(0, 30, 1 / 128)
        rate = np.degrees(np.gradient(np.radians(40) * np.sin(2 * np.pi * 0.4 * gyro_t), gyro_t))
        gyro_v = np.stack([rate, np.zeros_like(rate), np.zeros_like(rate)], axis=1)
        side, scores = band_arm(tracks, gyro_t, gyro_v)
        self.assertEqual(side, "left")             # the moving arm, even though the band is "right"
        self.assertGreater(scores["left"], 0.5)

    def test_identity_quaternion(self):
        self.assertTrue(np.allclose(quaternion_matrix(np.array([1.0, 0, 0, 0])), np.eye(3).ravel()))

    def test_recorders_refuse_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "arm.jsonl"
            existing.write_text("{}\n")
            with self.assertRaises(ValueError):
                refuse_overwrite(str(existing))
            refuse_overwrite(str(Path(tmp) / "new.jsonl"), None)


if __name__ == "__main__":
    unittest.main()
