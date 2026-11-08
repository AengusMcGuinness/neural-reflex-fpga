"""Arm-fit helpers: held-out split, band-arm detection, overwrite protection."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from neural_reflex.armfit import TARGETS, band_arm, chunk_split, fit_rounds, quaternion_matrix, robot_moves
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

    def test_robot_moves_keeps_reachable_round_windows(self):
        targets = np.array([[170.0, 10.0], [95.0, 80.0], [20.0, 30.0], [150.0, 0.0]])
        block = {"targets": targets, "cues": ["home", "bend", "lower", "sync"], "rounds": [1, 1, 1, None]}
        self.assertEqual(robot_moves(block).tolist(), [True, True, False, False])   # too low; not a round
        full = {"targets": targets, "cues": ["home", "elbow", "lift_forward", "free"], "rounds": [None] * 4}
        self.assertEqual(robot_moves(full).tolist(), [True, True, False, False])

    def test_leave_one_round_out_never_sees_its_round(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((900, 6))
        y = np.column_stack([x @ rng.standard_normal(6), x @ rng.standard_normal(6)]) + 0.05 * rng.standard_normal((900, 2))
        groups = np.repeat(np.array(["b#1", "b#2", "b#3"], dtype=object), 300)
        result = fit_rounds(x, y, groups)
        self.assertFalse(np.isnan(result["predicted"]).any())
        for name in TARGETS:
            self.assertGreater(result["scores"][name]["r2"], 0.95)

    def test_robot_protocol_has_slow_repeated_rounds(self):
        from neural_reflex.arm_pose import PROTOCOL, ROBOT_ROUNDS, protocol_step
        rounds = {step[3] for step in PROTOCOL if len(step) > 3}
        self.assertEqual(rounds, set(range(1, ROBOT_ROUNDS + 1)))
        self.assertLess(sum(step[2] for step in PROTOCOL), 120)
        # Lowering 90 degrees must not outrun the robot's ~32 deg/s cap.
        lower = next(step for step in PROTOCOL if step[0] == "lower")
        self.assertLessEqual(90 / lower[2], 32)
        self.assertEqual(protocol_step(20.0)[5], 1)

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
