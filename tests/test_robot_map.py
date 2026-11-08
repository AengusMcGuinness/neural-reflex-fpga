"""Human-to-robot translation: limits, scaling, smoothing and the speed-limited follower."""

import unittest

import numpy as np

from neural_reflex.leader_arm import JOINTS
from neural_reflex.robot_map import DT, follow, load_map, lowpass, stretch, translate

HOME = {"shoulder_pan": 2042, "shoulder_lift": 2003, "elbow_flex": 1074, "wrist_flex": 1945,
        "wrist_roll": 1928, "gripper": 1776}


def signals(n=200, **values):
    base = {"shoulder_pan": 0.0, "shoulder_lift": 180.0, "elbow_flex": 0.0, "wrist_flex": 0.0,
            "wrist_roll": 0.0, "gripper": 0.0}
    base.update(values)
    out = {j: np.full(n, float(base[j])) for j in JOINTS}
    out["camera_gap"] = np.zeros(n, dtype=bool)
    return out


class RobotMapTests(unittest.TestCase):
    def setUp(self):
        self.config = load_map()

    def test_config_matches_robot(self):
        self.assertEqual(set(self.config["joints"]), set(JOINTS))
        for joint, (lo, hi) in self.config["limits"].items():
            self.assertLess(lo, HOME[joint])
            self.assertLess(HOME[joint], hi)

    def test_arm_straight_up_maps_to_home(self):
        targets, clamped = translate(signals(), self.config, HOME)
        for joint in JOINTS:
            self.assertAlmostEqual(targets[joint][-1], HOME[joint], delta=1, msg=joint)
            self.assertFalse(clamped[joint].any())

    def test_robot_protocol_poses_fit_and_hanging_arm_clamps(self):
        # Shoulder height forward (90) with the elbow at 90: the robot protocol's lowest pose.
        targets, clamped = translate(signals(shoulder_lift=90.0, elbow_flex=90.0), self.config, HOME)
        self.assertFalse(clamped["shoulder_lift"].any() or clamped["elbow_flex"].any())
        self.assertAlmostEqual(targets["shoulder_lift"][-1] - HOME["shoulder_lift"], 90 * 4096 / 360, delta=2)
        # A hanging arm would point the robot through its table: it stops at the forward limit.
        targets, clamped = translate(signals(shoulder_lift=0.0), self.config, HOME)
        self.assertTrue(clamped["shoulder_lift"].all())
        self.assertEqual(targets["shoulder_lift"][-1], self.config["limits"]["shoulder_lift"][1] - self.config["margin_ticks"])

    def test_disabled_joints_hold_home(self):
        targets, _ = translate(signals(shoulder_pan=60.0, wrist_flex=40.0, wrist_roll=50.0), self.config, HOME)
        for joint in ("shoulder_pan", "wrist_flex", "wrist_roll"):
            self.assertFalse(self.config["joints"][joint]["enabled"])
            self.assertTrue((targets[joint] == HOME[joint]).all(), joint)

    def test_targets_never_leave_limits(self):
        targets, clamped = translate(signals(shoulder_pan=170.0, elbow_flex=400.0, wrist_roll=-500.0),
                                     self.config, HOME)
        margin = self.config["margin_ticks"]
        for joint in JOINTS:
            lo, hi = self.config["limits"][joint]
            self.assertTrue((targets[joint] >= lo + margin).all() and (targets[joint] <= hi - margin).all())
        self.assertTrue(clamped["elbow_flex"].any())

    def test_camera_gap_holds_still(self):
        s = signals(n=100)
        s["elbow_flex"][50:] = 90.0
        s["camera_gap"][50:70] = True
        targets, _ = translate(s, {**self.config, "joints": {j: {**v, "smooth_s": 0}
                                                             for j, v in self.config["joints"].items()}}, HOME)
        self.assertEqual(targets["elbow_flex"][60], targets["elbow_flex"][49])
        self.assertGreater(targets["elbow_flex"][80], HOME["elbow_flex"] + 500)

    def test_follower_respects_speed_cap(self):
        step = np.r_[np.zeros(10), np.full(400, 1000.0)]
        out = follow(step, 0.0, cap=360, accel=720)
        speed = np.abs(np.diff(out)) / DT
        self.assertLessEqual(speed.max(), 360 + 1e-6)
        self.assertAlmostEqual(out[-1], 1000.0)

    def test_lowpass_and_stretch(self):
        noisy = np.r_[np.zeros(50), np.ones(50)]
        self.assertLess(lowpass(noisy, 0.2)[51], 0.2)
        self.assertEqual(len(stretch(np.arange(11.0), 2.0)), 21)


if __name__ == "__main__":
    unittest.main()
