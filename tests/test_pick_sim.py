"""Planar pick simulation: kinematics, load, reach and a full planned pick."""

import unittest

import numpy as np

from neural_reflex.dashboard import load_home
from neural_reflex.pick_sim import (FORWARD_TICK_SIGN, HAND, SAFE_KG_CM, SHOULDER_HEIGHT, STALL_KG_CM, UPPER_ARM,
                                    FOREARM, check_pose, forward_kinematics, gravity_torque, inverse_kinematics,
                                    joint_limits, load_map, plan_pick, reach_map, to_ticks)


class PickSimTests(unittest.TestCase):
    def setUp(self):
        self.config, self.home = load_map(), load_home()
        self.limits = joint_limits(self.config, self.home)

    def test_home_points_straight_up_with_no_load(self):
        tip = forward_kinematics([0, 0, 0])[-1]
        self.assertAlmostEqual(tip[1], SHOULDER_HEIGHT + UPPER_ARM + FOREARM + HAND)
        self.assertEqual(gravity_torque([0, 0, 0])["shoulder_lift"], 0)

    def test_level_arm_is_near_stall(self):
        self.assertGreater(gravity_torque([90, 0, 0])["shoulder_lift"], 0.8 * STALL_KG_CM)

    def test_inverse_kinematics_round_trip(self):
        for x, z, pitch in ((0.2, 0.03, 180), (0.25, 0.08, 150), (0.15, 0.12, 170)):
            angles = inverse_kinematics(x, z, pitch)
            self.assertIsNotNone(angles)
            tip = forward_kinematics(angles)[-1]
            self.assertAlmostEqual(tip[0], x, places=6)
            self.assertAlmostEqual(tip[1], z, places=6)
            self.assertAlmostEqual(angles.sum(), pitch, places=6)
        self.assertIsNone(inverse_kinematics(0.6, 0.0, 180))

    def test_pose_checks(self):
        self.assertEqual(check_pose([0, 0, 0], self.limits), [])
        self.assertTrue(any("table" in p for p in check_pose([180, 0, 0], self.limits)))

    def test_there_is_a_safe_pick_zone(self):
        rows = reach_map(self.limits, 0.05, 0.08)
        safe = [r for r in rows if r["status"] == "safe"]
        self.assertTrue(safe)
        self.assertTrue(all(r["shoulder_kg_cm"] <= SAFE_KG_CM for r in safe))
        self.assertEqual(rows[0]["status"], "unreachable")      # too close to the base

    def test_planned_pick_reaches_item_and_stays_legal(self):
        plan = plan_pick(0.21, 0.05, 0.08, self.limits, 24.0)
        self.assertEqual(plan["problems"], [])
        tips = np.array([forward_kinematics(a)[-1] for a in plan["angles"]])
        grasp = plan["labels"].index("close the gripper")
        self.assertAlmostEqual(tips[grasp][0], 0.21, places=4)
        self.assertAlmostEqual(tips[grasp][1], 0.03, places=4)
        self.assertGreater(plan["grip"][plan["labels"].index("lift the item")], 0.99)
        self.assertLess(plan["torque"][:, 0].max(), 6.0)
        ticks = to_ticks(plan, self.home)
        for joint, values in ticks.items():
            lo, hi = self.config["limits"][joint]
            self.assertTrue(lo <= values.min() and values.max() <= hi, joint)
        self.assertTrue((ticks["shoulder_pan"] == self.home["shoulder_pan"]).all())   # one plane

    def test_forward_tick_signs_cover_planar_joints(self):
        self.assertEqual(set(FORWARD_TICK_SIGN), {"shoulder_lift", "elbow_flex", "wrist_flex"})


if __name__ == "__main__":
    unittest.main()
