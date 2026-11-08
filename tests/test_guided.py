"""Guided (follow-the-target) recording: target timeline, arm IK, labels and the robot check."""

import math
import unittest

import numpy as np

from neural_reflex import guided
from neural_reflex.arm_pose import PROTOCOLS, TARGET_KEYS

DT = 0.02
UP, RIGHT, FORWARD = np.array([0, -1.0, 0]), np.array([-1.0, 0, 0]), np.array([0, 0, -1.0])


class GuidedTests(unittest.TestCase):
    def test_arm_ik_known_poses(self):
        self.assertAlmostEqual(guided.arm_ik(0.0, -1.0)[0], 0.0, delta=0.5)      # hanging
        self.assertAlmostEqual(guided.arm_ik(1.0, 0.0)[0], 90.0, delta=0.5)      # level forward
        lift, elbow = guided.arm_ik(0.6, -0.5)
        a, b = math.radians(lift), math.radians(lift + elbow)
        hand = (guided.UPPER * math.sin(a) + guided.LOWER * math.sin(b),
                -guided.UPPER * math.cos(a) - guided.LOWER * math.cos(b))
        self.assertAlmostEqual(hand[0], 0.6, places=6)
        self.assertAlmostEqual(hand[1], -0.5, places=6)

    def test_timeline_visits_every_spot_each_round(self):
        grips = [(s[3], s[4]) for s in guided.TIMELINE if s[0] == "grip"]
        self.assertEqual(len(grips), guided.ROUNDS * len(guided.TARGETS))
        self.assertEqual(guided.TIMELINE[0][0], "ready")
        self.assertIsNone(guided.target_at(guided.TOTAL_S + 1))

    def test_target_is_continuous_and_grip_follows_steps(self):
        tr = guided.trajectory(DT)
        self.assertLess(np.abs(np.diff(tr["forward"])).max(), 0.02)
        self.assertLess(np.abs(np.diff(tr["up"])).max(), 0.02)
        lift = [g for g, s in zip(tr["grip"], tr["step"]) if s == "lift"]
        self.assertTrue(np.allclose(lift, 1.0))
        reach = [g for g, s in zip(tr["grip"], tr["step"]) if s == "reach"]
        self.assertTrue(np.allclose(reach, 0.0))

    def test_follow_protocol_is_available_and_labels_are_complete(self):
        self.assertIn("follow", PROTOCOLS)
        target = guided.target_at(30.0)
        self.assertTrue(set(TARGET_KEYS) <= set(target))

    def test_measured_angles_without_hips(self):
        p = np.zeros((33, 3))
        p[11], p[12] = -0.18 * RIGHT, 0.18 * RIGHT
        p[0] = 0.25 * UP + 0.08 * FORWARD
        p[14] = p[12] - 0.3 * UP                      # upper arm hanging
        p[18] = p[20] = p[14] + 0.3 * FORWARD         # forearm level forward
        p[16] = p[14] + 0.25 * FORWARD
        vis = [0.95] * 33
        vis[23] = vis[24] = 0.05                     # hips hidden by the table
        m = guided.measured_angles(p.tolist(), vis, "right")
        self.assertAlmostEqual(m["shoulder_lift"], 0, delta=1)
        self.assertAlmostEqual(m["elbow_flex"], 90, delta=1)
        self.assertIsNone(guided.measured_angles(None, None, "right"))


if __name__ == "__main__":
    unittest.main()
