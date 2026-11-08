"""Guided (follow-the-target) recording: target timeline, arm IK, labels and the robot check."""

import json
import math
import unittest

import numpy as np

from neural_reflex import guided
from neural_reflex.arm_pose import PROTOCOLS, TARGET_KEYS
from neural_reflex.dashboard import load_home
from neural_reflex.hand_map import MAP_FILE, robot_targets, solve
from neural_reflex.pick_sim import joint_limits
from neural_reflex.robot_map import DT, load_map

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

    def test_target_maps_into_the_robot_within_its_speed(self):
        cfg = json.loads(MAP_FILE.read_text())
        human = {"table_up": guided.TABLE_UP + 0.02, "forward_range": [min(guided.TARGETS), max(guided.TARGETS)]}
        tr = guided.trajectory(DT)
        x, z = robot_targets({"forward": tr["forward"], "up": tr["up"], "arm_m": 0.55}, human, cfg["robot"])
        lo, hi = cfg["robot"]["x_limits_m"]
        self.assertTrue(((x >= lo) & (x <= hi)).all())
        angles, _, moved = solve(x, z, joint_limits(load_map(), load_home()), cfg["max_pitch_change_deg"])
        self.assertFalse(moved.any())
        speed = np.abs(np.gradient(angles, DT, axis=0))
        self.assertLess(np.percentile(speed, 95), 32)


if __name__ == "__main__":
    unittest.main()
