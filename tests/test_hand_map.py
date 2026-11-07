"""Hand position -> robot gripper mapping, with synthetic landmarks."""

import json
import unittest

import numpy as np

from neural_reflex.dashboard import load_home
from neural_reflex.hand_map import MAP_FILE, body_frame, calibrate, hand_track, robot_targets, solve
from neural_reflex.pick_sim import forward_kinematics, joint_limits
from neural_reflex.robot_map import load_map

# MediaPipe-like world frame: x camera-right, y down, z away. Person faces the camera.
UP, RIGHT, FORWARD = np.array([0, -1.0, 0]), np.array([-1.0, 0, 0]), np.array([0, 0, -1.0])


def frame(hand_forward, hand_up, hips_visible=True, cue=None, t=0.0):
    p = np.zeros((33, 3))
    p[11], p[12] = -0.18 * RIGHT, 0.18 * RIGHT
    p[23], p[24] = -0.1 * RIGHT - 0.5 * UP, 0.1 * RIGHT - 0.5 * UP
    p[0] = 0.25 * UP + 0.08 * FORWARD
    shoulder = p[12]
    hand = shoulder + hand_forward * FORWARD + hand_up * UP
    p[14] = shoulder + 0.5 * (hand - shoulder) + 0.02 * UP      # elbow roughly halfway
    p[16] = hand - 0.05 * (hand - shoulder) / np.linalg.norm(hand - shoulder)
    p[18] = p[20] = hand
    vis = [0.95] * 33
    if not hips_visible:
        vis[23] = vis[24] = 0.1
    return {"host_time_s": t, "world_landmarks": p.tolist(), "visibility": vis, "cue": cue}


class HandMapTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(MAP_FILE.read_text())
        self.limits = joint_limits(load_map(), load_home())

    def test_body_frame_falls_back_to_camera_vertical_when_seated(self):
        f = frame(0.3, -0.2, hips_visible=False)
        up, forward, hips = body_frame(np.asarray(f["world_landmarks"]), f["visibility"])
        self.assertFalse(hips)
        self.assertTrue(np.allclose(up, UP) and np.allclose(forward, FORWARD))

    def test_hand_track_is_in_arm_lengths(self):
        rows = [frame(0.3, -0.2, t=i * 0.03) for i in range(60)]
        track = hand_track(rows, "right")
        self.assertGreater(track["forward"].mean(), 0)
        self.assertLess(track["up"].mean(), 0)
        ratio = track["forward"].mean() / -track["up"].mean()
        self.assertAlmostEqual(ratio, 1.5, delta=0.05)

    def test_calibration_uses_touch_steps(self):
        rows = [frame(0.4 + 0.01 * (i % 10), -0.3, cue="touch", t=i * 0.03) for i in range(60)] + \
               [frame(0.2, 0.1, cue="raise", t=2 + i * 0.03) for i in range(30)]
        human, source = calibrate(hand_track(rows, "right"), self.config)
        self.assertIn("table targets", source)
        self.assertLess(human["table_up"], 0)
        self.assertLess(human["forward_range"][0], human["forward_range"][1])

    def test_targets_stay_in_the_safe_zone_and_on_or_above_the_table(self):
        track = {"forward": np.array([-1.0, 0.35, 0.85, 3.0]), "up": np.array([-2.0, -0.55, 0.0, 2.0]), "arm_m": 0.5}
        x, z = robot_targets(track, self.config["human"], self.config["robot"])
        lo, hi = self.config["robot"]["x_limits_m"]
        self.assertTrue(((x >= lo) & (x <= hi)).all())
        self.assertTrue((z >= self.config["robot"]["grasp_height_m"]).all())
        self.assertTrue((z <= self.config["robot"]["max_height_m"]).all())

    def test_solve_reaches_targets_smoothly(self):
        x = np.linspace(0.18, 0.24, 100)
        z = 0.03 + 0.08 * np.abs(np.sin(np.linspace(0, 3, 100)))
        angles, pitches, moved = solve(x, z, self.limits, 15)
        tips = np.array([forward_kinematics(a)[-1] for a in angles])
        self.assertLess(np.hypot(tips[:, 0] - x, tips[:, 1] - z).max(), 0.011)
        self.assertLessEqual(np.abs(np.diff(pitches)).max(), 15)


if __name__ == "__main__":
    unittest.main()
