"""Live IMU control with synthetic band orientations and a simulated arm bus."""

import json
import math
import time
import unittest

import numpy as np

from neural_reflex.arm_web import ArmController
from neural_reflex.imu_teleop import (CONFIG_FILE, HOME_FILE, MOVING, ROBOT_MAP, Calibration, Mapper, quat_to_matrix,
                                      rotation_vector, run_robot)
from neural_reflex.leader_arm import JOINTS


def rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = math.radians(deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k


def matrix_to_quat(m):
    w = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    return [w, (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w)]


# Some band mounting: neutral orientation and the world axes the user's moves turn about.
NEUTRAL = rot([0.3, -0.5, 0.8], 70)
TILT_AXIS, SWING_AXIS, ROLL_AXIS = np.array([1.0, 0.2, 0]), np.array([0, 0, 1.0]), np.array([-0.2, 1.0, 0])


def calibration():
    return Calibration.from_poses(NEUTRAL, rot(TILT_AXIS, 45) @ NEUTRAL, rot(SWING_AXIS, 45) @ NEUTRAL,
                                  rot(ROLL_AXIS, 80) @ NEUTRAL)


class SimBus:
    def __init__(self, positions, limits):
        self.values = {n: {"Present_Position": positions[n], "Min_Position_Limit": limits[n][0],
                           "Max_Position_Limit": limits[n][1], "Torque_Enable": 0, "Operating_Mode": 0,
                           "Present_Voltage": 120, "Present_Temperature": 30, "Torque_Limit": 1000,
                           "Max_Torque_Limit": 1000} for n in JOINTS}
        self.goals = []

    def connect(self):
        pass

    def disconnect(self, *, disable_torque):
        pass

    def read(self, register, name, *, normalize):
        return self.values[name][register]

    def write(self, register, name, value, *, normalize):
        if register == "Goal_Position":
            self.goals.append((name, value))
            if self.values[name]["Torque_Enable"]:
                self.values[name]["Present_Position"] = value

    def enable_torque(self, name):
        self.values[name]["Torque_Enable"] = 1

    def disable_torque(self, name):
        self.values[name]["Torque_Enable"] = 0


class FakeFeed:
    def __init__(self, matrix):
        self.matrix, self.fresh = matrix, True

    def get(self):
        if not self.fresh:
            return (time.monotonic() - 5.0, matrix_to_quat(self.matrix))
        return (time.monotonic(), matrix_to_quat(self.matrix))


class ImuTeleopTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(CONFIG_FILE.read_text())
        self.robot_map = json.loads(ROBOT_MAP.read_text())
        self.home = json.loads(HOME_FILE.read_text())["positions"]

    def test_rotation_round_trips(self):
        m = rot([1, 2, 3], 37)
        self.assertTrue(np.allclose(quat_to_matrix(matrix_to_quat(m)), m, atol=1e-9))
        self.assertAlmostEqual(math.degrees(np.linalg.norm(rotation_vector(m))), 37, places=6)

    def test_calibration_recovers_each_axis_whatever_the_band_mounting(self):
        cal = calibration()
        for axis, idx in ((TILT_AXIS, 0), (SWING_AXIS, 1)):
            angles = cal.angles(rot(axis, 30) @ NEUTRAL)
            self.assertAlmostEqual(angles[idx], 30, delta=1.0)
        combo = cal.angles(rot(TILT_AXIS, 20) @ rot(SWING_AXIS, -15) @ NEUTRAL)
        self.assertAlmostEqual(combo[0], 20, delta=3)
        self.assertAlmostEqual(combo[1], -15, delta=3)

    def test_too_small_calibration_move_is_rejected(self):
        with self.assertRaises(ValueError):
            Calibration.from_poses(NEUTRAL, rot(TILT_AXIS, 5) @ NEUTRAL, rot(SWING_AXIS, 45) @ NEUTRAL,
                                   rot(ROLL_AXIS, 80) @ NEUTRAL)

    def test_mapper_moves_all_joints_inside_limits(self):
        mapper = Mapper(self.cfg, self.robot_map, self.home)
        low, high = mapper.degrees(-90, 30, 60), mapper.degrees(90, -30, -60)
        self.assertLess(low["height_m"], high["height_m"])
        for j in MOVING:
            self.assertNotAlmostEqual(low[j], high[j], places=0, msg=j)
            for deg in (low, high):
                t = mapper.ticks(deg)[j]
                self.assertTrue(mapper.tick_limits[j][0] <= t <= mapper.tick_limits[j][1], j)
        self.assertLessEqual(abs(mapper.degrees(0, 500, 0)["shoulder_pan"]), self.cfg["base_limit_deg"])

    def run_arm(self, start, feed, duration=1.2):
        limits = {j: tuple(self.robot_map["limits"][j]) for j in JOINTS}
        bus = SimBus(start, limits)
        controller = ArmController(bus, timeout=15)
        mapper = Mapper(self.cfg, self.robot_map, self.home)
        try:
            run_robot(controller, feed, calibration(), mapper, self.cfg, log=lambda *_: None, duration=duration)
        finally:
            controller.close()
        return bus, mapper

    def test_live_loop_moves_five_joints_together_and_ends_off(self):
        feed = FakeFeed(rot(TILT_AXIS, 25) @ rot(SWING_AXIS, 20) @ NEUTRAL)
        bus, mapper = self.run_arm(dict(self.home), feed)
        moved = {n for n, v in bus.goals if v != self.home[n]}
        self.assertTrue(set(MOVING) <= moved, moved)
        self.assertNotIn("gripper", moved)
        for n, v in bus.goals:
            lo, hi = self.robot_map["limits"][n]
            self.assertTrue(lo <= v <= hi)
        self.assertTrue(all(bus.values[n]["Torque_Enable"] == 0 for n in JOINTS))

    def test_stale_band_data_holds_still(self):
        feed = FakeFeed(rot(TILT_AXIS, 25) @ NEUTRAL)
        feed.fresh = False                                   # data 5 s old: must not chase it
        bus, mapper = self.run_arm(dict(self.home), feed, duration=0.6)
        neutral = mapper.ticks(mapper.degrees(0, 0, 0))
        for n in MOVING:
            self.assertLessEqual(abs(bus.values[n]["Present_Position"] - neutral[n]), 25, n)

    def test_refuses_to_start_away_from_home(self):
        start = dict(self.home, elbow_flex=self.home["elbow_flex"] + 600)
        with self.assertRaises(RuntimeError):
            self.run_arm(start, FakeFeed(NEUTRAL), duration=0.2)


if __name__ == "__main__":
    unittest.main()
