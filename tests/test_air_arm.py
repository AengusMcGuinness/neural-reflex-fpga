"""Air-mouse style arm control with synthetic band data and a simulated arm."""

import json
import math
import time
import unittest

import numpy as np

from neural_reflex.air_arm import AirAim, AirArm, FistMeter, OneEuro, follow_band, forearm_aim
from neural_reflex.arm_web import ArmController
from neural_reflex.imu_teleop import HOME_FILE, ROBOT_MAP, Mapper
from neural_reflex.imu_teleop import CONFIG_FILE as TELEOP_CONFIG
from neural_reflex.leader_arm import JOINTS
from neural_reflex.movements import BandStream
from tests.test_imu_teleop import SimBus, matrix_to_quat, rot


def band(azimuth=90.0, elevation=0.0, twist=0.0):
    """Quaternion for a right-wrist band whose forearm (+y) points at azimuth/elevation, twisted."""
    # Start: forearm along world +y (azimuth 90), band +x along world +x... then twist, raise, turn.
    m = rot([0, 0, 1], azimuth - 90) @ rot([1, 0, 0], elevation) @ rot([0, 1, 0], twist)
    return matrix_to_quat(m)


class AirArmTests(unittest.TestCase):
    def test_forearm_aim_follows_kinesis_axes(self):
        az, el, tw = forearm_aim(band(90, 0, 0))
        self.assertAlmostEqual(az, 90, delta=0.5)
        self.assertAlmostEqual(el, 0, delta=0.5)
        self.assertAlmostEqual(abs(tw), 0, delta=0.5)
        self.assertAlmostEqual(forearm_aim(band(60, 30, 0))[1], 30, delta=0.5)      # raise -> elevation
        self.assertAlmostEqual(forearm_aim(band(60, 30, 0))[0], 60, delta=0.5)      # turn -> azimuth
        a = forearm_aim(band(90, 0, 40))
        self.assertAlmostEqual(a[1], 0, delta=0.5)                                  # twist leaves pointing alone
        self.assertAlmostEqual(abs(a[2]), 40, delta=1)

    def test_air_aim_reports_changes_since_start(self):
        aim = AirAim("right")
        aim.update(band(170, 0, 0), 0.0)
        el, swing, twist = 0, 0, 0
        for k in range(1, 200):                                     # cross the ±180° seam smoothly
            el, swing, twist = aim.update(band(170 + 20 * min(1, k / 50), 10, 15), k * 0.02)
        self.assertAlmostEqual(swing, 20, delta=1.5)
        self.assertAlmostEqual(el, 10, delta=1.5)
        self.assertAlmostEqual(abs(twist), 15, delta=2)

    def test_one_euro_smooths_jitter_but_follows_moves(self):
        f, rng = OneEuro(), np.random.default_rng(0)
        still = [f(0.5 * rng.standard_normal(), k * 0.02) for k in range(100)]
        self.assertLess(np.std(still[20:]), 0.25)
        moved = [f(30.0, 2 + k * 0.02) for k in range(50)]
        self.assertGreater(moved[-1], 27)

    def test_fist_meter(self):
        meter = FistMeter(relaxed=[10] * 8, fist=[12, 60, 15, 10, 10, 10, 10, 10])
        self.assertEqual(meter.channel, 1)
        self.assertAlmostEqual(meter.level(np.array([10] * 8, float)), 0.0)
        self.assertAlmostEqual(meter.level(np.array([10, 60, 10, 10, 10, 10, 10, 10], float)), 1.0)
        with self.assertRaises(ValueError):
            FistMeter(relaxed=[10] * 8, fist=[12] * 8)

    def test_follow_band_moves_all_six_motors(self):
        cfg = json.loads(TELEOP_CONFIG.read_text())
        robot_map = json.loads(ROBOT_MAP.read_text())
        home = json.loads(HOME_FILE.read_text())["positions"]
        stream = BandStream.offline(keep_s=10)
        rng, t0 = np.random.default_rng(2), time.monotonic()
        for k in range(int(4 * 128)):                    # 4 s: start level, then raise, swing, twist and fist
            t = t0 - 1.0 + k / 128
            s = min(1.0, max(0.0, (t - t0) / 1.0))
            gain = 1 + 7 * s
            stream.add({"type": "emg", "host_time_s": t,
                        "payload": {"values": (32768 + rng.standard_normal((16, 8)) * 20 * gain).tolist()}})
            stream.add({"type": "orientation", "host_time_s": t,
                        "payload": {"values": band(90 - 25 * s, 25 * s, 30 * s)}})
        air = AirArm(stream, AirAim("right"), FistMeter([np.mean(np.abs(np.diff(rng.standard_normal(400) * 20)))] * 8,
                                                          [8 * 22.6] * 8), Mapper(cfg, robot_map, home), home, robot_map)
        air.tick(t0 - 0.5)                                # neutral: level, at the starting azimuth
        start = dict(home, shoulder_pan=home["shoulder_pan"] + 400, gripper=home["gripper"] + 300)   # base turned, handle moved
        bus = SimBus(start, {j: tuple(robot_map["limits"][j]) for j in JOINTS})
        controller = ArmController(bus, timeout=15)
        try:
            follow_band(controller, air, duration=1.5, show=False)
        finally:
            controller.close()
        moved = {n for n, v in bus.goals if v != start[n]}
        self.assertTrue({"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"} <= moved, moved)
        for n, v in bus.goals:
            lo, hi = robot_map["limits"][n]
            self.assertTrue(lo <= v <= hi)
        self.assertTrue(all(bus.values[n]["Torque_Enable"] == 0 for n in JOINTS))


if __name__ == "__main__":
    unittest.main()
