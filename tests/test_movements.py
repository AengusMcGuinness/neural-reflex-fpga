"""Movement teaching and control with synthetic band data and a simulated arm."""

import json
import math
import unittest

import numpy as np

from neural_reflex.arm_web import ArmController
from neural_reflex.leader_arm import JOINTS
from neural_reflex.movements import (CONFIG_FILE, HOME_FILE, ROBOT_MAP, BandStream, Classifier, Detector, Jogger,
                                     control_loop, cross_validate, dataset, features)

NEUTRAL_Q = [0.9, 0.1, 0.3, 0.2]


def q_rotated(axis, deg):
    """NEUTRAL_Q followed by a rotation about a band-frame axis."""
    a = math.radians(deg) / 2
    w, x, y, z = NEUTRAL_Q / np.linalg.norm(NEUTRAL_Q)
    bw, (bx, by, bz) = math.cos(a), np.asarray(axis, float) / np.linalg.norm(axis) * math.sin(a)
    return [w * bw - x * bx - y * by - z * bz, w * bx + x * bw + y * bz - z * by,
            w * by - x * bz + y * bw + z * bx, w * bz + x * by - y * bx + z * bw]


# What each synthetic movement does to the band: (channel gains, rotation axis, rotation degrees)
SIGNATURES = {"rest": (np.ones(8), None, 0), "fist": (np.array([6, 8, 3, 1, 1, 1, 2, 1.0]), None, 0),
              "index_pinch": (np.array([1, 1, 5, 6, 1, 1, 1, 1.0]), None, 0),
              "swing_left": (np.ones(8), [0, 0, 1], -35), "tilt_up": (np.ones(8), [1, 0, 0], 35)}


def feed(stream, name, t0, seconds, rng):
    gains, axis, deg = SIGNATURES[name]
    q = NEUTRAL_Q if axis is None else q_rotated(axis, deg)
    for k in range(int(seconds * 128)):
        t = t0 + k / 128
        values = 32768 + rng.standard_normal((16, 8)) * 20 * gains
        stream.add({"type": "emg", "host_time_s": t, "payload": {"values": values.tolist()}})
        stream.add({"type": "gyro", "host_time_s": t, "payload": {"values": (rng.standard_normal(3) * 5).tolist()}})
        stream.add({"type": "orientation", "host_time_s": t, "payload": {"values": q}})


def synthetic_session(names, reps=3, seed=0):
    rng = np.random.default_rng(seed)
    stream, segments, t = BandStream.offline(), [], 100.0
    for rep in range(1, reps + 1):
        for name in names:
            feed(stream, name, t, 2.0, rng)
            segments.append({"kind": "move" if name != "rest" else "rest", "name": name, "rep": rep,
                             "start": t, "end": t + 2.0})
            t += 2.0
    return stream, segments


class MovementTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(CONFIG_FILE.read_text())
        self.robot_map = json.loads(ROBOT_MAP.read_text())
        self.home = json.loads(HOME_FILE.read_text())["positions"]

    def test_features_need_enough_data(self):
        self.assertIsNone(features(None))
        self.assertIsNone(features(np.ones((64, 8))))                    # far short of a 250 ms window
        x = np.random.default_rng(0).standard_normal((512, 8))
        self.assertEqual(len(features(x)), 140)
        self.assertEqual(len(features(x, np.zeros((5, 3)), [0, 0, 0], use_imu=True)), 149)
        self.assertIsNone(features(x, None, [0, 0, 0], use_imu=True))

    def test_teaching_recognises_distinct_movements_on_held_out_repetitions(self):
        from neural_reflex.imu_teleop import quat_to_matrix
        names = list(SIGNATURES)
        stream, segments = synthetic_session(names)
        x, y, groups = dataset(stream, segments, quat_to_matrix(NEUTRAL_Q), use_imu=True)
        accuracy, recall, _ = cross_validate(x, y, groups, names)
        self.assertGreater(accuracy, 0.95)
        self.assertTrue(all(v > 0.9 for v in recall.values()), recall)
        hand = ["rest", "fist", "index_pinch"]                           # EMG only, no neutral pose needed
        x2, y2, g2 = dataset(stream, segments, None, names=set(hand))
        self.assertEqual(set(y2), set(hand))
        self.assertGreater(cross_validate(x2, y2, g2, hand)[0], 0.95)
        model = Classifier.fit(x, y, names)
        again = Classifier.from_json(json.loads(json.dumps(model.to_json())))
        self.assertTrue(np.allclose(again.proba(x[:5]), model.proba(x[:5])))
        self.assertTrue(np.allclose(model.proba(x[:5]).sum(axis=1), 1))

    def test_detector_needs_a_held_confident_movement(self):
        d = Detector(hold_s=0.3, min_confidence=0.7)
        self.assertEqual(d.update({"rest": 0.2, "fist": 0.8}, 0.0), "rest")       # just started
        self.assertEqual(d.update({"rest": 0.2, "fist": 0.8}, 0.35), "fist")      # held long enough
        self.assertEqual(d.update({"rest": 0.4, "fist": 0.6}, 0.4), "rest")       # not confident -> stop
        self.assertEqual(d.update({"rest": 0.1, "fist": 0.9}, 0.45), "rest")      # must be held again

    def test_jogger_moves_only_the_mapped_motor_and_respects_limits(self):
        jog = Jogger(self.cfg, self.robot_map, self.home)
        start = dict(jog.targets)
        out = jog.step("fist", 0.5)
        self.assertGreater(out["gripper"], start["gripper"])
        self.assertEqual({j for j in out if out[j] != round(start[j])}, {"gripper"})
        for _ in range(400):
            out = jog.step("fist", 0.1)
        self.assertEqual(out["gripper"], jog.limits["gripper"][1])
        held = jog.step("rest", 1.0)
        self.assertEqual(held, out)

    def test_jogger_refuses_to_drive_into_the_table(self):
        jog = Jogger(self.cfg, self.robot_map, self.home)
        for _ in range(300):                           # keep tilting the shoulder forward/down
            jog.step("tilt_down", 0.1)
        self.assertIsNotNone(jog.blocked)
        planar = [(jog.targets[p] - self.home[p]) / (4096 / 360) for p in ("shoulder_lift", "elbow_flex", "wrist_flex")]
        self.assertEqual(jog.check_pose(planar, jog.deg_limits), [])

    def test_control_loop_drives_the_simulated_arm(self):
        from neural_reflex.imu_teleop import quat_to_matrix
        from tests.test_imu_teleop import SimBus
        names = list(SIGNATURES)
        train_stream, segments = synthetic_session(names)
        x, y, _ = dataset(train_stream, segments, quat_to_matrix(NEUTRAL_Q), use_imu=True)
        model = Classifier.fit(x, y, names)
        cfg = dict(self.cfg, movements=[m for m in self.cfg["movements"] if m["name"] in names])
        jog = Jogger(cfg, self.robot_map, self.home)
        bus = SimBus(dict(self.home), {j: tuple(self.robot_map["limits"][j]) for j in JOINTS})
        controller = ArmController(bus, timeout=15)
        for j in jog.motors:
            controller.torque(j, True, controller.motion_epoch)
        import threading, time
        live = BandStream.offline(keep_s=5)
        feed(live, "fist", time.monotonic() - 1.0, 3.0, np.random.default_rng(1))   # a held fist, on the real clock
        stop = threading.Event()

        def motion_loop():
            while not stop.wait(0.02):
                controller.motion_step()

        threading.Thread(target=motion_loop, daemon=True).start()
        try:
            control_loop(live, model, cfg, jog, quat_to_matrix(NEUTRAL_Q), controller, duration=1.0, show=False,
                         use_imu=True)
        finally:
            stop.set()
            controller.close()
        self.assertGreater(jog.targets["gripper"], self.home["gripper"] + 50)
        for j in ("shoulder_lift", "elbow_flex", "wrist_flex", "shoulder_pan"):
            self.assertEqual(round(jog.targets[j]), self.home[j], j)
        self.assertTrue(all(bus.values[n]["Torque_Enable"] == 0 for n in JOINTS))


if __name__ == "__main__":
    unittest.main()
