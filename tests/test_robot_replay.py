"""Robot replay against a simulated arm bus. No physical servo is used."""

import json
from pathlib import Path
import tempfile
import unittest

from neural_reflex.arm_web import ArmController
from neural_reflex.leader_arm import JOINTS
from neural_reflex.robot_replay import PLANAR, Replay, apply_limits, load_plan, main, shoulder_load, summary


class SimBus:
    def __init__(self, start=2048):
        self.values = {n: {"Present_Position": start, "Min_Position_Limit": 1000, "Max_Position_Limit": 3000,
                           "Torque_Enable": 0, "Operating_Mode": 0, "Present_Voltage": 120,
                           "Present_Temperature": 30, "Torque_Limit": 1000, "Max_Torque_Limit": 1000}
                       for n in JOINTS}
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


def make_plan(folder: Path) -> Path:
    home = {n: 2048 for n in JOINTS}
    ramp = [2048 + min(i, 25) * 4 for i in range(50)]                 # +100 ticks over 0.5 s, then hold
    plan = {"block": "test", "source": "recorded", "speed": "fast", "slowdown": 1.0, "dt_s": 0.02,
            "home": home, "joints": list(PLANAR),
            "direction_checked": {"shoulder_lift": True, "elbow_flex": False, "wrist_flex": False},
            "limits_ticks": {j: [1060, 2940] for j in PLANAR}, "margin_ticks": 60,
            "targets": {"shoulder_lift": ramp, "elbow_flex": [2148] * 50, "wrist_flex": [2048] * 50}}
    path = folder / "robot_motor_plan.json"
    path.write_text(json.dumps(plan))
    return path


class ReplayTests(unittest.TestCase):
    def test_shoulder_load_matches_pick_sim(self):
        from neural_reflex.pick_sim import gravity_torque
        for pose in ((0, 0, 0), (90, 0, 0), (-38, 0, 0), (-20, 130, 60)):
            self.assertAlmostEqual(shoulder_load(*pose), gravity_torque(pose)["shoulder_lift"], delta=0.05)

    def test_amplitude_limit_clamps_moving_joints(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, first, last = load_plan(make_plan(Path(tmp)), ["shoulder_lift"], 0, None)
            plan["targets"]["shoulder_lift"] = [2048 + 600] * 50          # +53 degrees asked
            worst, _ = apply_limits(plan, ["shoulder_lift"], first, last, 20.0)
            self.assertLessEqual(max(plan["targets"]["shoulder_lift"]) - 2048, round(20 * 4096 / 360))
            self.assertLess(worst, 4.0)

    def test_replay_moves_only_listed_joints_and_ends_with_motors_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, first, last = load_plan(make_plan(Path(tmp)), ["shoulder_lift"], 0, None)
            bus = SimBus()
            controller = ArmController(bus, timeout=15)
            replay = Replay(controller, plan, ["shoulder_lift"], first, last, log=lambda *_: None)
            try:
                replay.start()
                replay.run()
            finally:
                replay.finish()
            moved = {n for n, v in bus.goals if v != 2048}
            self.assertEqual(moved, {"shoulder_lift"})                  # elbow plan ignored: it holds
            peak = max(v for n, v in bus.goals if n == "shoulder_lift")
            self.assertGreaterEqual(peak, 2140)
            self.assertTrue(all(1000 <= v <= 3000 for _, v in bus.goals))
            # Back home within the replay's arrival tolerance (15 ticks, about 1.3 degrees).
            self.assertLessEqual(abs(bus.values["shoulder_lift"]["Present_Position"] - 2048), 15)
            self.assertTrue(all(bus.values[n]["Torque_Enable"] == 0 for n in JOINTS))

    def test_refuses_to_start_away_from_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, first, last = load_plan(make_plan(Path(tmp)), ["shoulder_lift"], 0, None)
            bus = SimBus(start=2600)
            controller = ArmController(bus, timeout=15)
            replay = Replay(controller, plan, ["shoulder_lift"], first, last, log=lambda *_: None)
            try:
                with self.assertRaises(RuntimeError) as caught:
                    replay.start()
                self.assertIn("home pose", str(caught.exception))
                self.assertTrue(all(bus.values[n]["Torque_Enable"] == 0 for n in JOINTS))
            finally:
                replay.finish()

    def test_plan_outside_limits_is_rejected_and_dry_run_touches_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_plan(Path(tmp))
            plan = json.loads(path.read_text())
            text = summary(plan, ["shoulder_lift"], 0, 50)
            self.assertIn("MOVES", text)
            self.assertIn("holds at home", text)
            self.assertEqual(main([str(path)]), 0)                       # dry run, no port needed
            plan["targets"]["shoulder_lift"][10] = 2990
            path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                load_plan(path, ["shoulder_lift"], 0, None)


if __name__ == "__main__":
    unittest.main()
