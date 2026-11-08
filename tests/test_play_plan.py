"""Plan playback against a fake servo bus: torque only on moving joints, always off at the end."""

import unittest

from neural_reflex import play_plan
from neural_reflex.leader_arm import JOINTS


class FakeBus:
    def __init__(self, stuck=None):
        self.pos = {j: 2000 for j in JOINTS}
        self.torque = {j: 0 for j in JOINTS}
        self.stuck = stuck            # joint that never reaches its goal
        self.writes = []

    def read(self, register, name, normalize=False):
        return {"Present_Position": self.pos[name], "Min_Position_Limit": 500, "Max_Position_Limit": 3500,
                "Torque_Enable": self.torque[name], "Operating_Mode": 0}[register]

    def write(self, register, name, value, normalize=False):
        assert register == "Goal_Position" and self.torque[name] or value == self.pos[name]
        self.writes.append((name, value))
        if name != self.stuck:
            self.pos[name] = value

    def enable_torque(self, name):
        self.torque[name] = 1

    def disable_torque(self, name):
        self.torque[name] = 0


class PlayPlanTests(unittest.TestCase):
    def goals(self):
        targets = {"shoulder_lift": [2000 + i for i in range(0, 300, 3)], "elbow_flex": [2000] * 100}
        return play_plan.schedule({j: __import__("numpy").array(v, float) for j, v in targets.items()},
                                  {"shoulder_lift": 2000, "elbow_flex": 2000}, cap=180, fine=60, dt=0.02)

    def test_runs_and_turns_torque_off(self):
        bus = FakeBus()
        play_plan.play(bus, self.goals(), dt=0.0001, log=lambda *_: None)
        self.assertEqual({n for n, _ in bus.writes}, {"shoulder_lift", "elbow_flex"})
        self.assertFalse(any(bus.torque.values()))
        self.assertGreater(bus.pos["shoulder_lift"], 2000)

    def test_step_never_exceeds_cap(self):
        goals = self.goals()["shoulder_lift"]
        self.assertLessEqual(max(abs(b - a) for a, b in zip(goals, goals[1:])), 4)  # 180 ticks/s * 20 ms, rounded

    def test_stuck_joint_stops_with_torque_off(self):
        bus = FakeBus(stuck="shoulder_lift")
        old = play_plan.TRACKING_TIMEOUT
        play_plan.TRACKING_TIMEOUT = 0.0
        try:
            with self.assertRaises(RuntimeError):
                play_plan.play(bus, self.goals(), dt=0.0001, log=lambda *_: None)
        finally:
            play_plan.TRACKING_TIMEOUT = old
        self.assertFalse(any(bus.torque.values()))

    def test_refuses_when_torque_already_on(self):
        bus = FakeBus()
        bus.torque["wrist_roll"] = 1
        with self.assertRaises(ValueError):
            play_plan.play(bus, self.goals(), dt=0.0001, log=lambda *_: None)


if __name__ == "__main__":
    unittest.main()
