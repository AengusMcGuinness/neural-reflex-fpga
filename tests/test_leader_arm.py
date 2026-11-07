"""Check the arm's motion boundaries without connecting physical hardware."""

import unittest

from neural_reflex.leader_arm import JOINTS, JointState, JogSession, run_probe


class FakeBus:
    def __init__(self):
        self.events = []
        self.values = {
            name: {
                "Present_Position": 2048,
                "Min_Position_Limit": 1000,
                "Max_Position_Limit": 3000,
                "Torque_Enable": 0,
                "Operating_Mode": 0,
            }
            for name in JOINTS
        }

    def connect(self):
        self.events.append(("connect",))

    def disconnect(self, *, disable_torque):
        self.events.append(("disconnect", disable_torque))

    def read(self, register, name, *, normalize):
        self.events.append(("read", register, name, normalize))
        return self.values[name][register]

    def write(self, register, name, value, *, normalize):
        self.events.append(("write", register, name, value, normalize))

    def enable_torque(self):
        self.events.append(("enable_torque",))

    def disable_torque(self):
        self.events.append(("disable_torque",))


class LeaderArmTests(unittest.TestCase):
    def setUp(self):
        self.bus = FakeBus()
        self.states = {
            name: JointState(2048, 1000, 3000, False, 0)
            for name in JOINTS
        }

    def test_probe_never_changes_torque_or_position(self):
        run_probe(self.bus)
        self.assertEqual(self.bus.events[0], ("connect",))
        self.assertEqual(self.bus.events[-1], ("disconnect", False))
        self.assertFalse(any(event[0] in ("write", "enable_torque", "disable_torque")
                             for event in self.bus.events))

    def test_jog_requires_arm_and_holds_current_pose_before_torque(self):
        session = JogSession(self.bus, self.states)
        with self.assertRaises(RuntimeError):
            session.jog("shoulder_pan", 5)
        session.arm()
        writes = [event for event in self.bus.events if event[0] == "write"]
        self.assertEqual(len(writes), 6)
        self.assertTrue(all(event[1] == "Goal_Position" and event[3] == 2048
                            and event[4] is False for event in writes))
        self.assertEqual(self.bus.events[-1], ("enable_torque",))
        self.assertEqual(session.jog("shoulder_pan", 10), 2058)
        session.stop()
        self.assertEqual(self.bus.events[-1], ("disable_torque",))

    def test_jog_rejects_large_steps_and_travel(self):
        session = JogSession(self.bus, self.states)
        session.arm()
        with self.assertRaises(ValueError):
            session.jog("shoulder_pan", 11)
        for _ in range(8):
            session.jog("shoulder_pan", 10)
        with self.assertRaises(ValueError):
            session.jog("shoulder_pan", 1)
        session.stop()

    def test_jog_rejects_existing_torque(self):
        self.states["elbow_flex"] = JointState(2048, 1000, 3000, True, 0)
        with self.assertRaises(ValueError):
            JogSession(self.bus, self.states)


if __name__ == "__main__":
    unittest.main()
