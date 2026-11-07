"""Direction check logic, without an arm or a terminal."""

import copy
import json
import unittest

from neural_reflex.direction_check import apply_results, check_move, sign_for
from neural_reflex.robot_map import MAP_FILE

START = {"shoulder_pan": 2000, "shoulder_lift": 2000, "elbow_flex": 1100, "wrist_flex": 1900,
         "wrist_roll": 1900, "gripper": 1700}


class DirectionCheckTests(unittest.TestCase):
    def test_signs(self):
        self.assertEqual(sign_for("shoulder_lift", +300), -1)   # forward tilt = lowering your arm
        self.assertEqual(sign_for("shoulder_lift", -300), +1)
        self.assertEqual(sign_for("elbow_flex", +300), +1)
        self.assertEqual(sign_for("shoulder_pan", +300, "right"), +1)
        self.assertEqual(sign_for("shoulder_pan", +300, "left"), -1)

    def test_move_validation(self):
        moved = dict(START, elbow_flex=1400)
        self.assertIsNone(check_move(START, moved, "elbow_flex"))
        self.assertIn("only moved", check_move(START, dict(START, elbow_flex=1130), "elbow_flex"))
        self.assertIn("moved more", check_move(START, dict(moved, shoulder_lift=2600), "elbow_flex"))

    def test_apply_marks_verified_but_not_roll(self):
        config = copy.deepcopy(json.loads(MAP_FILE.read_text()))
        out = apply_results(config, {"elbow_flex": -250, "wrist_roll": 400}, "right")
        self.assertEqual(out["joints"]["elbow_flex"]["sign"], -1)
        self.assertTrue(out["joints"]["elbow_flex"]["verified"])
        self.assertFalse(out["joints"]["wrist_roll"]["verified"])
        self.assertEqual(out["joints"]["wrist_roll"]["robot_clockwise_tick_sign"], 1)


if __name__ == "__main__":
    unittest.main()
