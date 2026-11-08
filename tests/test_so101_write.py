"""Writing demo: text strokes, Newton-only plan for "hi", tick conversion, the dry run, the read-only
arm check, the elbow mechanical stop and the keep-alive at the MOVE prompt (simulated bus).

Run: .venv-ik/bin/python -m unittest tests.test_so101_write
"""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from so101_ik import write_arm, write_plan
from so101_ik.text_strokes import MAX_SPACING_M, strokes_bbox, text_to_strokes

REPO = Path(__file__).resolve().parent.parent


class TextStrokesTest(unittest.TestCase):
    def test_hello_neo_strokes(self):
        strokes = text_to_strokes("hello Neo", 0.03)
        self.assertGreater(len(strokes), 5)
        xmin, ymin, xmax, ymax = strokes_bbox(strokes)
        self.assertAlmostEqual(xmin, 0.0, places=9)
        self.assertAlmostEqual(ymax, 0.03, places=6)            # cap height
        self.assertGreater(xmax, 0.1)
        self.assertLess(xmax, 0.3)
        for s in strokes:
            self.assertEqual(s.ndim, 2)
            self.assertEqual(s.shape[1], 2)
            self.assertGreaterEqual(len(s), 2)
            self.assertTrue(np.all(s[:, 0] >= xmin - 1e-12) and np.all(s[:, 0] <= xmax + 1e-12))
            self.assertTrue(np.all(s[:, 1] >= ymin - 1e-12) and np.all(s[:, 1] <= ymax + 1e-12))
            spacing = np.linalg.norm(np.diff(s, axis=0), axis=1)
            self.assertLessEqual(spacing.max(), MAX_SPACING_M + 1e-9)

    def test_empty_text(self):
        self.assertEqual(text_to_strokes("", 0.03), [])


class WritePlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "plan.json"
        cls.png = Path(cls.tmp.name) / "plan.png"
        buf = io.StringIO()
        with redirect_stdout(buf):
            cls.code = write_plan.main(["--text", "hi", "--distance_m", "0.20",
                                        "--plane_z_m", str(write_plan.TABLE_Z_M),
                                        "--out", str(cls.out), "--preview", str(cls.png)])
        cls.stdout = buf.getvalue()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_newton_only_plan_converges(self):
        self.assertIn("Newton only", self.stdout)
        self.assertTrue(self.out.exists())
        self.assertTrue(self.png.exists() and self.png.stat().st_size > 1000)
        plan = json.loads(self.out.read_text())
        s = plan["summary"]
        self.assertEqual(s["solver"], "newton-only")
        self.assertGreater(s["convergence_pct"], 95.0)
        self.assertEqual(s["n_accepted"] + s["n_rejected"], s["n_waypoints"])
        self.assertTrue(any(w["pen_down"] for w in plan["waypoints"]))
        self.assertTrue(any(not w["pen_down"] for w in plan["waypoints"]))
        for w in plan["waypoints"]:
            self.assertTrue(w["converged"])
            self.assertEqual(len(w["q_rad"]), 5)
            self.assertLess(w["pos_err_mm"], 1.0)
            z = w["xyz"][2]
            self.assertAlmostEqual(z, write_plan.TABLE_Z_M + (0.0 if w["pen_down"] else 0.01), places=6)

    def test_waypoints_hit_targets_with_pen_down(self):
        from so101_ik.fk import build_so101_model, forward_kinematics
        plan = json.loads(self.out.read_text())
        model = build_so101_model(tuple(plan["params"]["pen_tip_offset_m"]))
        for w in plan["waypoints"][::7]:
            T = forward_kinematics(np.array(w["q_rad"]), model)
            self.assertLess(np.linalg.norm(T[:3, 3] - np.array(w["xyz"])) * 1000, 1.0)
            self.assertLess(T[2, 2], -0.999)                     # tool z pointing down

    def test_write_arm_dry_run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = write_arm.main(["--plan", str(self.out)])
        text = buf.getvalue()
        self.assertEqual(code, 0, text)
        self.assertIn("DRY RUN", text)
        self.assertIn("shoulder_p", text)                        # table header
        self.assertIn("peak tick velocity", text)
        self.assertIn("360", text)
        self.assertGreater(text.count("\n"), 60)                 # one row per waypoint
        self.assertIn("elbow cap: 2757 ticks = home + 148 deg mechanical stop", text)
        self.assertRegex(text, r"peak elbow bend \d+\.\d deg")
        self.assertIn("approach to the first hover point at 'fine' (60 ticks/s)", text)

    def test_plan_holds_wrist_roll_and_reports_elbow(self):
        plan = json.loads(self.out.read_text())
        s = plan["summary"]
        lo, hi = s["wrist_roll_range_rad"]
        self.assertAlmostEqual(lo, hi, places=9)
        self.assertAlmostEqual(lo, write_plan.ROLL_MID_RAD, places=5)
        self.assertEqual(s["elbow_max_deg"], 148)
        self.assertLess(s["peak_elbow_bend_deg"], 148)
        self.assertGreater(s["peak_elbow_bend_deg"], 0)
        self.assertIn("peak elbow bend", self.stdout)
        self.assertIn("wrist roll", self.stdout)

    def test_plan_rejects_elbow_beyond_mechanical_stop(self):
        plan = json.loads(self.out.read_text())
        peak = plan["summary"]["peak_elbow_bend_deg"]
        with tempfile.TemporaryDirectory() as d:
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = write_plan.main(["--text", "hi", "--distance_m", "0.20", "--elbow_max_deg", str(peak - 5),
                                        "--out", str(Path(d) / "p.json"), "--preview", ""])
        self.assertEqual(code, 2)
        self.assertIn("REJECTED", buf.getvalue())
        self.assertIn("mechanical stop", buf.getvalue())


class TickConversionTest(unittest.TestCase):
    def setUp(self):
        self.cal = write_arm.load_calibration(REPO / "config" / "so101_ik_calibration.json")

    def test_calibration_shape(self):
        home = json.loads((REPO / "config" / "arm_home.json").read_text())["positions"]
        robot_map = json.loads((REPO / "config" / "robot_map.json").read_text())
        for j in write_arm.JOINTS:
            c = self.cal["joints"][j]
            self.assertEqual(c["home_tick"], home[j])
            self.assertEqual(list(c["limits_ticks"]), list(robot_map["limits"][j]))
            self.assertIn(c["sign"], (1, -1))
            self.assertFalse(c["verified"])
            self.assertTrue(c["how_to_verify"])
        self.assertNotIn("gripper", self.cal["joints"])

    def test_roundtrip(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = rng.uniform(-1.5, 1.5, 5)
            ticks = write_arm.q_to_ticks(q, self.cal)
            np.testing.assert_allclose(write_arm.ticks_to_q(ticks, self.cal), q, atol=1e-9)
            back = write_arm.ticks_to_q(np.rint(ticks), self.cal)
            self.assertLess(np.abs(back - q).max(), 0.5 / write_arm.TICKS_PER_RAD + 1e-12)
        home_q = np.array([self.cal["joints"][j]["q_home_rad"] for j in write_arm.JOINTS])
        home_t = np.array([self.cal["joints"][j]["home_tick"] for j in write_arm.JOINTS], float)
        np.testing.assert_allclose(write_arm.q_to_ticks(home_q, self.cal), home_t)

    def test_shoulder_lift_forward_is_more_ticks(self):
        c = self.cal["joints"]["shoulder_lift"]
        q = np.array([0.0, c["q_home_rad"] + 0.5, 0.0, 0.0, 0.0])    # tilt forward (about +y)
        self.assertGreater(write_arm.q_to_ticks(q, self.cal)[1], c["home_tick"])

    def test_refuses_out_of_range(self):
        plan = {"text": "x", "waypoints": [{"t": 0, "xyz": [0, 0, 0], "pen_down": False,
                                            "q_rad": [1.9, 0, 0, 0, 0]}]}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.json"
            path.write_text(json.dumps(plan))
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = write_arm.main(["--plan", str(path)])
        self.assertEqual(code, 2)
        self.assertIn("REFUSED", buf.getvalue())
        self.assertIn("shoulder_pan", buf.getvalue())

    def test_schedule_respects_cap(self):
        ticks = np.array([[0, 0, 0, 0, 0], [100, -50, 0, 0, 0], [100, -50, 9, 0, 0]])
        targets, owner = write_arm.expand_schedule(ticks, 20.0, 180.0)
        self.assertLessEqual(np.abs(np.diff(targets, axis=0)).max(), 9)
        self.assertEqual(owner[0], 0)
        self.assertEqual(owner[-1], 2)
        np.testing.assert_array_equal(targets[-1], ticks[-1])


class SimulatedDriveTest(unittest.TestCase):
    """drive() against the simulated bus: torque on the five joints only, targets logged, gripper untouched."""

    def test_drive_simulated(self):
        from neural_reflex.arm_web import ArmController
        from neural_reflex.leader_arm import JOINTS as ALL_JOINTS
        from tests.helpers import SimBus

        cal = write_arm.load_calibration(REPO / "config" / "so101_ik_calibration.json")
        robot_map = json.loads((REPO / "config" / "robot_map.json").read_text())
        strokes_plan = write_plan.build_plan(write_plan.parse_args(
            ["--text", "i", "--distance_m", "0.20", "--out", "unused", "--preview", ""]), log=lambda *a: None)[0]
        strokes_plan["waypoints"] = strokes_plan["waypoints"][:12]
        ticks, clamp, _ = write_arm.plan_to_ticks(strokes_plan, cal)
        self.assertEqual(write_arm.check_clamping(clamp, cal), [])
        targets, owner = write_arm.expand_schedule(ticks, 50.0, 360.0)
        start = {j: int(ticks[0][i]) for i, j in enumerate(write_arm.JOINTS)}
        start["gripper"] = 1776
        bus = SimBus(start, {j: tuple(robot_map["limits"][j]) for j in ALL_JOINTS})
        controller = ArmController(bus, timeout=15)
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "run.jsonl"
            buf = io.StringIO()
            log = write_arm.RunLog(log_path)
            ok = write_arm.drive(controller, targets, owner, strokes_plan, speed="fast", rate_hz=50.0,
                                 start_pause_s=0.0, log=log, ask=lambda _: "MOVE", out=buf)
            controller.close()
            log.close()
            rows = [json.loads(line) for line in log_path.read_text().splitlines()]
        self.assertTrue(ok, buf.getvalue())
        self.assertEqual(bus.values["gripper"]["Torque_Enable"], 0)
        self.assertFalse(any(name == "gripper" for name, _ in bus.goals))
        steps = [r for r in rows if "step" in r and r.get("event") is None]
        self.assertEqual(len(steps), len(targets))
        self.assertTrue(all("targets" in r for r in steps))
        self.assertTrue(any(r.get("positions") for r in steps))
        final = {j: bus.values[j]["Present_Position"] for j in write_arm.JOINTS}
        for i, j in enumerate(write_arm.JOINTS):
            self.assertEqual(final[j], int(targets[-1][i]), j)
        self.assertIn("finished writing", buf.getvalue())
        self.assertEqual(controller.speed, "fast")                              # switched after MOVE
        self.assertIn("at 'fine' speed", buf.getvalue())
        self.assertIn("writing at 'fast' speed", buf.getvalue())


def _sim_setup(n_waypoints=12):
    """(plan, cal, ticks, targets, owner, robot_map) for a short Newton-only plan."""
    cal = write_arm.load_calibration(REPO / "config" / "so101_ik_calibration.json")
    robot_map = json.loads((REPO / "config" / "robot_map.json").read_text())
    plan = write_plan.build_plan(write_plan.parse_args(
        ["--text", "i", "--distance_m", "0.20", "--out", "unused", "--preview", ""]), log=lambda *a: None)[0]
    plan["waypoints"] = plan["waypoints"][:n_waypoints]
    ticks, clamp, _ = write_arm.plan_to_ticks(plan, cal)
    targets, owner = write_arm.expand_schedule(ticks, 50.0, 360.0)
    return plan, cal, ticks, targets, owner, robot_map


def _sim_bus(ticks0, robot_map, **overrides):
    from neural_reflex.leader_arm import JOINTS as ALL_JOINTS
    from tests.helpers import SimBus
    start = {j: int(ticks0[i]) for i, j in enumerate(write_arm.JOINTS)}
    start["gripper"] = 1776
    start.update(overrides)
    return SimBus(start, {j: tuple(robot_map["limits"][j]) for j in ALL_JOINTS})


class KeepAliveAtPromptTest(unittest.TestCase):
    """MAJOR 2: torque must survive > 20 s at the MOVE prompt (ArmController's inactivity timeout is 15 s)."""

    def test_arm_stays_enabled_while_prompt_waits(self):
        import time
        from neural_reflex.arm_web import ArmController
        plan, cal, ticks, targets, owner, robot_map = _sim_setup()
        bus = _sim_bus(ticks[0], robot_map)
        controller = ArmController(bus, timeout=15)
        seen = []

        def slow_ask(_prompt):
            t0 = time.monotonic()
            while time.monotonic() - t0 < 21.0:
                time.sleep(1.0)
                snap = controller.status()
                seen.append((round(time.monotonic() - t0, 1), snap.get("fault"),
                             all(j["torque_enabled"] for j in snap["joints"] if j["name"] in write_arm.JOINTS)))
            return "MOVE"

        buf = io.StringIO()
        log = write_arm.RunLog(None)
        try:
            ok = write_arm.drive(controller, targets, owner, plan, speed="fast", rate_hz=50.0,
                                 start_pause_s=0.0, log=log, ask=slow_ask, out=buf)
        finally:
            controller.close()
        self.assertTrue(ok, buf.getvalue())
        self.assertGreaterEqual(seen[-1][0], 20.0)
        self.assertTrue(all(enabled for _, _, enabled in seen), seen)
        self.assertTrue(all(fault is None for _, fault, _ in seen), seen)
        self.assertEqual(bus.values["gripper"]["Torque_Enable"], 0)

    def test_refuses_schedule_when_a_joint_is_not_enabled(self):
        from neural_reflex.arm_web import ArmController
        plan, cal, ticks, targets, owner, robot_map = _sim_setup()
        bus = _sim_bus(ticks[0], robot_map)
        controller = ArmController(bus, timeout=15)
        try:
            for j in write_arm.JOINTS[:-1]:
                controller.torque(j, True, controller.motion_epoch)
            with self.assertRaisesRegex(RuntimeError, "refusing to start.*wrist_roll"):
                write_arm.require_enabled(controller)
            controller.torque("wrist_roll", True, controller.motion_epoch)
            self.assertEqual(set(write_arm.require_enabled(controller)) >= set(write_arm.JOINTS), True)
        finally:
            controller.close()

    def test_torque_lost_at_prompt_is_reported(self):
        from neural_reflex.arm_web import ArmController
        plan, cal, ticks, targets, owner, robot_map = _sim_setup()
        bus = _sim_bus(ticks[0], robot_map)
        controller = ArmController(bus, timeout=15)

        def pull_the_plug(_prompt):
            import time
            bus.values["elbow_flex"]["Torque_Enable"] = 0        # torque changed outside the session
            time.sleep(1.2)
            return "MOVE"

        try:
            with self.assertRaisesRegex(RuntimeError, "torque|Motor fault|prompt"):
                write_arm.drive(controller, targets, owner, plan, speed="fast", rate_hz=50.0,
                                start_pause_s=0.0, log=write_arm.RunLog(None), ask=pull_the_plug, out=io.StringIO())
        finally:
            controller.close()


class ArmCheckTest(unittest.TestCase):
    """BLOCKER 1 / MINOR 4: the read-only pre-flight check and its refusals."""

    @classmethod
    def setUpClass(cls):
        cls.plan, cls.cal, cls.ticks, cls.targets, cls.owner, cls.robot_map = _sim_setup()
        cls.home = write_arm.load_home(REPO / "config" / "arm_home.json")
        cls.home_ticks = [cls.home[j] for j in write_arm.JOINTS]

    def states(self, **overrides):
        bus = _sim_bus(self.home_ticks, self.robot_map, **overrides)
        states = write_arm.read_arm_ticks(bus)
        self.assertEqual(bus.goals, [])                                     # read-only
        self.assertTrue(all(v["Torque_Enable"] == 0 for v in bus.values.values()))
        return states

    def check(self, states, **kw):
        buf = io.StringIO()
        problems = write_arm.arm_check(states, self.plan, self.cal, self.home, self.targets[0],
                                       fine_cap=60.0, out=buf, **kw)
        return problems, buf.getvalue()

    def test_report_contents_at_home(self):
        problems, text = self.check(self.states())
        self.assertIn("ARM CHECK (read-only", text)
        for j in write_arm.JOINTS:
            self.assertIn(j, text)
            self.assertIn(self.cal["joints"][j]["how_to_verify"][:40], text)
        self.assertIn("FK pen tip", text)
        self.assertIn("tool z-axis in base frame (-0.038, +0.000, +0.999)", text)   # straight up
        self.assertIn("within 150 ticks of straight up", text)
        self.assertIn("simulated approach to the first hover point at 'fine' (60 ticks/s)", text)
        self.assertRegex(text, r"shoulder_pan\s+2042\s+off\s+0\.000")
        # only the unverified calibration blocks a start from home
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("calibration not verified", problems[0])
        for j in write_arm.JOINTS:
            self.assertIn(j, problems[0])
        self.assertEqual(self.check(self.states(), i_verified_calibration=True)[0], [])

    def test_refuses_start_away_from_home(self):
        far = self.states(elbow_flex=self.home["elbow_flex"] + 151)
        problems, _ = self.check(far, i_verified_calibration=True)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("not straight up", problems[0])
        self.assertIn("elbow_flex +151 ticks", problems[0])
        problems, text = self.check(far, i_verified_calibration=True, allow_any_start=True)
        self.assertEqual(problems, [])
        self.assertIn("WARNING (--allow_any_start)", text)
        near = self.states(shoulder_lift=self.home["shoulder_lift"] + 149, shoulder_pan=self.home["shoulder_pan"] + 400)
        self.assertEqual(self.check(near, i_verified_calibration=True)[0], [])       # pan is not weight-bearing

    def test_refuses_torque_already_on(self):
        states = self.states()
        states["wrist_flex"]["torque_enabled"] = True
        problems, _ = self.check(states, i_verified_calibration=True)
        self.assertTrue(any("torque is already on for wrist_flex" in p for p in problems), problems)

    def test_refuses_approach_through_the_plane(self):
        plan = json.loads(json.dumps(self.plan))
        plan["params"]["plane_z_m"] = 0.3                 # the first hover point is 1 cm above z = 0
        buf = io.StringIO()
        problems = write_arm.arm_check(self.states(), plan, self.cal, self.home, self.targets[0],
                                       fine_cap=60.0, out=buf, i_verified_calibration=True)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("below the writing plane", problems[0])

    def test_simulate_approach_from_home(self):
        from so101_ik.fk import build_so101_model
        model = build_so101_model(tuple(self.plan["params"]["pen_tip_offset_m"]))
        z_min, n = write_arm.simulate_approach(self.home_ticks, self.targets[0], self.cal, model, 60.0)
        worst = max(abs(int(t) - h) for t, h in zip(self.targets[0], self.home_ticks))
        self.assertEqual(n, int(np.ceil(worst / 1.2)))
        self.assertGreater(z_min, 0.0)
        self.assertLess(z_min, 0.02)                      # ends at the 1 cm hover

    def test_cli_check_and_execute_refusal(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.json"
            path.write_text(json.dumps(self.plan))
            make_bus = lambda port: _sim_bus(self.home_ticks, self.robot_map)
            with mock.patch("neural_reflex.leader_arm.make_bus", make_bus), \
                    mock.patch("builtins.input", side_effect=AssertionError("must refuse before the MOVE prompt")):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    code = write_arm.main(["--plan", str(path), "--check", "--port", "sim"])
                self.assertEqual(code, 2, buf.getvalue())
                self.assertIn("ARM CHECK", buf.getvalue())
                self.assertIn("CHECK FAILED", buf.getvalue())
                self.assertIn("calibration not verified", buf.getvalue())
                buf = io.StringIO()
                with redirect_stdout(buf):
                    code = write_arm.main(["--plan", str(path), "--check", "--port", "sim", "--i_verified_calibration"])
                self.assertEqual(code, 0, buf.getvalue())
                self.assertIn("CHECK PASSED", buf.getvalue())
                self.assertIn("Nothing moved", buf.getvalue())
                buf = io.StringIO()
                with redirect_stdout(buf):
                    code = write_arm.main(["--plan", str(path), "--execute", "--port", "sim"])
                self.assertEqual(code, 2, buf.getvalue())
                self.assertIn("REFUSED to execute", buf.getvalue())
                far = lambda port: _sim_bus(self.home_ticks, self.robot_map, wrist_flex=self.home["wrist_flex"] - 300)
            with mock.patch("neural_reflex.leader_arm.make_bus", far), \
                    mock.patch("builtins.input", side_effect=AssertionError("must refuse before the MOVE prompt")):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    code = write_arm.main(["--plan", str(path), "--execute", "--port", "sim", "--i_verified_calibration"])
                self.assertEqual(code, 2, buf.getvalue())
                self.assertIn("not straight up", buf.getvalue())
                self.assertIn("wrist_flex -300 ticks", buf.getvalue())


class ElbowCapAndSpeedCapTest(unittest.TestCase):
    """MAJOR 3 / MINOR 5."""

    def setUp(self):
        self.cal = write_arm.load_calibration(REPO / "config" / "so101_ik_calibration.json")

    def test_elbow_cap_from_air_arm_config(self):
        air = write_arm.load_air_config(REPO / "config" / "air_arm.json")
        self.assertEqual(air["elbow_max_deg"], 148)
        cap = write_arm.elbow_cap_ticks(self.cal, air["elbow_max_deg"])
        self.assertEqual(cap, int(1074 + 148 * 4096 / 360))                # 2757
        self.assertEqual(write_arm.tick_limits(self.cal)["elbow_flex"][1], cap)
        self.assertEqual(write_arm.tick_limits(self.cal, None)["elbow_flex"][1], cap)
        self.assertEqual(write_arm.tick_limits(self.cal, 200)["elbow_flex"][1], 3095 - 60)   # motor limit wins
        # bend of exactly elbow_max_deg maps to the cap tick, through the calibration sign
        q_cap = self.cal["joints"]["elbow_flex"]["q_home_rad"] + self.cal["joints"]["elbow_flex"]["sign"] * np.radians(148)
        self.assertAlmostEqual(float(write_arm.elbow_bend_deg(q_cap, self.cal)), 148.0, places=9)
        self.assertAlmostEqual(write_arm.q_to_ticks([0, 0, q_cap, 0, 0], self.cal)[2], 1074 + 148 * 4096 / 360, places=6)

    def test_plan_beyond_elbow_stop_is_refused(self):
        c = self.cal["joints"]["elbow_flex"]
        q_elbow = c["q_home_rad"] + c["sign"] * np.radians(160)          # 160 deg bend: inside the motor range
        self.assertLess(write_arm.q_to_ticks([0, 0, q_elbow, 0, 0], self.cal)[2], 3095 - 60)
        plan = {"text": "x", "waypoints": [{"t": 0, "xyz": [0, 0, 0], "pen_down": False,
                                            "q_rad": [0, 0, float(q_elbow), 0, 0]}]}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.json"
            path.write_text(json.dumps(plan))
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = write_arm.main(["--plan", str(path)])
        self.assertEqual(code, 2)
        self.assertIn("REFUSED", buf.getvalue())
        self.assertIn("elbow_flex", buf.getvalue())
        self.assertIn("peak elbow bend 160.0 deg", buf.getvalue())
        stop = write_plan.ElbowStop(cal=self.cal)
        self.assertFalse(stop.ok(plan["waypoints"][0]["q_rad"]))
        self.assertTrue(stop.ok([0, 0, c["q_home_rad"] + c["sign"] * np.radians(147), 0, 0]))

    def test_speed_caps_above_hard_cap_are_rejected(self):
        cal = json.loads((REPO / "config" / "so101_ik_calibration.json").read_text())
        cal["speed_caps_ticks_per_s"]["fast"] = write_arm.HARD_CAP_TICKS_PER_S + 1
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cal.json"
            path.write_text(json.dumps(cal))
            with self.assertRaisesRegex(ValueError, "hard cap"):
                write_arm.load_calibration(path)
            cal["speed_caps_ticks_per_s"]["fast"] = write_arm.HARD_CAP_TICKS_PER_S
            path.write_text(json.dumps(cal))
            write_arm.load_calibration(path)


if __name__ == "__main__":
    unittest.main()
