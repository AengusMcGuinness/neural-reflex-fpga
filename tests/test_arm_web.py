"""Simulated web arm tests. No physical servo is accessed."""

import http.client
from http.server import ThreadingHTTPServer
import json
import threading
import time
import unittest

from neural_reflex.arm_web import ArmController, make_handler
from neural_reflex.leader_arm import JOINTS


class SimBus:
    def __init__(self):
        self.values = {
            name: {"Present_Position": 2048, "Min_Position_Limit": 1000,
                   "Max_Position_Limit": 3000, "Torque_Enable": 0,
                   "Operating_Mode": 0, "Present_Voltage": 120,
                   "Present_Temperature": 30, "Torque_Limit": 1000,
                   "Max_Torque_Limit": 1000}
            for name in JOINTS
        }
        self.writes = []
        self.overloaded = set()
        self.stalled = set()
        self.torque_off_attempts = []

    def connect(self):
        pass

    def disconnect(self, *, disable_torque):
        pass

    def read(self, register, name, *, normalize):
        if name in self.overloaded:
            raise RuntimeError("[ServoStatus] Overload error!")
        return self.values[name][register]

    def write(self, register, name, value, *, normalize):
        self.writes.append((register, name, value))
        if (register == "Goal_Position" and name not in self.stalled
                and self.values[name]["Torque_Enable"]):
            self.values[name]["Present_Position"] = value

    def enable_torque(self, name):
        self.values[name]["Torque_Enable"] = 1

    def disable_torque(self, name):
        self.torque_off_attempts.append(name)
        self.values[name]["Torque_Enable"] = 0
        if name in self.overloaded:
            raise RuntimeError("[ServoStatus] Overload error!")


class ArmWebTests(unittest.TestCase):
    def setUp(self):
        self.bus = SimBus()
        self.controller = ArmController(self.bus, timeout=15)

    def tearDown(self):
        self.controller.close()

    def enable(self, name="shoulder_pan"):
        state = self.controller.torque(name, True, self.controller.motion_epoch)
        self.assertIsNone(state["fault"])
        return state["motion_epoch"]

    def keep_alive_for_simulated_time(self):
        self.controller.lease_deadline = self.controller.last_step + 100
        self.controller.deadline = self.controller.last_step + 100

    def steps(self, count, dt=0.02):
        for _ in range(count):
            self.controller.motion_step(now=self.controller.last_step + dt)

    def goals(self, name):
        return [value for register, motor, value in self.bus.writes
                if register == "Goal_Position" and motor == name]

    def test_target_is_queued_then_respects_speed_and_acceleration(self):
        epoch = self.enable()
        self.controller.set_speed("fine")
        before = len(self.goals("shoulder_pan"))
        state = self.controller.target("shoulder_pan", 2300, epoch)
        self.assertEqual(state["joints"][0]["target"], 2300)
        self.assertEqual(len(self.goals("shoulder_pan")), before)
        self.assertEqual(self.bus.values["shoulder_pan"]["Present_Position"], 2048)
        self.keep_alive_for_simulated_time()
        previous_velocity = 0.0
        for _ in range(60):
            self.steps(1)
            velocity = self.controller.velocities["shoulder_pan"]
            self.assertLessEqual(abs(velocity), 60.001)
            self.assertLessEqual(abs(velocity - previous_velocity), 720 * 0.02 + 0.01)
            previous_velocity = velocity
        writes = self.goals("shoulder_pan")[before:]
        self.assertTrue(writes)
        self.assertGreater(writes[-1], 2048)
        self.assertTrue(all(0 < b - a <= 20 for a, b in zip([2048] + writes, writes)))

    def test_startup_with_enabled_motor_serves_fault_and_turns_every_motor_off(self):
        bus = SimBus()
        bus.values["elbow_flex"]["Torque_Enable"] = 1
        controller = ArmController(bus)
        try:
            state = controller.status()
            self.assertIsNotNone(state["fault"])
            self.assertFalse(state["armed"])
            self.assertEqual(bus.torque_off_attempts[-6:], list(JOINTS))
            self.assertTrue(all(not v["Torque_Enable"] for v in bus.values.values()))
            self.assertIsNone(controller.recover()["fault"])
        finally:
            controller.close()

    def test_latest_target_replaces_older_target(self):
        epoch = self.enable()
        self.controller.target("shoulder_pan", 2300, epoch)
        self.controller.target("shoulder_pan", 1900, epoch)
        self.assertEqual(self.controller.targets["shoulder_pan"], 1900)
        self.keep_alive_for_simulated_time()
        self.steps(40)
        self.assertLess(self.goals("shoulder_pan")[-1], 2048)
        self.assertTrue(all(goal <= 2048 for goal in self.goals("shoulder_pan")))

    def test_fast_speed_reaches_higher_rate_without_exceeding_step_limit(self):
        epoch = self.enable()
        state = self.controller.set_speed("fast")
        self.assertEqual(state["max_speed_ticks_per_second"], 360)
        self.controller.target("shoulder_pan", 3000, epoch)
        self.keep_alive_for_simulated_time()
        self.steps(60)
        self.assertGreater(self.controller.velocities["shoulder_pan"], 180)
        self.assertLessEqual(self.controller.velocities["shoulder_pan"], 360)
        writes = self.goals("shoulder_pan")
        self.assertTrue(all(0 < b - a <= 20 for a, b in zip(writes, writes[1:])))

    def test_reversing_a_moving_slider_does_not_continue_in_old_direction(self):
        epoch = self.enable()
        self.controller.set_speed("fast")
        self.controller.target("shoulder_pan", 2900, epoch)
        self.keep_alive_for_simulated_time()
        self.steps(40)
        before = self.controller.goals["shoulder_pan"]
        self.assertGreater(self.controller.velocities["shoulder_pan"], 0)
        self.controller.target("shoulder_pan", 1800, epoch)
        self.keep_alive_for_simulated_time()
        self.steps(10)
        self.assertLess(self.controller.goals["shoulder_pan"], before)
        self.assertLessEqual(self.controller.velocities["shoulder_pan"], 0)

    def test_pending_stop_cannot_be_cleared_by_queued_enable(self):
        epoch = self.controller.motion_epoch
        with self.controller.lock:
            stopper = threading.Thread(target=self.controller.stop)
            stopper.start()
            self.assertTrue(self.controller.stop_pending.wait(timeout=1))
            with self.assertRaises(RuntimeError):
                self.controller.torque("shoulder_pan", True, epoch)
            self.assertEqual(self.bus.writes, [])
        stopper.join(timeout=2)
        self.assertFalse(stopper.is_alive())
        self.assertTrue(all(not v["Torque_Enable"] for v in self.bus.values.values()))

    def test_every_joint_has_full_configured_target_range(self):
        for name in JOINTS:
            self.enable(name)
        epoch = self.controller.motion_epoch
        state = self.controller.status()
        self.assertIsNone(state["max_travel"])
        for joint in state["joints"]:
            self.assertEqual((joint["target_minimum"], joint["target_maximum"]),
                             (1000, 3000))
            self.controller.target(joint["name"], 2088, epoch)
        with self.assertRaises(ValueError):
            self.controller.target("wrist_roll", 999, epoch)
        with self.assertRaises(ValueError):
            self.controller.target("wrist_roll", 3001, epoch)
        self.keep_alive_for_simulated_time()
        self.steps(50)
        self.assertTrue(all(self.goals(name)[-1] > 2048 for name in JOINTS))

    def test_stop_invalidates_epoch_and_blocks_queued_motion(self):
        epoch = self.enable()
        self.controller.target("shoulder_pan", 2300, epoch)
        state = self.controller.stop()
        self.assertFalse(state["armed"])
        self.assertGreater(state["motion_epoch"], epoch)
        before = len(self.goals("shoulder_pan"))
        self.steps(10)
        self.assertEqual(len(self.goals("shoulder_pan")), before)
        for command in (lambda: self.controller.target("shoulder_pan", 2200, epoch),
                        lambda: self.controller.jog("shoulder_pan", 10, epoch),
                        lambda: self.controller.heartbeat(epoch),
                        lambda: self.controller.pause(epoch),
                        lambda: self.controller.torque("shoulder_pan", True, epoch)):
            with self.assertRaises(ValueError):
                command()
        self.assertTrue(all(not v["Torque_Enable"] for v in self.bus.values.values()))

    def test_pause_preserves_torque_and_cancels_pending_target(self):
        epoch = self.enable()
        self.controller.target("shoulder_pan", 2300, epoch)
        self.keep_alive_for_simulated_time()
        self.steps(20)
        state = self.controller.pause(epoch)
        self.assertTrue(state["armed"])
        self.assertGreater(state["motion_epoch"], epoch)
        commanded = self.controller.goals["shoulder_pan"]
        self.assertEqual(self.controller.targets["shoulder_pan"], commanded)
        before = len(self.goals("shoulder_pan"))
        self.steps(20)
        self.assertEqual(len(self.goals("shoulder_pan")), before)
        self.assertEqual(self.bus.values["shoulder_pan"]["Torque_Enable"], 1)
        with self.assertRaises(ValueError):
            self.controller.target("shoulder_pan", 2200, epoch)

    def test_motion_lease_expiry_stops_motors(self):
        epoch = self.enable()
        self.controller.target("shoulder_pan", 2300, epoch)
        lease_deadline = self.controller.lease_deadline
        self.assertIsNotNone(lease_deadline)
        self.controller.motion_step(now=lease_deadline + 0.01)
        self.assertTrue(all(not v["Torque_Enable"] for v in self.bus.values.values()))
        self.assertFalse(self.controller.status()["armed"])

    def test_heartbeat_renews_live_motion_lease(self):
        epoch = self.enable()
        self.controller.target("shoulder_pan", 2300, epoch)
        self.controller.lease_deadline = time.monotonic() + 0.1
        prior_deadline = self.controller.lease_deadline
        self.controller.heartbeat(epoch)
        self.assertGreater(self.controller.lease_deadline, prior_deadline + 0.5)
        self.assertEqual(self.bus.values["shoulder_pan"]["Torque_Enable"], 1)

    def test_stalled_tracking_and_overload_disable_every_joint(self):
        epoch = self.enable()
        self.bus.stalled.add("shoulder_pan")
        self.controller.target("shoulder_pan", 2600, epoch)
        self.keep_alive_for_simulated_time()
        self.steps(100)
        self.assertIsNotNone(self.controller.fault)
        self.assertEqual(self.bus.torque_off_attempts[-6:], list(JOINTS))
        self.assertTrue(all(not v["Torque_Enable"] for v in self.bus.values.values()))

        self.bus.stalled.clear()
        self.assertIsNone(self.controller.recover()["fault"])
        self.enable()
        self.bus.overloaded.update(("shoulder_lift", "elbow_flex"))
        fault = self.controller.status()
        self.assertIn("Overload", fault["fault"])
        self.assertTrue(fault["torque_unknown"])
        self.assertEqual(self.bus.torque_off_attempts[-6:], list(JOINTS))
        self.bus.overloaded.clear()
        self.assertIsNone(self.controller.recover()["fault"])

    def test_http_page_is_read_only_and_post_requires_local_token(self):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), Placeholder)
        except PermissionError:
            self.skipTest("sandbox does not permit loopback socket binding")
        port = server.server_port
        server.RequestHandlerClass = make_handler(self.controller, "test-token", port)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            def request(method, path, payload=None, headers=None):
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                connection.request(method, path, json.dumps(payload or {}) if method == "POST" else None,
                                   headers or {})
                response = connection.getresponse()
                result = response.status, response.read()
                connection.close()
                return result

            code, page = request("GET", "/")
            self.assertEqual(code, 200)
            self.assertIn(b"test-token", page)
            self.assertEqual(self.bus.writes, [])
            payload = {"joint": "shoulder_pan", "enabled": True}
            self.assertEqual(request("POST", "/api/torque", payload)[0], 403)
            self.assertEqual(request("POST", "/api/torque", payload,
                                     {"X-Arm-Token": "test-token"})[0], 403)
            self.assertEqual(self.bus.writes, [])
            headers = {"X-Arm-Token": "test-token", "Origin": f"http://127.0.0.1:{port}"}
            payload["motion_epoch"] = self.controller.motion_epoch
            self.assertEqual(request("POST", "/api/torque", payload, headers)[0], 200)
            move = {"joint": "shoulder_pan", "target": 2300,
                    "motion_epoch": self.controller.motion_epoch}
            code, body = request("POST", "/api/target", move, headers)
            self.assertEqual(code, 200)
            self.assertTrue(json.loads(body)["moving"])
            self.assertEqual(self.bus.values["shoulder_pan"]["Present_Position"], 2048)
            self.assertEqual(request("POST", "/api/stop", {}, headers)[0], 200)
            self.assertEqual(request("POST", "/api/target", move, headers)[0], 400)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class Placeholder:
    """Replaced before serving to select an ephemeral port."""


if __name__ == "__main__":
    unittest.main()
