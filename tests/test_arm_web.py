"""Exercise the local arm HTTP controls without moving physical hardware."""

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
        self.connected = False
        self.overloaded = set()
        self.torque_off_attempts = []

    def connect(self):
        self.connected = True

    def disconnect(self, *, disable_torque):
        if disable_torque:
            for name in JOINTS:
                self.disable_torque(name)
        self.connected = False

    def read(self, register, name, *, normalize):
        if name in self.overloaded:
            raise RuntimeError("[ServoStatus] Overload error!")
        return self.values[name][register]

    def write(self, register, name, value, *, normalize):
        self.writes.append((register, name, value))
        if register == "Goal_Position" and self.values[name]["Torque_Enable"]:
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
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), BaseHandlerPlaceholder
        )
        port = self.server.server_port
        self.server.RequestHandlerClass = make_handler(self.controller, "test-token", port)
        self.port = port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.controller.close()

    def request(self, method, path, payload=None, *, token=True, origin=True):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        headers = {}
        if method == "POST":
            headers["Content-Type"] = "application/json"
            if token:
                headers["X-Arm-Token"] = "test-token"
            if origin:
                headers["Origin"] = f"http://127.0.0.1:{self.port}"
        connection.request(method, path, json.dumps(payload or {}) if method == "POST" else None,
                           headers)
        response = connection.getresponse()
        body = response.read()
        result = (response.status, response.getheaders(), body)
        connection.close()
        return result

    def test_page_is_read_only_until_enable_and_unauthorized_post_is_rejected(self):
        status, headers, page = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"test-token", page)
        self.assertIn("Content-Security-Policy", dict(headers))
        self.assertEqual(self.bus.writes, [])
        status, _, state = self.request("GET", "/api/status")
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(state)["armed"])
        self.assertEqual(len(json.loads(state)["joints"]), 6)
        self.assertEqual(self.request("POST", "/api/torque",
                                      {"joint": "shoulder_pan", "enabled": True},
                                      token=False)[0], 403)
        self.assertEqual(self.request("POST", "/api/torque",
                                      {"joint": "shoulder_pan", "enabled": True},
                                      origin=False)[0], 403)
        self.assertEqual(self.bus.writes, [])

    def test_joint_motion_bounds_stop_and_timeout(self):
        for name in JOINTS:
            self.assertEqual(self.request("POST", "/api/torque",
                                          {"joint": name, "enabled": True})[0], 200)
        self.assertTrue(all(v["Torque_Enable"] for v in self.bus.values.values()))
        for name in JOINTS:
            status, _, body = self.request("POST", "/api/jog",
                                           {"joint": name, "delta": 20})
            self.assertEqual(status, 200)
            state = json.loads(body)
            self.assertEqual(next(j["position"] for j in state["joints"]
                                  if j["name"] == name), 2068)
        self.assertEqual(self.request("POST", "/api/jog",
                                      {"joint": "shoulder_pan", "delta": 21})[0], 400)
        for _ in range(7):
            self.assertEqual(self.request("POST", "/api/jog",
                                          {"joint": "shoulder_pan", "delta": 20})[0], 200)
        self.assertEqual(self.request("POST", "/api/jog",
                                      {"joint": "shoulder_pan", "delta": 20})[0], 400)
        self.assertEqual(self.request("POST", "/api/stop")[0], 200)
        self.assertTrue(all(not v["Torque_Enable"] for v in self.bus.values.values()))
        self.assertEqual(self.request("POST", "/api/torque",
                                      {"joint": "shoulder_pan", "enabled": True})[0], 200)
        self.controller.deadline = time.monotonic() - 1
        self.controller.expire()
        self.assertTrue(all(not v["Torque_Enable"] for v in self.bus.values.values()))

    def test_overload_becomes_visible_fault_and_retry_recovers_after_power_cycle(self):
        self.assertEqual(self.request("POST", "/api/torque",
                                      {"joint": "shoulder_pan", "enabled": True})[0], 200)
        self.bus.overloaded.update(("shoulder_lift", "elbow_flex"))
        status, _, body = self.request("GET", "/api/status")
        self.assertEqual(status, 200)
        fault = json.loads(body)
        self.assertIn("Overload", fault["fault"])
        self.assertTrue(fault["torque_unknown"])
        self.assertEqual(self.bus.torque_off_attempts[-6:], list(JOINTS))
        self.bus.overloaded.clear()
        status, _, body = self.request("POST", "/api/retry")
        self.assertEqual(status, 200)
        self.assertIsNone(json.loads(body)["fault"])
        self.assertTrue(all(not v["Torque_Enable"] for v in self.bus.values.values()))

    def test_individual_torque_and_bounded_target(self):
        status, _, body = self.request("POST", "/api/mode", {"mode": "position"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["control_mode"], "position")
        self.assertTrue(all(not v["Torque_Enable"] for v in self.bus.values.values()))
        status, _, body = self.request("POST", "/api/torque",
                                       {"joint": "wrist_roll", "enabled": True})
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["armed"])
        self.assertEqual([name for name in JOINTS
                          if self.bus.values[name]["Torque_Enable"]], ["wrist_roll"])
        status, _, body = self.request("POST", "/api/target",
                                       {"joint": "wrist_roll", "target": 2088})
        self.assertEqual(status, 200)
        wrist = next(j for j in json.loads(body)["joints"]
                     if j["name"] == "wrist_roll")
        self.assertEqual(wrist["position"], 2088)
        self.assertEqual(wrist["target_minimum"], 1888)
        self.assertEqual(wrist["target_maximum"], 2208)
        self.assertEqual(self.request("POST", "/api/target",
                                      {"joint": "wrist_roll", "target": 2209})[0], 400)
        status, _, body = self.request("POST", "/api/torque",
                                       {"joint": "wrist_roll", "enabled": False})
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["armed"])
        self.assertTrue(all(not v["Torque_Enable"] for v in self.bus.values.values()))
        status, _, body = self.request("POST", "/api/mode", {"mode": "manual"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["control_mode"], "manual")


class BaseHandlerPlaceholder:
    """Replaced before serving; lets the test select an ephemeral port."""


if __name__ == "__main__":
    unittest.main()
