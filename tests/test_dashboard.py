"""Dashboard tests with a simulated arm bus and canned Kinesis events. No hardware."""

import http.client
from http.server import ThreadingHTTPServer
import json
import threading
import time
import unittest

from neural_reflex.dashboard import ArmReader, BandRelay, Hub, make_handler
from neural_reflex.leader_arm import JOINTS


class ReadOnlyBus:
    def __init__(self):
        self.values = {name: {"Present_Position": 2000 + i, "Min_Position_Limit": 1000,
                              "Max_Position_Limit": 3000, "Torque_Enable": 0,
                              "Present_Voltage": 121}
                       for i, name in enumerate(JOINTS)}
        self.forbidden = []
        self.disconnected_with = None

    def connect(self):
        pass

    def disconnect(self, *, disable_torque):
        self.disconnected_with = disable_torque

    def read(self, register, name, *, normalize):
        return self.values[name][register]

    def write(self, *args, **kwargs):
        self.forbidden.append(("write", args))

    def enable_torque(self, name):
        self.forbidden.append(("enable_torque", name))

    def disable_torque(self, name):
        self.forbidden.append(("disable_torque", name))


def drain(q):
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


class ArmReaderTests(unittest.TestCase):
    def test_reads_positions_and_never_writes(self):
        bus, hub = ReadOnlyBus(), Hub()
        q = hub.subscribe()
        reader = ArmReader(lambda: bus, hub, rate_hz=100)
        thread = threading.Thread(target=reader.run)
        thread.start()
        time.sleep(0.15)
        bus.values["elbow_flex"]["Present_Position"] = 2500   # moved by hand
        time.sleep(0.1)
        reader.stop_event.set()
        thread.join(timeout=2)
        frames = [json.loads(data) for name, data in drain(q) if name == "arm"]
        self.assertGreater(len(frames), 5)
        self.assertTrue(frames[0]["connected"])
        self.assertEqual(frames[0]["positions"]["shoulder_pan"], 2000)
        self.assertEqual(frames[-1]["positions"]["elbow_flex"], 2500)
        self.assertEqual(frames[0]["voltage_v"], 12.1)
        self.assertEqual(bus.forbidden, [])
        self.assertIs(bus.disconnected_with, False)

    def test_degrees_are_relative_to_home(self):
        bus, hub = ReadOnlyBus(), Hub()
        q = hub.subscribe()
        home = {name: 2000 + i for i, name in enumerate(JOINTS)}
        home["elbow_flex"] -= 1024                       # quarter turn away from home
        reader = ArmReader(lambda: bus, hub, home=home)
        reader.run_once(bus, reader.read_static(bus), 0)
        frame = json.loads(drain(q)[0][1])
        self.assertEqual(frame["degrees_from_home"]["shoulder_pan"], 0)
        self.assertEqual(frame["degrees_from_home"]["elbow_flex"], 90)
        self.assertEqual(frame["home"], home)

    def test_load_home_validates_file(self):
        import tempfile
        from pathlib import Path
        from neural_reflex.dashboard import load_home
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arm_home.json"
            self.assertIsNone(load_home(path))
            good = {name: 2048 for name in JOINTS}
            path.write_text(json.dumps({"positions": good}))
            self.assertEqual(load_home(path), good)
            path.write_text(json.dumps({"positions": {**good, "gripper": 9999}}))
            with self.assertRaises(ValueError):
                load_home(path)

    def test_repo_home_file_is_valid(self):
        from neural_reflex.dashboard import load_home
        self.assertEqual(set(load_home()), set(JOINTS))

    def test_reports_missing_arm(self):
        hub = Hub()
        q = hub.subscribe()

        def missing():
            raise OSError("could not open port /dev/cu.nothing")

        reader = ArmReader(missing, hub)
        thread = threading.Thread(target=reader.run)
        thread.start()
        time.sleep(0.05)
        reader.stop_event.set()
        thread.join(timeout=3)
        frame = json.loads(drain(q)[0][1])
        self.assertFalse(frame["connected"])
        self.assertIn("could not open port", frame["error"])


class BandRelayTests(unittest.TestCase):
    def test_rejects_non_local_urls(self):
        relay = BandRelay(Hub())
        with self.assertRaises(ValueError):
            relay.set_url("http://example.com/events?token=x")
        relay.set_url("http://127.0.0.1:8766/events?token=abc")

    def test_republishes_band_events(self):
        hub = Hub()
        q = hub.subscribe()
        relay = BandRelay(hub, "http://127.0.0.1:8766/events?token=abc")
        emg = {"type": "emg", "payload": {"sequence": 4, "values": [[1] * 8] * 16}}
        lines = [": connected", "", "event: emg", "data: " + json.dumps(emg), "",
                 "event: mystery", 'data: {"type": "mystery"}', ""]
        relay.handle_lines(lines, relay.generation)
        events = drain(q)
        self.assertEqual([name for name, _ in events], ["emg"])
        self.assertEqual(json.loads(events[0][1])["payload"]["sequence"], 4)


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.hub = Hub()
        self.relay = BandRelay(self.hub)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.hub, self.relay, 0, True))
        self.port = self.server.server_address[1]
        self.server.RequestHandlerClass = make_handler(self.hub, self.relay, self.port, True)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, data

    def test_serves_page_and_rejects_other_hosts(self):
        status, page = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b'content="true"', page)
        status, _ = self.request("GET", "/", headers={"Host": "evil.example"})
        self.assertEqual(status, 403)

    def test_stream_url_requires_same_origin(self):
        body = json.dumps({"url": "http://127.0.0.1:8766/events?token=abc"})
        status, _ = self.request("POST", "/api/stream-url", body,
                                 {"Content-Type": "application/json", "Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        status, _ = self.request("POST", "/api/stream-url", body,
                                 {"Content-Type": "application/json",
                                  "Origin": f"http://127.0.0.1:{self.port}"})
        self.assertEqual(status, 200)
        self.assertEqual(self.relay.url, "http://127.0.0.1:8766/events?token=abc")

    def test_event_stream_delivers_latest_arm_state(self):
        self.hub.publish("arm", {"connected": True, "positions": {"gripper": 1776}}, sticky=True)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        conn.request("GET", "/events")
        response = conn.getresponse()
        self.assertEqual(response.getheader("Content-Type"), "text/event-stream; charset=utf-8")
        text = b""
        while b"1776" not in text:
            text += response.read1(4096)
        conn.close()
        self.assertIn(b"event: arm", text)


if __name__ == "__main__":
    unittest.main()
