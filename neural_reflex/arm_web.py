"""Loopback-only, bounded web controls for the connected SO-ARM101 leader."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time

from .leader_arm import JOINTS, JogSession, disable_all_torque, inspect


WEB_DIR = Path(__file__).resolve().parent.parent / "web" / "arm"


class ArmController:
    """Serialize motor commands, bound each joint, and stop on faults."""

    def __init__(self, bus, *, max_step: int = 20, max_travel: int = 160,
                 timeout: float = 15.0):
        self.bus = bus
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.max_step = max_step
        self.max_travel = max_travel
        self.timeout = timeout
        self.deadline: float | None = None
        self.notice = "Motors are off. Enable a joint to start."
        self.control_mode = "manual"
        self.fault: str | None = None
        self.torque_unknown = False
        self.enabled: set[str] = set()
        self.starts: dict[str, int] = {}
        self.goals: dict[str, int] = {}
        self.telemetry: dict[str, dict] = {}
        self.telemetry_due = 0.0
        self.bus.connect()
        try:
            self.last_states = inspect(bus)
            JogSession(bus, self.last_states, max_step, max_travel)
        except BaseException:
            disable_all_torque(self.bus)
            self.bus.disconnect(disable_torque=False)
            raise

    def _stop_locked(self, notice: str) -> dict[str, str]:
        errors = disable_all_torque(self.bus)
        self.deadline = None
        self.control_mode = "manual"
        self.torque_unknown = bool(errors)
        if errors:
            self.notice = "Could not verify torque off; disconnect 12 V power."
        else:
            self.enabled.clear()
            self.starts.clear()
            self.goals.clear()
            self.notice = notice
        return errors

    def _fault_locked(self, error: Exception | str) -> None:
        reason = str(error).splitlines()[0]
        self.cancel.set()
        errors = self._stop_locked("Motors are off after a motor fault.")
        self.fault = f"Motor fault: {reason}"
        if errors:
            self.fault += (" Torque-off could not be verified for "
                           + ", ".join(errors) + "; disconnect 12 V power.")
        else:
            self.fault += " Motors are off. Clear the fault before retrying."

    def _fault_status_locked(self) -> dict:
        return {
            "fault": self.fault,
            "stale": True,
            "armed": False,
            "control_mode": self.control_mode,
            "torque_unknown": self.torque_unknown,
            "remaining_seconds": 0,
            "max_step": self.max_step,
            "max_travel": self.max_travel,
            "notice": self.fault,
            "joints": [self._joint_data(name, state, stale=True)
                       for name, state in self.last_states.items()],
        }

    def _joint_data(self, name, state, *, stale=False) -> dict:
        start = self.starts.get(name, state.position)
        lower = max(state.minimum, start - self.max_travel)
        upper = min(state.maximum, start + self.max_travel)
        extra = self.telemetry.get(name, {})
        return {"name": name, "position": state.position,
                "minimum": state.minimum, "maximum": state.maximum,
                "target": self.goals.get(name, state.position),
                "target_minimum": lower, "target_maximum": upper,
                "torque_enabled": None if stale else state.torque_enabled,
                "operating_mode": state.operating_mode,
                "voltage_v": extra.get("voltage_v"),
                "temperature_c": extra.get("temperature_c"),
                "torque_limit": extra.get("torque_limit"),
                "max_torque_limit": extra.get("max_torque_limit")}

    def _read_telemetry_locked(self) -> None:
        if time.monotonic() < self.telemetry_due:
            return
        telemetry = {}
        for name in JOINTS:
            telemetry[name] = {
                "voltage_v": round(int(self.bus.read("Present_Voltage", name,
                                                       normalize=False)) / 10, 1),
                "temperature_c": int(self.bus.read("Present_Temperature", name,
                                                        normalize=False)),
                "torque_limit": int(self.bus.read("Torque_Limit", name,
                                                   normalize=False)),
                "max_torque_limit": int(self.bus.read("Max_Torque_Limit", name,
                                                       normalize=False)),
            }
        self.telemetry = telemetry
        self.telemetry_due = time.monotonic() + 5

    def expire(self) -> None:
        with self.lock:
            if self.enabled and self.deadline is not None:
                if time.monotonic() >= self.deadline:
                    self.cancel.set()
                    errors = self._stop_locked("Motors turned off after inactivity.")
                    if errors:
                        self.fault = ("Could not verify torque off after inactivity: "
                                      + ", ".join(errors))

    def _status_locked(self) -> dict:
        if self.fault:
            return self._fault_status_locked()
        try:
            states = inspect(self.bus)
            for name, state in states.items():
                if state.torque_enabled != (name in self.enabled):
                    raise RuntimeError("Motor torque changed outside this control session.")
            self._read_telemetry_locked()
        except Exception as error:
            self._fault_locked(error)
            return self._fault_status_locked()
        self.last_states = states
        return {
            "fault": None,
            "stale": False,
            "armed": bool(self.enabled),
            "control_mode": self.control_mode,
            "torque_unknown": False,
            "remaining_seconds": (max(0, round(self.deadline - time.monotonic(), 1))
                                  if self.enabled and self.deadline else 0),
            "max_step": self.max_step,
            "max_travel": self.max_travel,
            "notice": self.notice,
            "joints": [self._joint_data(name, state)
                       for name, state in states.items()],
        }

    def status(self) -> dict:
        with self.lock:
            self.expire()
            return self._status_locked()

    def _enable_joint_locked(self, name: str) -> None:
        if name in self.enabled:
            return
        state = inspect(self.bus)[name]
        if state.torque_enabled or state.operating_mode != 0:
            raise RuntimeError(f"{name} must have torque off and position mode 0")
        if not state.minimum <= state.position <= state.maximum:
            raise RuntimeError(f"{name} is outside its configured limits")
        self.bus.write("Goal_Position", name, state.position, normalize=False)
        self.bus.enable_torque(name)
        if int(self.bus.read("Torque_Enable", name, normalize=False)) != 1:
            raise RuntimeError(f"{name} torque did not enable")
        self.enabled.add(name)
        self.starts[name] = state.position
        self.goals[name] = state.position
        self.deadline = time.monotonic() + self.timeout

    def torque(self, name: str, enabled: bool) -> dict:
        if name not in JOINTS or type(enabled) is not bool:
            raise ValueError("Choose a joint and true or false for torque.")
        with self.lock:
            if self.fault:
                return self._fault_status_locked()
            self.expire()
            if enabled:
                self.cancel.clear()
                self.control_mode = "position"
                try:
                    self._enable_joint_locked(name)
                except Exception as error:
                    self._fault_locked(error)
                    return self._fault_status_locked()
                self.notice = f"{name} is holding its measured position."
            else:
                try:
                    self.bus.disable_torque(name)
                    if int(self.bus.read("Torque_Enable", name, normalize=False)) != 0:
                        raise RuntimeError(f"{name} torque did not turn off")
                except Exception as error:
                    self._fault_locked(error)
                    return self._fault_status_locked()
                self.enabled.discard(name)
                self.starts.pop(name, None)
                self.goals.pop(name, None)
                if not self.enabled:
                    self.deadline = None
                self.notice = f"{name} torque is off."
            return self._status_locked()

    def _move_to_locked(self, name: str, target: int) -> dict:
        if name not in self.enabled:
            raise RuntimeError(f"Enable {name} torque first.")
        state = self.last_states[name]
        lower = max(state.minimum, self.starts[name] - self.max_travel)
        upper = min(state.maximum, self.starts[name] + self.max_travel)
        if not lower <= target <= upper:
            raise ValueError(f"{name} target must be {lower}..{upper} ticks")
        if self.goals[name] == target:
            return self._status_locked()
        while self.goals[name] != target:
            if self.cancel.is_set():
                self._stop_locked("Motors are off.")
                return self._status_locked()
            difference = target - self.goals[name]
            step = max(-self.max_step, min(self.max_step, difference))
            next_goal = self.goals[name] + step
            try:
                self.bus.write("Goal_Position", name, next_goal, normalize=False)
                self.goals[name] = next_goal
                time.sleep(0.2)
                measured = int(self.bus.read("Present_Position", name,
                                             normalize=False))
            except Exception as error:
                self._fault_locked(error)
                return self._fault_status_locked()
            if abs(next_goal - measured) > 60:
                self._fault_locked(f"{name} did not track its target")
                return self._fault_status_locked()
            self.deadline = time.monotonic() + self.timeout
        self.notice = f"{name}: target {target}, measured {measured}."
        return self._status_locked()

    def jog(self, name: str, delta: int) -> dict:
        if name not in JOINTS or type(delta) is not int or not 1 <= abs(delta) <= self.max_step:
            raise ValueError(f"Choose a joint and a step of 1..{self.max_step} ticks.")
        with self.lock:
            self.expire()
            if self.fault:
                return self._fault_status_locked()
            return self._move_to_locked(name, self.goals.get(name, 0) + delta)

    def target(self, name: str, target: int) -> dict:
        if name not in JOINTS or type(target) is not int:
            raise ValueError("Choose a joint and an integer target tick.")
        with self.lock:
            self.expire()
            if self.fault:
                return self._fault_status_locked()
            return self._move_to_locked(name, target)

    def stop(self) -> dict:
        self.cancel.set()
        with self.lock:
            errors = self._stop_locked("Motors are off.")
            self.control_mode = "manual"
            if errors:
                self.fault = ("Could not verify torque off for "
                              + ", ".join(errors) + "; disconnect 12 V power.")
            return self._status_locked()

    def set_mode(self, mode: str) -> dict:
        if mode not in ("manual", "position"):
            raise ValueError("Choose manual or position control mode.")
        if mode == "manual":
            return self.stop()
        with self.lock:
            if self.fault:
                return self._fault_status_locked()
            self.control_mode = "position"
            self.notice = "Position control selected; enable joints individually."
            return self._status_locked()

    def recover(self) -> dict:
        """Turn every joint off before clearing a fault after power cycling."""
        self.cancel.set()
        with self.lock:
            errors = self._stop_locked("Checking the arm after its fault.")
            if errors:
                self.fault = ("Still cannot turn off " + ", ".join(errors)
                              + "; disconnect 12 V power and check the arm.")
                return self._fault_status_locked()
            try:
                states = inspect(self.bus)
                JogSession(self.bus, states, self.max_step, self.max_travel)
            except Exception as error:
                self.fault = f"Motor fault persists: {str(error).splitlines()[0]}"
                return self._fault_status_locked()
            self.last_states = states
            self.fault = None
            self.control_mode = "manual"
            self.cancel.clear()
            self.notice = "Arm reconnected; motors are off."
            return self._status_locked()

    def close(self) -> None:
        self.cancel.set()
        with self.lock:
            try:
                self._stop_locked("Motors are off.")
            finally:
                self.bus.disconnect(disable_torque=False)


def make_handler(controller: ArmController, token: str, http_port: int):
    expected_host = f"127.0.0.1:{http_port}"
    expected_origin = f"http://{expected_host}"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; script-src 'self'; style-src 'self'; "
                             "connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, data: dict) -> None:
            self._send(code, json.dumps(data).encode(), "application/json; charset=utf-8")

        def _host_ok(self) -> bool:
            if self.headers.get("Host") != expected_host:
                self._json(403, {"error": "Open the loopback URL printed by the server."})
                return False
            return True

        def do_GET(self):
            if not self._host_ok():
                return
            try:
                if self.path == "/api/status":
                    self._json(200, controller.status())
                elif self.path == "/":
                    page = (WEB_DIR / "index.html").read_text()
                    page = page.replace("__ARM_TOKEN__", token)
                    page = page.replace("__ARM_TIMEOUT__", f"{controller.timeout:g}")
                    self._send(200, page.encode(), "text/html; charset=utf-8")
                elif self.path in ("/arm.js", "/arm.css"):
                    content_type = ("text/javascript" if self.path.endswith(".js")
                                    else "text/css")
                    self._send(200, (WEB_DIR / self.path[1:]).read_bytes(),
                               content_type + "; charset=utf-8")
                else:
                    self._json(404, {"error": "Not found"})
            except Exception as error:
                self._json(500, {"error": str(error)})

        def do_POST(self):
            if not self._host_ok():
                return
            if self.headers.get("Origin") != expected_origin:
                self._json(403, {"error": "Open the page from the local server."})
                return
            if not secrets.compare_digest(self.headers.get("X-Arm-Token", ""), token):
                self._json(403, {"error": "Invalid control token."})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
                if self.path == "/api/jog":
                    result = controller.jog(data.get("joint"), data.get("delta"))
                elif self.path == "/api/target":
                    result = controller.target(data.get("joint"), data.get("target"))
                elif self.path == "/api/torque":
                    result = controller.torque(data.get("joint"), data.get("enabled"))
                elif self.path == "/api/mode":
                    result = controller.set_mode(data.get("mode"))
                elif self.path == "/api/stop":
                    result = controller.stop()
                elif self.path == "/api/retry":
                    result = controller.recover()
                else:
                    self._json(404, {"error": "Not found"})
                    return
                self._json(200, result)
            except (ValueError, RuntimeError) as error:
                self._json(400, {"error": str(error)})
            except Exception as error:
                try:
                    controller.stop()
                finally:
                    self._json(500, {"error": str(error)})

    return Handler


def serve(bus, *, http_port: int = 8767, timeout: float = 15.0) -> None:
    if not 1 <= http_port <= 65535:
        raise ValueError("HTTP port must be 1..65535")
    if not 1 <= timeout <= 60:
        raise ValueError("inactivity timeout must be 1..60 seconds")
    controller = ArmController(bus, timeout=timeout)
    token = secrets.token_urlsafe(24)
    server = None
    stop_watchdog = threading.Event()

    def watchdog():
        while not stop_watchdog.wait(0.25):
            try:
                controller.expire()
            except Exception as error:
                controller.notice = f"Could not turn off motors: {error}"

    try:
        server = ThreadingHTTPServer(("127.0.0.1", http_port),
                                     make_handler(controller, token, http_port))
        thread = threading.Thread(target=watchdog, daemon=True)
        thread.start()
        print(f"Arm controls: http://127.0.0.1:{http_port}/", flush=True)
        print("Motors are off. Enable joints individually in the page. Ctrl+C stops.",
              flush=True)
        try:
            server.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            pass
    finally:
        stop_watchdog.set()
        if server is not None:
            server.server_close()
        controller.close()
