"""Loopback-only, bounded web controls for the connected SO-ARM101 leader."""

from __future__ import annotations

from dataclasses import replace
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time

from .leader_arm import JOINTS, disable_all_torque, inspect, validate_motion_state


WEB_DIR = Path(__file__).resolve().parent.parent / "web" / "arm"


class ArmController:
    """Latest-target position control with a bounded, interruptible motion ramp."""

    SPEEDS = {"fine": 60, "normal": 180, "fast": 360}
    acceleration = 720.0
    motion_lease_seconds = 0.8
    tracking_tolerance = 60
    tracking_timeout = 0.5

    def __init__(self, bus, *, max_step: int = 20, max_travel: int | None = None,
                 timeout: float = 15.0):
        if not 1 <= max_step <= 20:
            raise ValueError("max_step must be 1..20 ticks")
        if max_travel is not None and max_travel < 1:
            raise ValueError("max_travel must be positive")
        self.bus = bus
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.stop_pending = threading.Event()
        self.max_step = max_step
        self.max_travel = max_travel
        self.timeout = timeout
        self.deadline: float | None = None
        self.lease_deadline: float | None = None
        self.last_step = time.monotonic()
        self.motion_epoch = 0
        self.speed = "normal"
        self.notice = "Motors are off. Enable a joint to start."
        self.control_mode = "manual"
        self.fault: str | None = None
        self.torque_unknown = False
        self.enabled: set[str] = set()
        self.starts: dict[str, int] = {}
        self.goals: dict[str, int] = {}
        self.targets: dict[str, int] = {}
        self.command_positions: dict[str, float] = {}
        self.velocities: dict[str, float] = {}
        self.tracking_since: dict[str, float] = {}
        self.telemetry: dict[str, dict] = {}
        self.telemetry_due = 0.0
        self.last_states = {}
        self.bus.connect()
        try:
            self.last_states = inspect(bus)
            validate_motion_state(self.last_states)
        except Exception as error:
            # Keep the page available for Stop and Retry after a latched servo fault.
            self._fault_locked(error)
        except BaseException:
            disable_all_torque(self.bus)
            self.bus.disconnect(disable_torque=False)
            raise

    def _moving_locked(self) -> bool:
        return any(self.goals[name] != self.targets[name] for name in self.enabled
                   if name in self.targets and name in self.goals)

    def _stop_locked(self, notice: str) -> dict[str, str]:
        self.motion_epoch += 1
        self.targets.clear()
        self.velocities.clear()
        self.command_positions.clear()
        self.tracking_since.clear()
        self.deadline = self.lease_deadline = None
        errors = disable_all_torque(self.bus)
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

    def _bounds_locked(self, name: str) -> tuple[int, int]:
        state = self.last_states[name]
        if self.max_travel is None:
            return state.minimum, state.maximum
        start = self.starts.get(name, state.position)
        return (max(state.minimum, start - self.max_travel),
                min(state.maximum, start + self.max_travel))

    def _joint_data(self, name, state, *, stale=False) -> dict:
        lower, upper = self._bounds_locked(name)
        extra = self.telemetry.get(name, {})
        target = self.targets.get(name, state.position)
        commanded = self.goals.get(name, state.position)
        return {"name": name, "position": state.position,
                "minimum": state.minimum, "maximum": state.maximum,
                "target": target, "commanded_position": commanded,
                "moving": not stale and name in self.enabled and target != commanded,
                "target_minimum": lower, "target_maximum": upper,
                "torque_enabled": None if stale else state.torque_enabled,
                "operating_mode": state.operating_mode,
                "voltage_v": extra.get("voltage_v"),
                "temperature_c": extra.get("temperature_c"),
                "torque_limit": extra.get("torque_limit"),
                "max_torque_limit": extra.get("max_torque_limit")}

    def _snapshot_locked(self) -> dict:
        stale = bool(self.fault)
        return {
            "fault": self.fault, "stale": stale,
            "armed": bool(self.enabled) and not stale,
            "control_mode": self.control_mode,
            "torque_unknown": self.torque_unknown,
            "motion_epoch": self.motion_epoch,
            "moving": not stale and self._moving_locked(),
            "speed": self.speed,
            "max_speed_ticks_per_second": self.SPEEDS[self.speed],
            "remaining_seconds": (max(0, round(self.deadline - time.monotonic(), 1))
                                  if self.enabled and self.deadline else 0),
            "max_step": self.max_step, "max_travel": self.max_travel,
            "notice": self.fault or self.notice,
            "joints": [self._joint_data(name, state, stale=stale)
                       for name, state in self.last_states.items()],
        }

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

    def expire(self, now: float | None = None) -> None:
        with self.lock:
            now = time.monotonic() if now is None else now
            if self.fault or not self.enabled:
                return
            notice = None
            if self._moving_locked() and (self.lease_deadline is None or
                                          now >= self.lease_deadline):
                notice = "Motors stopped: live control connection timed out."
            elif self.deadline is not None and now >= self.deadline:
                notice = "Motors turned off after inactivity."
            if notice:
                self.cancel.set()
                errors = self._stop_locked(notice)
                if errors:
                    self.fault = ("Could not verify torque off: " + ", ".join(errors)
                                  + "; disconnect 12 V power.")

    def _status_locked(self) -> dict:
        if self.fault:
            return self._snapshot_locked()
        try:
            states = inspect(self.bus)
            for name, state in states.items():
                if state.torque_enabled != (name in self.enabled):
                    raise RuntimeError("Motor torque changed outside this control session.")
                if name in self.enabled and state.operating_mode != 0:
                    raise RuntimeError(f"{name} left position mode")
            self._read_telemetry_locked()
            self.last_states = states
        except Exception as error:
            self._fault_locked(error)
        return self._snapshot_locked()

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
        self.starts[name] = self.goals[name] = self.targets[name] = state.position
        self.command_positions[name] = float(state.position)
        self.velocities[name] = 0.0
        self.last_step = time.monotonic()
        self.deadline = self.last_step + self.timeout

    def torque(self, name: str, enabled: bool, motion_epoch: int | None = None) -> dict:
        if name not in JOINTS or type(enabled) is not bool:
            raise ValueError("Choose a joint and true or false for torque.")
        with self.lock:
            self.expire()
            if enabled:
                self._check_epoch_locked(motion_epoch)
                if self.stop_pending.is_set():
                    raise RuntimeError("Stop is in progress. Wait for motors off before enabling.")
            if self.fault:
                return self._snapshot_locked()
            if enabled:
                self.cancel.clear()
                self.control_mode = "position"
                try:
                    self._enable_joint_locked(name)
                except Exception as error:
                    self._fault_locked(error)
                    return self._snapshot_locked()
                self.notice = f"{name} enabled. Drag its slider to move."
            else:
                self.motion_epoch += 1
                try:
                    self.bus.disable_torque(name)
                    if int(self.bus.read("Torque_Enable", name, normalize=False)) != 0:
                        raise RuntimeError(f"{name} torque did not turn off")
                except Exception as error:
                    self._fault_locked(error)
                    return self._snapshot_locked()
                self.enabled.discard(name)
                for values in (self.starts, self.goals, self.targets,
                               self.command_positions, self.velocities, self.tracking_since):
                    values.pop(name, None)
                if not self.enabled:
                    self.deadline = self.lease_deadline = None
                self.notice = f"{name} torque is off."
            return self._status_locked()

    def _check_epoch_locked(self, motion_epoch: int) -> None:
        if type(motion_epoch) is not int or motion_epoch != self.motion_epoch:
            raise ValueError("Control session changed. Read the current state before moving.")

    def _queue_target_locked(self, name: str, target: int) -> dict:
        if name not in self.enabled or self.cancel.is_set():
            raise RuntimeError(f"Enable {name} torque first.")
        lower, upper = self._bounds_locked(name)
        if not lower <= target <= upper:
            raise ValueError(f"{name} target must be {lower}..{upper} ticks")
        self.targets[name] = target
        now = time.monotonic()
        self.lease_deadline = now + self.motion_lease_seconds
        self.deadline = now + self.timeout
        self.notice = f"{name}: moving toward {target} ticks."
        return self._snapshot_locked()

    def jog(self, name: str, delta: int, motion_epoch: int) -> dict:
        if name not in JOINTS or type(delta) is not int or not 1 <= abs(delta) <= self.max_step:
            raise ValueError(f"Choose a joint and a step of 1..{self.max_step} ticks.")
        with self.lock:
            self.expire()
            self._check_epoch_locked(motion_epoch)
            if self.fault:
                return self._snapshot_locked()
            return self._queue_target_locked(name, self.targets.get(name, 0) + delta)

    def target(self, name: str, target: int, motion_epoch: int) -> dict:
        if name not in JOINTS or type(target) is not int:
            raise ValueError("Choose a joint and an integer target tick.")
        with self.lock:
            self.expire()
            self._check_epoch_locked(motion_epoch)
            if self.fault:
                return self._snapshot_locked()
            return self._queue_target_locked(name, target)

    def heartbeat(self, motion_epoch: int) -> dict:
        with self.lock:
            self.expire()
            self._check_epoch_locked(motion_epoch)
            if not self.fault and self.enabled and not self.cancel.is_set():
                self.lease_deadline = time.monotonic() + self.motion_lease_seconds
            return self._snapshot_locked()

    def pause(self, motion_epoch: int) -> dict:
        with self.lock:
            self._check_epoch_locked(motion_epoch)
            self.motion_epoch += 1
            for name in self.enabled:
                if name in self.goals:
                    self.targets[name] = self.goals[name]
                    self.command_positions[name] = float(self.goals[name])
                    self.velocities[name] = 0.0
            self.lease_deadline = None
            self.notice = "Movement paused; enabled motors still hold position."
            return self._snapshot_locked()

    def set_speed(self, speed: str) -> dict:
        if speed not in self.SPEEDS:
            raise ValueError("Choose fine, normal or fast speed.")
        with self.lock:
            self.speed = speed
            return self._snapshot_locked()

    def motion_step(self, now: float | None = None) -> None:
        """Advance one small ramp step. Serial writes never wait for a whole move."""
        with self.lock:
            now = time.monotonic() if now is None else now
            dt = max(0.0, min(0.05, now - self.last_step))
            self.last_step = now
            self.expire(now)
            if dt == 0 or self.fault or self.cancel.is_set() or not self.enabled:
                return
            try:
                for name in tuple(self.enabled):
                    if self.cancel.is_set():
                        return
                    measured = int(self.bus.read("Present_Position", name, normalize=False))
                    self.last_states[name] = replace(self.last_states[name], position=measured)
                    error = abs(self.goals[name] - measured)
                    if error > self.tracking_tolerance:
                        since = self.tracking_since.setdefault(name, now)
                        if now - since >= self.tracking_timeout:
                            raise RuntimeError(f"{name} is not tracking its target; check for obstruction")
                        # Let the servo catch up before commanding more travel.
                        self.velocities[name] = 0.0
                        continue
                    self.tracking_since.pop(name, None)
                    difference = self.targets[name] - self.command_positions[name]
                    if abs(difference) < 0.001:
                        self.velocities[name] = 0.0
                        continue
                    # A reversed slider must not send more travel in the old direction.
                    if self.velocities[name] * difference < 0:
                        self.velocities[name] = 0.0
                    self.velocities[name] = max(-self.SPEEDS[self.speed],
                                               min(self.SPEEDS[self.speed], self.velocities[name]))
                    desired_velocity = math.copysign(min(self.SPEEDS[self.speed],
                        math.sqrt(2 * self.acceleration * abs(difference))), difference)
                    dv = desired_velocity - self.velocities[name]
                    velocity = self.velocities[name] + max(-self.acceleration * dt,
                                                           min(self.acceleration * dt, dv))
                    movement = velocity * dt
                    next_position = self.command_positions[name] + movement
                    if (difference > 0 and next_position >= self.targets[name]) or (
                            difference < 0 and next_position <= self.targets[name]):
                        next_position = float(self.targets[name])
                        velocity = 0.0
                    lower, upper = self._bounds_locked(name)
                    next_position = max(lower, min(upper, next_position))
                    next_position = max(self.goals[name] - self.max_step,
                                        min(self.goals[name] + self.max_step, next_position))
                    next_goal = round(next_position)
                    next_goal = max(self.goals[name] - self.max_step,
                                    min(self.goals[name] + self.max_step, next_goal))
                    if next_goal != self.goals[name]:
                        self.bus.write("Goal_Position", name, next_goal, normalize=False)
                        self.goals[name] = next_goal
                        self.deadline = now + self.timeout
                    self.command_positions[name] = next_position
                    self.velocities[name] = velocity
            except Exception as error:
                self._fault_locked(error)

    def stop(self) -> dict:
        self.stop_pending.set()
        self.cancel.set()
        with self.lock:
            try:
                errors = self._stop_locked("Motors are off.")
                if errors:
                    self.fault = ("Could not verify torque off for "
                                  + ", ".join(errors) + "; disconnect 12 V power.")
                return self._status_locked()
            finally:
                self.stop_pending.clear()

    def set_mode(self, mode: str) -> dict:
        if mode not in ("manual", "position"):
            raise ValueError("Choose manual or position control mode.")
        if mode == "manual":
            return self.stop()
        with self.lock:
            if self.fault:
                return self._snapshot_locked()
            self.control_mode = "position"
            self.notice = "Position control selected; enable joints individually."
            return self._status_locked()

    def recover(self) -> dict:
        self.cancel.set()
        with self.lock:
            errors = self._stop_locked("Checking the arm after its fault.")
            if errors:
                self.fault = ("Still cannot turn off " + ", ".join(errors)
                              + "; disconnect 12 V power and check the arm.")
                return self._snapshot_locked()
            try:
                states = inspect(self.bus)
                validate_motion_state(states)
            except Exception as error:
                self.fault = f"Motor fault persists: {str(error).splitlines()[0]}"
                return self._snapshot_locked()
            self.last_states = states
            self.fault = None
            self.cancel.clear()
            self.notice = "Arm reconnected; motors are off."
            return self._status_locked()

    def close(self) -> None:
        self.stop_pending.set()
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
                    result = controller.jog(data.get("joint"), data.get("delta"), data.get("motion_epoch"))
                elif self.path == "/api/target":
                    result = controller.target(data.get("joint"), data.get("target"), data.get("motion_epoch"))
                elif self.path == "/api/heartbeat":
                    result = controller.heartbeat(data.get("motion_epoch"))
                elif self.path == "/api/pause":
                    result = controller.pause(data.get("motion_epoch"))
                elif self.path == "/api/speed":
                    result = controller.set_speed(data.get("speed"))
                elif self.path == "/api/torque":
                    result = controller.torque(data.get("joint"), data.get("enabled"),
                                               data.get("motion_epoch"))
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
    thread = None
    stop_watchdog = threading.Event()

    def watchdog():
        while not stop_watchdog.wait(0.02):
            try:
                controller.motion_step()
            except Exception as error:
                with controller.lock:
                    controller._fault_locked(error)

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
        controller.cancel.set()
        stop_watchdog.set()
        if thread is not None:
            thread.join(timeout=2)
        if server is not None:
            server.server_close()
        controller.close()
