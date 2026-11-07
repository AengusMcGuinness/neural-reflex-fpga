"""Read-only live dashboard: SO-ARM101 joint positions beside the Kinesis band stream.

The arm side only reads registers. It never writes goals or changes torque, so
with torque off the arm can be moved by hand while the page follows it. The
band side relays the local Kinesis Server-Sent Events feed to the page, which
keeps the Kinesis token on the server and avoids cross-origin rules.
"""

from __future__ import annotations

import argparse
import io
import json
import queue
import socket
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .kinesis import _consume_sse_line, assert_local_stream_url
from .leader_arm import JOINTS

REPO = Path(__file__).resolve().parent.parent
WEB_DIR = REPO / "web" / "dashboard"
HOME_FILE = REPO / "config" / "arm_home.json"
STATIC = {
    "/dashboard.js": "text/javascript; charset=utf-8",
    "/dashboard.css": "text/css; charset=utf-8",
}
BAND_EVENTS = ("hello", "status", "emg", "emg_configuration", "gyro", "orientation", "gesture")


class Hub:
    """Fan out events to every connected browser; a slow browser drops events."""

    def __init__(self, maxsize: int = 2000):
        self.lock = threading.Lock()
        self.clients: set[queue.Queue] = set()
        self.maxsize = maxsize
        self.latest: dict[str, str] = {}

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(self.maxsize)
        with self.lock:
            self.clients.add(q)
            # Give a new page the latest state right away instead of waiting.
            for name, data in self.latest.items():
                q.put_nowait((name, data))
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            self.clients.discard(q)

    def publish(self, name: str, data: dict | str, *, sticky: bool = False) -> None:
        text = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
        with self.lock:
            if sticky:
                self.latest[name] = text
            for q in self.clients:
                try:
                    q.put_nowait((name, text))
                except queue.Full:
                    pass


class ArmReader:
    """Poll joint positions. Limits and torque are re-read every few seconds."""

    def __init__(self, bus_factory, hub: Hub, *, rate_hz: float = 30.0,
                 home: dict[str, int] | None = None):
        self.bus_factory = bus_factory
        self.hub = hub
        # Angles are reported relative to this pose (the arm pointing straight up).
        self.home = home or {name: 2048 for name in JOINTS}
        self.period = 1.0 / rate_hz
        self.stop_event = threading.Event()

    def read_static(self, bus) -> dict:
        joints = {}
        for name in JOINTS:
            raw = lambda register: int(bus.read(register, name, normalize=False))
            joints[name] = {
                "minimum": raw("Min_Position_Limit"),
                "maximum": raw("Max_Position_Limit"),
                "torque_enabled": bool(raw("Torque_Enable")),
            }
        voltage = int(bus.read("Present_Voltage", JOINTS[0], normalize=False)) / 10
        return {"joints": joints, "voltage_v": round(voltage, 1)}

    def read_positions(self, bus) -> dict[str, int]:
        return {name: int(bus.read("Present_Position", name, normalize=False)) for name in JOINTS}

    def run_once(self, bus, static: dict, sequence: int) -> None:
        positions = self.read_positions(bus)
        degrees = {name: round((positions[name] - self.home[name]) * 360 / 4096, 1)
                   for name in JOINTS}
        self.hub.publish("arm", {"connected": True, "time_s": time.time(),
                                 "sequence": sequence, "positions": positions,
                                 "home": self.home, "degrees_from_home": degrees,
                                 **static}, sticky=True)

    def run(self) -> None:
        while not self.stop_event.is_set():
            bus = None
            try:
                bus = self.bus_factory()
                bus.connect()
                static, static_due, sequence = self.read_static(bus), time.monotonic() + 2, 0
                while not self.stop_event.is_set():
                    started = time.monotonic()
                    if started >= static_due:
                        static, static_due = self.read_static(bus), started + 2
                    self.run_once(bus, static, sequence)
                    sequence += 1
                    self.stop_event.wait(max(0.0, self.period - (time.monotonic() - started)))
            except Exception as error:
                lines = [line.strip() for line in str(error).splitlines() if line.strip()]
                self.hub.publish("arm", {"connected": False,
                                         "error": lines[0] if lines else type(error).__name__},
                                 sticky=True)
                self.stop_event.wait(2.0)
            finally:
                if bus is not None:
                    try:
                        # Leave whatever torque state the arm had; this page never changes it.
                        bus.disconnect(disable_torque=False)
                    except Exception:
                        pass


class BandRelay:
    """Read the Kinesis SSE feed and republish its events; reconnect on failure."""

    def __init__(self, hub: Hub, url: str | None = None):
        self.hub = hub
        self.lock = threading.Lock()
        self.url = assert_local_stream_url(url) if url else None
        self.generation = 0
        self.stop_event = threading.Event()

    def set_url(self, url: str) -> None:
        url = assert_local_stream_url(url.strip())
        with self.lock:
            self.url = url
            self.generation += 1

    def _current(self) -> tuple[str | None, int]:
        with self.lock:
            return self.url, self.generation

    def status(self, state: str, detail: str = "") -> None:
        url, _ = self._current()
        port = urllib.parse.urlparse(url).port if url else None
        self.hub.publish("band_link", {"state": state, "detail": detail, "port": port},
                         sticky=True)

    def handle_lines(self, lines, generation: int) -> None:
        event_type, data_lines = None, []
        for line in lines:
            if self.stop_event.is_set() or self._current()[1] != generation:
                return
            event_type, data_lines, payload = _consume_sse_line(line, event_type, data_lines)
            if payload is None:
                continue
            kind = payload.get("type")
            if kind in BAND_EVENTS:
                self.hub.publish(kind, payload, sticky=kind in ("hello", "status",
                                                                  "emg_configuration"))

    def run(self) -> None:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while not self.stop_event.is_set():
            url, generation = self._current()
            if not url:
                self.status("waiting", "Paste the stream URL from Kinesis.")
                self.stop_event.wait(0.5)
                continue
            try:
                self.status("connecting")
                request = urllib.request.Request(
                    url, headers={"Accept": "text/event-stream", "Cache-Control": "no-cache"})
                with opener.open(request, timeout=5) as response:
                    self.status("live")
                    self.handle_lines(io.TextIOWrapper(response, encoding="utf-8"), generation)
                if self._current()[1] == generation and not self.stop_event.is_set():
                    self.status("error", "Kinesis closed the feed.")
            except urllib.error.HTTPError as error:
                hint = " The link is stale: copy a new one from Kinesis." if error.code in (401, 403) else ""
                self.status("error", f"HTTP {error.code}.{hint}")
            except (OSError, socket.timeout, ValueError) as error:
                self.status("error", f"{type(error).__name__}: {error}. Is the local sensor feed on?")
            if self._current()[1] == generation:
                self.stop_event.wait(2.0)


def make_handler(hub: Hub, relay: BandRelay, http_port: int, arm_enabled: bool):
    expected_host = f"127.0.0.1:{http_port}"
    expected_origin = f"http://{expected_host}"
    csp = ("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
           "img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass

        def _headers(self, code: int, content_type: str, length: int | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            if length is not None:
                self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", csp)
            self.end_headers()

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self._headers(code, content_type, len(body))
            self.wfile.write(body)

        def _json(self, code: int, data: dict) -> None:
            self._send(code, json.dumps(data).encode(), "application/json; charset=utf-8")

        def _host_ok(self) -> bool:
            if self.headers.get("Host") != expected_host:
                self._json(403, {"error": f"Open http://{expected_host}/"})
                return False
            return True

        def do_GET(self):
            if not self._host_ok():
                return
            if self.path == "/":
                page = (WEB_DIR / "index.html").read_text()
                page = page.replace("__ARM_ENABLED__", "true" if arm_enabled else "false")
                self._send(200, page.encode(), "text/html; charset=utf-8")
            elif self.path in STATIC:
                self._send(200, (WEB_DIR / self.path[1:]).read_bytes(), STATIC[self.path])
            elif self.path == "/events":
                self._stream()
            else:
                self._json(404, {"error": "Not found"})

        def _stream(self) -> None:
            self.close_connection = True
            self._headers(200, "text/event-stream; charset=utf-8")
            q = hub.subscribe()
            try:
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                while True:
                    try:
                        name, data = q.get(timeout=10)
                        chunk = f"event: {name}\ndata: {data}\n\n"
                        # Drain whatever else is waiting so the socket sees fewer writes.
                        for _ in range(200):
                            try:
                                name, data = q.get_nowait()
                            except queue.Empty:
                                break
                            chunk += f"event: {name}\ndata: {data}\n\n"
                    except queue.Empty:
                        chunk = ": keep-alive\n\n"
                    self.wfile.write(chunk.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                hub.unsubscribe(q)

        def do_POST(self):
            if not self._host_ok():
                return
            if self.headers.get("Origin") != expected_origin:
                self._json(403, {"error": "Open the page from the local server."})
                return
            if self.path != "/api/stream-url":
                self._json(404, {"error": "Not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2048:
                    raise ValueError("invalid request size")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict) or not isinstance(data.get("url"), str):
                    raise ValueError("expected {\"url\": \"...\"}")
                relay.set_url(data["url"])
                self._json(200, {"ok": True})
            except ValueError as error:
                self._json(400, {"error": str(error)})

    return Handler


def load_home(path: Path = HOME_FILE) -> dict[str, int] | None:
    """Read the saved home pose; None (servo centre 2048) if there is no file."""
    if not path.exists():
        print(f"No {path.name}; angles are relative to servo centre 2048.", flush=True)
        return None
    positions = json.loads(path.read_text())["positions"]
    if set(positions) != set(JOINTS) or not all(isinstance(v, int) and 0 <= v <= 4095
                                                for v in positions.values()):
        raise ValueError(f"{path} must give an integer 0..4095 tick for each of {', '.join(JOINTS)}")
    return positions


def serve(*, arm_port: str | None, stream_url: str | None, http_port: int = 8768,
          rate_hz: float = 30.0, bus_factory=None) -> None:
    hub = Hub()
    relay = BandRelay(hub, stream_url)
    threads = [threading.Thread(target=relay.run, daemon=True)]
    reader = None
    if arm_port:
        if bus_factory is None:
            from .leader_arm import make_bus
            bus_factory = lambda: make_bus(arm_port)
        reader = ArmReader(bus_factory, hub, rate_hz=rate_hz, home=load_home())
        threads.append(threading.Thread(target=reader.run, daemon=True))
    server = ThreadingHTTPServer(("127.0.0.1", http_port),
                                 make_handler(hub, relay, http_port, bool(arm_port)))
    server.daemon_threads = True
    for thread in threads:
        thread.start()
    print(f"Dashboard: http://127.0.0.1:{http_port}/", flush=True)
    print("Read-only: this never changes motor torque or goals. Ctrl+C stops.", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        relay.stop_event.set()
        if reader:
            reader.stop_event.set()
        server.server_close()
        for thread in threads:
            thread.join(timeout=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", help="arm USB serial port, e.g. /dev/cu.usbmodem…; omit for band only")
    parser.add_argument("--url", help="Kinesis stream URL (can also be pasted in the page)")
    parser.add_argument("--http-port", type=int, default=8768)
    parser.add_argument("--rate", type=float, default=30.0, help="arm reads per second")
    args = parser.parse_args(argv)
    if not 1 <= args.http_port <= 65535:
        parser.error("--http-port must be 1..65535")
    if not 1 <= args.rate <= 100:
        parser.error("--rate must be 1..100")
    serve(arm_port=args.port, stream_url=args.url, http_port=args.http_port, rate_hz=args.rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
