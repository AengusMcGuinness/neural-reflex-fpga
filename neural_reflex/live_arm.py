"""Live loop: band EMG + motion -> features -> decoder (CPU, golden integer model, or Urbana FPGA) -> leader arm.

    # no hardware: replay a recorded block through the FPGA's integer math, print what the arm would do
    python -m neural_reflex.live_arm --replay data/sessions/<id>/blocks/arm-reach --decoder golden
    # live band, FPGA decoder, arm moving
    ./scripts/live_arm --url http://127.0.0.1:PORT/stream --decoder fpga --fpga-port /dev/cu.usbserial-XXXX \\
        --arm-port /dev/cu.YOUR_ARM_PORT --enable-motion

Features are computed exactly as in training (armfit.pair_block): causal band-pass/notch filter,
window_features over the last 200 ms every 20 ms, four lags, plus band orientation and gyro.
The decoder outputs robot ticks (decode_arm folded robot_map.json into the weights).
The arm thread low-pass filters each joint, limits speed, checks tracking, and turns torque off if
predictions stop arriving for STALE_S, on any error, or on Ctrl-C.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.signal import sosfilt

from .armfit import quaternion_matrix
from .config import HOP_MS, LAGS_MS, SAMPLE_RATE_HZ, WINDOW_MS
from .decode_arm import ARM_TARGETS, predict_ticks
from .handfit import read_jsonl
from .leader_arm import disable_all_torque, inspect, make_bus, validate_motion_state
from .robot_map import load_map
from .signal import design_sos, window_features

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "urbana" / "python"))
import fixed_point as fp  # noqa: E402
import uart_protocol as up  # noqa: E402

STALE_S = 0.3
SETTLE_S = 0.1       # band-pass start-up transient after the filter (re)starts; those windows are not decoded
TRACKING_TOLERANCE = 60
TRACKING_TIMEOUT = 0.5


class FeatureStream:
    """Turns EMG batches and motion events into the decoder's 175 features, one row per 20 ms hop."""

    def __init__(self, use_motion: bool = True):
        self.window = int(SAMPLE_RATE_HZ * WINDOW_MS / 1000)
        self.hop = int(SAMPLE_RATE_HZ * HOP_MS / 1000)
        self.lags = [int(ms // HOP_MS) for ms in LAGS_MS]
        self.sos = design_sos(SAMPLE_RATE_HZ)
        self.zi = np.zeros((self.sos.shape[0], 2, 8))
        self.samples = deque(maxlen=self.window)
        self.valid = deque(maxlen=self.window)
        self.rows = deque(maxlen=max(self.lags) + 1)
        self.n = 0
        self.last_sequence = None
        self.settle = int(SAMPLE_RATE_HZ * SETTLE_S)
        self.since_restart = 0
        self.use_motion = use_motion
        self.orientation = None
        self.gyro = deque()
        self.clock_offset = None   # host_time - band_time, smallest seen (least delayed batch)

    def motion(self, event: dict) -> None:
        if event.get("type") == "orientation":
            self.orientation = quaternion_matrix(np.array(event["payload"]["values"], float))
        elif event.get("type") == "gyro":
            k = event["payload"].get("observed_degrees_per_second_per_count", 0.07)
            self.gyro.append((event["host_time_s"], np.array(event["payload"]["values"], float) * k))

    def emg(self, record: dict) -> list[tuple[float, np.ndarray]]:
        """Feed one 16-sample batch; return (window end time, features) for each row it completes.

        The time is the window's last sample on the host clock: band sample time plus the smallest
        host-minus-band offset seen so far, so batch delivery jitter does not shift labels."""
        gap = self.last_sequence is not None and record["sequence"] - self.last_sequence > 1
        self.last_sequence = record["sequence"]
        raw = np.asarray(record["values"], dtype=float)
        valid = np.ones(len(raw), bool)
        if gap:  # same rule as training: the first sample after a gap is invalid and the filter restarts
            valid[0] = False
            self.zi[:] = 0
            self.since_restart = -1   # counts from the first filtered sample after the gap
            out = np.zeros_like(raw)
            out[1:], self.zi = sosfilt(self.sos, raw[1:], axis=0, zi=self.zi)
        else:
            out, self.zi = sosfilt(self.sos, raw, axis=0, zi=self.zi)
        t = record["host_time_s"]
        band_s = record["band_timestamp_us"] * 1e-6
        offset = t - band_s
        self.clock_offset = offset if self.clock_offset is None else min(self.clock_offset, offset)
        interval_s = record["sample_interval_us"] * 1e-6
        while self.gyro and self.gyro[0][0] <= t - WINDOW_MS / 1000:
            self.gyro.popleft()
        rows = []
        for index, (sample, ok) in enumerate(zip(out, valid)):
            self.since_restart += 1
            self.samples.append(sample)
            self.valid.append(ok and self.since_restart >= self.settle)
            end = self.n
            self.n += 1
            if end < self.window - 1 or (end - (self.window - 1)) % self.hop:
                continue
            self.rows.append(window_features(np.array(self.samples)) if all(self.valid) else None)
            if len(self.rows) <= max(self.lags):
                continue
            parts = [self.rows[-1 - lag] for lag in self.lags]
            if any(p is None for p in parts):
                continue
            x = np.concatenate(parts)
            if self.use_motion:
                if self.orientation is None or not self.gyro:
                    continue
                g = np.array([v for _, v in self.gyro])
                x = np.concatenate([x, self.orientation.ravel(), g.mean(axis=0), np.abs(g).mean(axis=0)])
            rows.append((band_s + index * interval_s + self.clock_offset, x))
        return rows


class CpuDecoder:
    name = "cpu"

    def __init__(self, model_path: Path):
        m = np.load(model_path)
        self.model = {k: m[k] for k in ("mean", "scale", "W", "b", "clamp_lo", "clamp_hi")}

    def __call__(self, x):
        return predict_ticks(self.model, x[None])[0].round().astype(int).tolist()


class GoldenDecoder:
    """The FPGA's exact integer math, run on the CPU (fixed_point.predict_int)."""
    name = "golden"

    def __init__(self, quant_path: Path):
        self.qp = fp.QuantParams.load(quant_path)

    def __call__(self, x):
        return fp.predict_int(fp.quantize_features(x, self.qp), self.qp).tolist()


class FpgaDecoder:
    """Urbana over USB UART: send a feature frame, wait for the decoded frame."""
    name = "fpga"

    def __init__(self, port: str, baud: int, quant_path: Path, load_weights: bool, timeout: float = 0.05):
        import serial
        self.qp = fp.QuantParams.load(quant_path)
        self.golden = GoldenDecoder(quant_path)
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.parser = up.Parser(self.qp.n_out)
        self.timeout = timeout
        self.mismatches = 0
        self.round_trips = deque(maxlen=500)
        if load_weights:
            self.ser.write(up.load_frame(self.qp))
            reply = self._wait(1.0)
            if reply != ("load_ok", None):
                raise RuntimeError(f"weight load failed: {reply}")

    def _wait(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for msg in self.parser.feed(self.ser.read(self.ser.in_waiting or 1)):
                return msg
        return None

    def __call__(self, x):
        f_q = fp.quantize_features(x, self.qp)
        t0 = time.perf_counter()
        self.ser.write(up.features_frame(f_q))
        msg = self._wait(self.timeout)
        self.round_trips.append(time.perf_counter() - t0)
        if msg is None or msg[0] != "result":
            raise RuntimeError(f"FPGA did not answer a feature frame: {msg}")
        if msg[1] != fp.predict_int(f_q, self.qp).tolist():
            self.mismatches += 1
        return msg[1]


class ArmThread(threading.Thread):
    """Follows the latest prediction at 50 Hz with smoothing, speed limits and safety stops."""

    def __init__(self, bus, joints, config: dict, speed: str, dt: float = 0.02):
        super().__init__(daemon=True)
        self.bus, self.joints, self.dt = bus, list(joints), dt
        self.cap = config["speed_caps_ticks_per_s"][speed]
        self.alpha = {j: dt / (config["joints"][j]["smooth_s"] + dt) for j in self.joints}
        margin = config["margin_ticks"]
        self.bounds = {j: (config["limits"][j][0] + margin, config["limits"][j][1] - margin) for j in self.joints}
        self.latest, self.latest_t = None, 0.0
        self.stop_event = threading.Event()
        self.error = None
        self.lock = threading.Lock()

    def update(self, ticks) -> None:
        with self.lock:
            self.latest, self.latest_t = dict(zip(self.joints, ticks)), time.monotonic()

    def run(self) -> None:
        try:
            states = inspect(self.bus)
            validate_motion_state(states)
            goal = {j: float(states[j].position) for j in self.joints}
            smooth = dict(goal)
            for j in self.joints:
                self.bus.write("Goal_Position", j, states[j].position, normalize=False)
                self.bus.enable_torque(j)
            lagging, step = {}, 0
            while not self.stop_event.is_set():
                t = time.monotonic()
                with self.lock:
                    latest, age = self.latest, t - self.latest_t
                if latest is None or age > STALE_S:
                    if latest is not None:
                        raise RuntimeError(f"predictions stale for {age:.2f}s")
                else:
                    for j in self.joints:
                        lo, hi = self.bounds[j]
                        smooth[j] += self.alpha[j] * (np.clip(latest[j], lo, hi) - smooth[j])
                        goal[j] += float(np.clip(smooth[j] - goal[j], -self.cap * self.dt, self.cap * self.dt))
                        self.bus.write("Goal_Position", j, int(round(goal[j])), normalize=False)
                if step % 5 == 0:
                    for j in self.joints:
                        err = abs(int(self.bus.read("Present_Position", j, normalize=False)) - goal[j])
                        if err <= TRACKING_TOLERANCE:
                            lagging.pop(j, None)
                        elif t - lagging.setdefault(j, t) > TRACKING_TIMEOUT:
                            raise RuntimeError(f"{j} is {err:.0f} ticks from its goal")
                step += 1
                time.sleep(max(0.0, self.dt - (time.monotonic() - t)))
        except BaseException as error:  # stop the arm on anything, report to the main loop
            self.error = error
        finally:
            errors = disable_all_torque(self.bus)
            if errors:
                print("Could not verify torque off for " + ", ".join(errors) + "; disconnect 12 V power.", flush=True)
            else:
                print("Arm: torque off on all joints.", flush=True)


def replay_events(block: Path, realtime: bool):
    emg = [("emg", r["host_time_s"], r) for r in read_jsonl(block / "emg.jsonl")]
    motion = [("motion", e["host_time_s"], e) for e in read_jsonl(block / "motion.jsonl")]
    events = sorted(emg + motion, key=lambda e: e[1])
    t0, wall0 = events[0][1], time.monotonic()
    for kind, t, e in events:
        if realtime:
            time.sleep(max(0.0, (t - t0) - (time.monotonic() - wall0)))
        yield kind, e


def kinesis_events(url: str):
    import io
    import queue
    import socket
    import urllib.request
    from .kinesis import assert_local_stream_url, iter_emg_records
    socket.setdefaulttimeout(5)
    request = urllib.request.Request(assert_local_stream_url(url),
                                     headers={"Accept": "text/event-stream", "Cache-Control": "no-cache"})
    pending = queue.SimpleQueue()
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=5) as response:
        for record in iter_emg_records(io.TextIOWrapper(response, encoding="utf-8"), pending.put):
            while not pending.empty():
                yield "motion", pending.get()
            yield "emg", record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="local Kinesis SSE stream")
    src.add_argument("--replay", help="recorded block folder (emg.jsonl + motion.jsonl)")
    parser.add_argument("--realtime", action="store_true", help="pace a replay at recording speed")
    parser.add_argument("--decoder", choices=("cpu", "golden", "fpga"), default="golden")
    parser.add_argument("--model", default=str(REPO / "data/models/arm_decoder.npz"))
    parser.add_argument("--quant", default=str(REPO / "urbana/build/quant_params.json"))
    parser.add_argument("--fpga-port")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--load-weights", action="store_true", help="send the weights over UART at start")
    parser.add_argument("--arm-port")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--speed", choices=("fine", "normal", "fast"), default="normal")
    parser.add_argument("--log", help="write every prediction to this JSONL file")
    args = parser.parse_args(argv)

    use_motion = str(np.load(args.model)["inputs"]) == "emg+motion"
    if args.decoder == "cpu":
        decode = CpuDecoder(Path(args.model))
    elif args.decoder == "golden":
        decode = GoldenDecoder(Path(args.quant))
    else:
        if not args.fpga_port:
            parser.error("--decoder fpga needs --fpga-port")
        decode = FpgaDecoder(args.fpga_port, args.baud, Path(args.quant), args.load_weights)
    arm = None
    if args.arm_port:
        if not args.enable_motion:
            parser.error("moving the arm needs --enable-motion")
        bus = make_bus(args.arm_port)
        bus.connect()
        arm = ArmThread(bus, ARM_TARGETS, load_map(), args.speed)
    log = open(args.log, "w") if args.log else None
    features = FeatureStream(use_motion)
    events = replay_events(Path(args.replay), args.realtime) if args.replay else kinesis_events(args.url)
    print(f"decoder={decode.name}, inputs={'emg+motion' if use_motion else 'emg'}, arm={'on' if arm else 'off'}; Ctrl-C stops")
    if arm:
        input("Clear the arm's sweep area. Press Enter to enable torque and follow the band... ")
        arm.start()
    n, latency, last_print = 0, deque(maxlen=500), time.monotonic()
    try:
        for kind, event in events:
            if kind == "motion":
                features.motion(event)
                continue
            received = time.perf_counter()
            for _, x in features.emg(event):
                ticks = decode(x)
                latency.append(time.perf_counter() - received)
                n += 1
                if arm:
                    if arm.error:
                        raise RuntimeError(f"arm stopped: {arm.error}")
                    arm.update(ticks)
                if log:
                    log.write(json.dumps({"t": event["host_time_s"], "ticks": ticks}) + "\n")
                now = time.monotonic()
                if now - last_print >= 1.0:
                    last_print = now
                    extra = ""
                    if isinstance(decode, FpgaDecoder):
                        rt = np.array(decode.round_trips) * 1000
                        extra = f"  uart round trip p50 {np.percentile(rt, 50):.1f} ms p95 {np.percentile(rt, 95):.1f} ms, mismatches {decode.mismatches}"
                    lat = np.array(latency) * 1000
                    print(f"{n:6d} frames  " + "  ".join(f"{j}={v}" for j, v in zip(ARM_TARGETS, ticks))
                          + f"  batch->decoded p50 {np.percentile(lat, 50):.2f} ms p95 {np.percentile(lat, 95):.2f} ms" + extra,
                          flush=True)
    except KeyboardInterrupt:
        print("\nStopped by Ctrl-C.")
    finally:
        if arm:
            arm.stop_event.set()
            arm.join(timeout=2)
            arm.bus.disconnect(disable_torque=False)
        if log:
            log.close()
    print(f"{n} decoded frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
