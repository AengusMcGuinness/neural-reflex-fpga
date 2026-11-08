"""Live band-IMU control of the SO-ARM101: your forearm moves the whole robot at once.

  forearm tilt up/down    -> gripper height: shoulder lift, elbow and wrist flex move together
                             (inverse kinematics at a fixed reach, in front of the robot)
  forearm swing left/right -> base rotation
  forearm roll (palm up)   -> wrist roll
The gripper stays off. Settings: config/imu_teleop.json.

Calibration (about 12 s, prompted): hold your forearm level and still, then raise it, swing it right
and roll your palm up, holding each briefly. That teaches which way the band's axes point, so it
does not matter how the band sits on your arm.

DRY RUN by default: prints the joint angles live as you move; no arm needed. With --execute (and
typing MOVE) it checks the arm is at straight-up home, powers the five joints, moves to the neutral
pose, then follows your forearm at the controller's fastest speed. Ctrl+C holds the arm and asks you
to support it before the motors turn off. If band data stops arriving, the arm holds still.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

from .kinesis import _consume_sse_line, assert_local_stream_url

REPO = Path(__file__).resolve().parent.parent
CONFIG_FILE = REPO / "config" / "imu_teleop.json"
ROBOT_MAP = REPO / "config" / "robot_map.json"
HOME_FILE = REPO / "config" / "arm_home.json"
TICKS_PER_DEGREE = 4096 / 360
PLANAR = ("shoulder_lift", "elbow_flex", "wrist_flex")
MOVING = ("shoulder_pan", *PLANAR, "wrist_roll")
HOME_TOLERANCE = 150


# ---------- rotation helpers ----------------------------------------------------------------------
def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rotation_vector(m: np.ndarray) -> np.ndarray:
    """Axis * angle (radians) of a rotation matrix."""
    cos = np.clip((np.trace(m) - 1) / 2, -1.0, 1.0)
    angle = math.acos(cos)
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    if np.linalg.norm(axis) < 1e-9:                      # ~180 degrees: use the symmetric part
        axis = np.sqrt(np.clip((np.diag(m) + 1) / 2, 0, None))
    return axis / np.linalg.norm(axis) * angle


def mean_rotation(quats) -> np.ndarray:
    """Average of several nearby orientations (sign-aligned quaternion mean)."""
    q = np.asarray(quats, dtype=float)
    q = q * np.sign(q @ q[0])[:, None]
    return quat_to_matrix(q.mean(axis=0))


class Calibration:
    """Neutral pose plus the band-independent tilt, swing and roll axes, in the band's world frame."""

    def __init__(self, neutral: np.ndarray, tilt: np.ndarray, swing: np.ndarray, roll: np.ndarray):
        self.neutral = neutral
        t = tilt / np.linalg.norm(tilt)
        s = swing - np.dot(swing, t) * t
        s /= np.linalg.norm(s)
        r = roll - np.dot(roll, t) * t - np.dot(roll, s) * s
        r /= np.linalg.norm(r)
        self.axes = np.stack([t, s, r])                  # rows: tilt, swing, roll (positive = calibrated way)

    @classmethod
    def from_poses(cls, neutral, raised, swung, rolled, min_deg=15.0):
        rel = [rotation_vector(m @ neutral.T) for m in (raised, swung, rolled)]
        for name, v in zip(("raise", "swing right", "roll palm up"), rel):
            if math.degrees(np.linalg.norm(v)) < min_deg:
                raise ValueError(f"the '{name}' move was only {math.degrees(np.linalg.norm(v)):.0f}°: "
                                 f"move further (at least {min_deg:g}°) and calibrate again")
        return cls(neutral, *rel)

    def angles(self, current: np.ndarray) -> np.ndarray:
        """(tilt, swing, roll) in degrees from neutral."""
        return np.degrees(self.axes @ rotation_vector(current @ self.neutral.T))


# ---------- forearm -> robot ----------------------------------------------------------------------
class Mapper:
    """Forearm angles -> target ticks for the five moving joints, inside limits and load."""

    def __init__(self, cfg: dict, robot_map: dict, home: dict):
        from .pick_sim import check_pose, inverse_kinematics
        self.cfg, self.home = cfg, home
        margin = robot_map["margin_ticks"]
        self.tick_limits = {j: (lo + margin, hi - margin) for j, (lo, hi) in robot_map["limits"].items()}
        limits = {j: ((self.tick_limits[j][0] - home[j]) / TICKS_PER_DEGREE,
                      (self.tick_limits[j][1] - home[j]) / TICKS_PER_DEGREE) for j in PLANAR}
        # Height -> (lift, elbow, wrist) table, built top-down so the gripper angle changes smoothly.
        heights, table, prev_pitch = [], [], 180.0
        for z in np.arange(cfg["height_at_top_m"], cfg["height_at_table_m"] - 1e-9, -0.002):
            for pitch in sorted(np.arange(180, 104, -1.0), key=lambda p: (abs(p - prev_pitch), -p)):
                a = inverse_kinematics(cfg["reach_m"], z, pitch)
                if a is not None and not check_pose(a, limits):
                    heights.append(z)
                    table.append(a)
                    prev_pitch = pitch
                    break
        if len(heights) < 10:
            raise ValueError("no reachable heights at this reach; change reach_m in config/imu_teleop.json")
        order = np.argsort(heights)
        self.heights, self.table = np.array(heights)[order], np.array(table)[order]

    def height_for(self, tilt: float) -> float:
        c = self.cfg
        s = (tilt - c["tilt_deg_at_table"]) / (c["tilt_deg_at_top"] - c["tilt_deg_at_table"])
        z = c["height_at_table_m"] + min(max(s, 0.0), 1.0) * (c["height_at_top_m"] - c["height_at_table_m"])
        return float(np.clip(z, self.heights[0], self.heights[-1]))

    def degrees(self, tilt: float, swing: float, roll: float) -> dict:
        c = self.cfg
        z = self.height_for(tilt)
        lift, elbow, wrist = (float(np.interp(z, self.heights, self.table[:, k])) for k in range(3))
        base = float(np.clip(c["swing_gain"] * swing, -c["base_limit_deg"], c["base_limit_deg"]))
        wroll = float(np.clip(c["roll_gain"] * roll, -c["wrist_roll_limit_deg"], c["wrist_roll_limit_deg"]))
        return {"shoulder_pan": c["base_sign"] * base, "shoulder_lift": lift, "elbow_flex": elbow,
                "wrist_flex": wrist, "wrist_roll": c["wrist_roll_sign"] * wroll, "height_m": z}

    def ticks(self, deg: dict) -> dict:
        out = {}
        for j in MOVING:
            lo, hi = self.tick_limits[j]
            out[j] = int(round(min(max(self.home[j] + deg[j] * TICKS_PER_DEGREE, lo), hi)))
        return out


# ---------- live band orientation -----------------------------------------------------------------
class OrientationFeed:
    def __init__(self, url: str):
        self.url = assert_local_stream_url(url)
        self.lock = threading.Lock()
        self.latest = None          # (receipt monotonic time, quaternion)
        self.error = None
        self.stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            request = urllib.request.Request(self.url, headers={"Accept": "text/event-stream"})
            with opener.open(request, timeout=5) as response:
                event, data = None, []
                for raw in response:
                    if self.stop.is_set():
                        return
                    event, data, payload = _consume_sse_line(raw.decode("utf-8"), event, data)
                    if payload and payload.get("type") == "orientation":
                        with self.lock:
                            self.latest = (time.monotonic(), payload["payload"]["values"])
        except Exception as error:   # noqa: BLE001 - reported to the user
            self.error = str(error)

    def get(self):
        with self.lock:
            return self.latest

    def collect(self, seconds: float) -> list:
        out, end, last = [], time.monotonic() + seconds, None
        while time.monotonic() < end:
            item = self.get()
            if item and item is not last:
                out.append(item[1])
                last = item
            time.sleep(0.01)
        return out


def prompt_hold(feed, text, seconds, settle=0.8):
    for left in range(int(math.ceil(seconds + settle)), 0, -1):
        sys.stdout.write(f"\r  {text}  {left} ")
        sys.stdout.flush()
        if left > seconds:
            time.sleep(1.0)
    sys.stdout.write("\r  " + text + "  holding...")
    sys.stdout.flush()
    samples = feed.collect(seconds)
    print(" ok")
    if len(samples) < 10:
        raise RuntimeError("no band orientation arriving: is Kinesis sending orientation? (turn the local "
                           "sensor feed off and on, then copy the new URL)")
    return mean_rotation(samples)


def calibrate(feed, cfg) -> Calibration:
    hold = cfg["calibration_hold_s"]
    print("\nCALIBRATION. Sit as you will during control; move only your band forearm.")
    neutral = prompt_hold(feed, "Forearm LEVEL, pointing forward, palm down. Hold still.", hold)
    raised = prompt_hold(feed, "RAISE your forearm about 45° (hand up) and hold.", hold)
    prompt_hold(feed, "Back to level.", 1.0)
    swung = prompt_hold(feed, "SWING your forearm to your RIGHT about 45° and hold.", hold)
    prompt_hold(feed, "Back to the middle.", 1.0)
    rolled = prompt_hold(feed, "ROLL your palm to face UP and hold.", hold)
    prompt_hold(feed, "Back to palm down, level. Hold.", 1.0)
    return Calibration.from_poses(neutral, raised, swung, rolled, cfg["min_calibration_move_deg"])


def status_line(angles, deg, age):
    tilt, swing, roll = angles
    stale = "  BAND DATA STALE: holding" if age is None else ""
    return (f"\r you: tilt {tilt:+5.0f}° swing {swing:+5.0f}° roll {roll:+5.0f}°  ->  base {deg['shoulder_pan']:+4.0f}° "
            f"shoulder {deg['shoulder_lift']:+4.0f}° elbow {deg['elbow_flex']:+4.0f}° wrist {deg['wrist_flex']:+4.0f}° "
            f"roll {deg['wrist_roll']:+4.0f}°  height {deg['height_m'] * 100:4.1f} cm{stale}   ")


class Smoother:
    def __init__(self, seconds):
        self.seconds, self.value, self.t = seconds, None, None

    def __call__(self, values: np.ndarray, now: float) -> np.ndarray:
        if self.value is None or self.seconds <= 0:
            self.value, self.t = values, now
            return values
        alpha = min(1.0, (now - self.t) / (self.seconds + (now - self.t)))
        self.value, self.t = self.value + alpha * (values - self.value), now
        return self.value


def live_targets(feed, calibration, mapper, smoother, stale_after):
    item = feed.get()
    now = time.monotonic()
    if item is None or now - item[0] > stale_after:
        return None, None, None
    angles = smoother(calibration.angles(quat_to_matrix(item[1])), now)
    deg = mapper.degrees(*angles)
    return angles, deg, mapper.ticks(deg)


def run_robot(controller, feed, calibration, mapper, cfg, log=print, duration=None, rate_hz=50):
    """Follow the band until Ctrl+C (or `duration` seconds, for tests). Returns normally after Ctrl+C."""
    snap = controller.status()
    if snap.get("fault"):
        raise RuntimeError(snap["fault"])
    pos = {j["name"]: j["position"] for j in snap["joints"]}
    off = {j: pos[j] - mapper.home[j] for j in MOVING if abs(pos[j] - mapper.home[j]) > HOME_TOLERANCE}
    if off:
        raise RuntimeError("put the arm in its straight-up home pose first; off by " +
                           ", ".join(f"{j} {v / TICKS_PER_DEGREE:+.0f}°" for j, v in off.items()))
    controller.set_speed(cfg["speed"])
    stop_loop = threading.Event()

    def motion_loop():
        while not stop_loop.wait(0.02):
            controller.motion_step()

    loop = threading.Thread(target=motion_loop, daemon=True)
    loop.start()
    try:
        for j in MOVING:
            snap = controller.torque(j, True, controller.motion_epoch)
            if snap.get("fault"):
                raise RuntimeError(snap["fault"])
        neutral = mapper.ticks(mapper.degrees(0.0, 0.0, 0.0))
        log("motors on; moving to the neutral pose...")
        deadline = time.monotonic() + 40
        while True:
            for j, t in neutral.items():
                controller.target(j, t, controller.motion_epoch)
            snap = controller.status()
            if snap.get("fault"):
                raise RuntimeError(snap["fault"])
            pos = {j["name"]: j["position"] for j in snap["joints"]}
            if all(abs(pos[j] - t) <= 20 for j, t in neutral.items()):
                break
            if time.monotonic() > deadline:
                raise RuntimeError("arm did not reach the neutral pose")
            time.sleep(0.1)
        log("following your forearm now (Ctrl+C to stop)")
        smoother, last, t0, checked = Smoother(cfg["smooth_s"]), neutral, time.monotonic(), 0.0
        while duration is None or time.monotonic() - t0 < duration:
            angles, deg, ticks = live_targets(feed, calibration, mapper, smoother, cfg["stale_after_s"])
            if ticks is not None:
                last = ticks
            for j, t in last.items():                    # stale band data: keep holding the last targets
                controller.target(j, t, controller.motion_epoch)
            now = time.monotonic()
            if now - checked > 0.5:
                checked = now
                snap = controller.status()
                if snap.get("fault"):
                    raise RuntimeError(snap["fault"])
                if deg is not None:
                    sys.stdout.write(status_line(angles, deg, 0.0))
                    sys.stdout.flush()
            time.sleep(1.0 / rate_hz)
    finally:
        stop_loop.set()
        loop.join(timeout=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="Kinesis stream URL")
    parser.add_argument("--port", help="arm USB serial port (needed with --execute)")
    parser.add_argument("--execute", action="store_true", help="drive the arm (default: dry run, prints angles)")
    args = parser.parse_args(argv)
    cfg = json.loads(CONFIG_FILE.read_text())
    robot_map = json.loads(ROBOT_MAP.read_text())
    home = json.loads(HOME_FILE.read_text())["positions"]
    mapper = Mapper(cfg, robot_map, home)
    print(f"reachable gripper heights at {cfg['reach_m'] * 100:.0f} cm out: "
          f"{mapper.heights[0] * 100:.0f}–{mapper.heights[-1] * 100:.0f} cm; speed '{cfg['speed']}'")
    feed = OrientationFeed(args.url)
    time.sleep(1.0)
    if feed.error or feed.get() is None:
        print(f"No band orientation arriving ({feed.error or 'nothing yet'}). Check Kinesis and the stream URL.")
        return 1
    try:
        calibration = calibrate(feed, cfg)
    except (ValueError, RuntimeError) as error:
        print(f"\nCalibration failed: {error}")
        return 1
    print("calibrated.")
    if not args.execute:
        print("\nDRY RUN: move your forearm and watch the robot joints change. Ctrl+C to quit.")
        smoother = Smoother(cfg["smooth_s"])
        try:
            while True:
                angles, deg, _ = live_targets(feed, calibration, mapper, smoother, cfg["stale_after_s"])
                if deg is not None:
                    sys.stdout.write(status_line(angles, deg, 0.0))
                else:
                    sys.stdout.write("\r  band data stale ...                                         ")
                sys.stdout.flush()
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nDry run ended; nothing moved.")
            return 0
    if not args.port:
        parser.error("--execute needs --port")
    print("\nThe arm will move FAST (up to ~32°/s per joint) and five motors move at once.")
    print("Clear its area, keep the 12 V plug in reach, start with the arm straight up.")
    if input("Type MOVE and press Enter to start: ").strip() != "MOVE":
        print("Not started.")
        return 1
    from .arm_web import ArmController
    from .leader_arm import make_bus
    controller = ArmController(make_bus(args.port), timeout=15.0)
    code = 0
    try:
        run_robot(controller, feed, calibration, mapper, cfg)
    except KeyboardInterrupt:
        print("\nStopped: holding the arm where it is.")
    except RuntimeError as error:
        print(f"\nStopped: {error}")
        code = 1
    finally:
        try:
            controller.pause(controller.motion_epoch)
        except Exception:   # noqa: BLE001
            pass
        try:
            input("\nSupport the arm with your hand, then press Enter to turn the motors off... ")
        finally:
            controller.close()
            print("Motors off.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
