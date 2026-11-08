"""Drive the SO-ARM101 the way Kinesis's air mouse drives the cursor, plus a fist for the gripper.

Forearm aim, as in Kinesis (Sources/Kinesis/AirCursor.swift): the band's orientation quaternion
(w, x, y, z) turns the band's body frame into a world frame whose +z is up, and the band's body +y
runs along the forearm toward the hand on a right wrist (toward the elbow on a left wrist). So:
  elevation  forearm angle above the horizon (absolute, from gravity: no drift) -> gripper height
             (shoulder lift, elbow and wrist flex together, inverse kinematics at a fixed reach)
  azimuth    compass angle around gravity; its CHANGE since start (drift-free enough short term)
             -> base rotation
  twist      rotation around the forearm, change since start -> wrist roll
each smoothed with a 1-euro filter (heavy smoothing when nearly still, light when moving fast).
Fist: EMG activity relative to your relaxed level, calibrated with one relax and one fist -> gripper.

Dry run by default (prints everything live). --execute drives the arm through ArmController's
safety; Ctrl+C holds the arm and asks you to support it before the motors turn off.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .imu_teleop import CONFIG_FILE as TELEOP_CONFIG, HOME_FILE, HOME_TOLERANCE, MOVING, ROBOT_MAP, Mapper, quat_to_matrix
from .movements import BandStream

TICKS_PER_DEGREE = 4096 / 360
PLANAR = ("shoulder_lift", "elbow_flex", "wrist_flex")
RELATIVE = ("shoulder_pan", "wrist_roll", "gripper")   # start where they are, move relative to that
GRIPPER_RANGE_TICKS = 500          # fully closed, from config/robot_map.json's gripper gain
STALE_S = 0.3


class OneEuro:
    """1-euro filter (Casiez, Roussel and Vogel 2012), as Kinesis uses: min_cutoff 1.2 Hz, beta 0.12."""

    def __init__(self, min_cutoff=1.2, beta=0.12, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1 / (2 * math.pi * cutoff)
        return 1 / (1 + tau / dt)

    def __call__(self, x, t):
        if self.x is None or t <= self.t:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = t - self.t
        dx = (x - self.x) / dt
        self.dx += self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        self.x += self._alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x


def forearm_aim(quaternion, hand: str = "right"):
    """(azimuth, elevation, twist) in degrees. Twist: the band's +x around the forearm, measured from
    the horizontal line across the forearm."""
    r = quat_to_matrix(quaternion)
    axis = np.array([0.0, 1.0 if hand == "right" else -1.0, 0.0])
    f = r @ axis
    azimuth = math.degrees(math.atan2(f[1], f[0]))
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, f[2]))))
    across = np.cross(f, [0.0, 0.0, 1.0])      # horizontal, across the forearm: band +x when untwisted
    if np.linalg.norm(across) < 1e-6:                     # forearm vertical: twist undefined
        return azimuth, elevation, None
    across /= np.linalg.norm(across)
    x = r @ np.array([1.0, 0.0, 0.0])
    x -= np.dot(x, f) * f
    twist = math.degrees(math.atan2(np.dot(np.cross(across, x), f), np.dot(across, x)))
    return azimuth, elevation, twist


def unwrap(previous, angle):
    """Continue `angle` from `previous` across the ±180° seam."""
    if previous is None:
        return angle
    return previous + (angle - previous + 180) % 360 - 180


class AirAim:
    """Filtered forearm aim -> (elevation, change in azimuth, change in twist) since start."""

    def __init__(self, hand: str):
        self.hand = hand
        self.filters = [OneEuro(), OneEuro(), OneEuro()]
        self.last = [None, None, None]
        self.start = None

    def update(self, quaternion, t):
        az, el, tw = forearm_aim(quaternion, self.hand)
        if tw is None:
            tw = self.last[2] if self.last[2] is not None else 0.0
        raw = [unwrap(self.last[0], az), el, unwrap(self.last[2], tw)]
        self.last = raw
        smooth = [f(v, t) for f, v in zip(self.filters, raw)]
        if self.start is None:
            self.start = smooth
        return smooth[1], smooth[0] - self.start[0], smooth[2] - self.start[2]


def emg_activity(emg):
    """Per channel mean |change| of the raw counts over the window."""
    return np.mean(np.abs(np.diff(emg, axis=0)), axis=0)


class FistMeter:
    """0 = relaxed, 1 = your calibration fist, from the channel that rises most."""

    def __init__(self, relaxed, fist):
        self.relaxed = np.maximum(np.asarray(relaxed, float), 1e-6)
        ratio = np.asarray(fist, float) / self.relaxed
        self.channel = int(np.argmax(ratio))
        self.ratio = float(ratio[self.channel])
        if self.ratio < 1.8:
            raise ValueError(f"the fist only raised EMG {self.ratio:.1f}x (need 1.8x): tighten the band, make a "
                             "firmer fist, and calibrate again")

    def level(self, activity):
        r = activity[self.channel] / self.relaxed[self.channel]
        return float(np.clip((r - 1.3) / (self.ratio - 1.3), 0.0, 1.0))


def calibrate_fist(stream: BandStream, sleep=time.sleep, out=sys.stdout):
    def hold(text, seconds):
        t0 = time.monotonic()
        for left in range(int(seconds), 0, -1):
            out.write(f"\r  {text}  {left} ")
            out.flush()
            sleep(1.0)
        out.write("\n")
        samples = []
        t = t0 + 0.7
        while t < time.monotonic():
            emg, _, _ = stream.window(t, 0.15)
            if emg is not None and len(emg) > 16:
                samples.append(emg_activity(emg))
            t += 0.05
        if len(samples) < 5:
            raise RuntimeError("no EMG arriving")
        return np.median(samples, axis=0)

    relaxed = hold("Relax your hand (forearm level, palm down)", 3)
    fist = hold(">>> Make a FIRM FIST and hold", 3)
    hold("Relax", 2)
    return FistMeter(relaxed, fist)


def band_hand(url):
    try:
        from .fist_check import band_hand as hand_from_kinesis
        return hand_from_kinesis(url) or "right"
    except Exception:   # noqa: BLE001
        return "right"


class AirArm:
    """Turns the live band into joint targets every tick."""

    def __init__(self, stream, aim: AirAim, meter: FistMeter, mapper: Mapper, home: dict, robot_map: dict):
        self.stream, self.aim, self.meter, self.mapper, self.home = stream, aim, meter, mapper, home
        lo, hi = robot_map["limits"]["gripper"]
        margin = robot_map["margin_ticks"]
        self.grip_limits = (lo + margin, hi - margin)
        self.grip = OneEuro(min_cutoff=2.0, beta=0.5)
        self.tick_limits = dict(mapper.tick_limits, gripper=self.grip_limits)
        self.state = None

    def tick(self, now):
        emg, _, q = self.stream.window(now, 0.15)
        if q is None or emg is None or len(emg) < 16:
            return None
        elevation, swing, twist = self.aim.update(q, now)
        deg = self.mapper.degrees(elevation, swing, twist)
        targets = self.mapper.ticks(deg)
        grip = float(np.clip(self.grip(self.meter.level(emg_activity(emg)), now), 0, 1))
        targets["gripper"] = int(round(min(max(self.home["gripper"] + GRIPPER_RANGE_TICKS * grip,
                                               self.grip_limits[0]), self.grip_limits[1])))
        self.state = {"elevation": elevation, "swing": swing, "twist": twist, "grip": grip, "deg": deg}
        return targets

    def line(self):
        s = self.state
        if s is None:
            return "\r  waiting for band data...                         "
        d = s["deg"]
        return (f"\r  forearm: up {s['elevation']:+4.0f}° swing {s['swing']:+4.0f}° twist {s['twist']:+4.0f}° "
                f"fist {100 * s['grip']:3.0f}%  ->  base {d['shoulder_pan']:+4.0f}° height {d['height_m'] * 100:4.1f} cm "
                f"(shoulder {d['shoulder_lift']:+4.0f}° elbow {d['elbow_flex']:+4.0f}° wrist {d['wrist_flex']:+4.0f}°) "
                f"roll {d['wrist_roll']:+4.0f}° gripper {100 * s['grip']:3.0f}%   ")


def follow_band(controller, air: AirArm, duration=None, show=True, rate_hz=50):
    """Drive the arm from the band until Ctrl+C or `duration`. Holds the last targets on stale data."""
    snap = controller.status()
    if snap.get("fault"):
        raise RuntimeError(snap["fault"])
    pos = {j["name"]: j["position"] for j in snap["joints"]}
    joints = (*MOVING, "gripper")
    # Only the joints that carry the arm's weight must start straight up. Base, wrist roll and gripper
    # start wherever they are and move relative to that.
    off = [j for j in PLANAR if abs(pos[j] - air.home[j]) > HOME_TOLERANCE]
    if off:
        raise RuntimeError(f"put the arm straight up first ({', '.join(off)} off by "
                           + ", ".join(f"{(pos[j] - air.home[j]) / TICKS_PER_DEGREE:+.0f}°" for j in off) + ")")
    offset = {j: pos[j] - air.home[j] for j in RELATIVE}
    limits = air.tick_limits
    controller.set_speed("fast")
    stop = threading.Event()

    def motion_loop():
        while not stop.wait(0.02):
            controller.motion_step()

    threading.Thread(target=motion_loop, daemon=True).start()
    try:
        for j in joints:
            snap = controller.torque(j, True, controller.motion_epoch)
            if snap.get("fault"):
                raise RuntimeError(snap["fault"])
        last, t0, checked = None, time.monotonic(), 0.0
        while duration is None or time.monotonic() - t0 < duration:
            now = time.monotonic()
            targets = air.tick(now)
            if targets is not None:
                for j in RELATIVE:
                    lo, hi = limits[j]
                    targets[j] = int(min(max(targets[j] + offset[j], lo), hi))
                last = targets
            if last is not None:
                for j, t in last.items():
                    controller.target(j, t, controller.motion_epoch)
            else:
                controller.heartbeat(controller.motion_epoch)
            if now - checked > 0.5:
                checked = now
                snap = controller.status()
                if snap.get("fault"):
                    raise RuntimeError(snap["fault"])
                if show:
                    sys.stdout.write(air.line())
                    sys.stdout.flush()
            time.sleep(1.0 / rate_hz)
    finally:
        stop.set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="Kinesis stream URL")
    parser.add_argument("--port", help="arm USB serial port (needed with --execute)")
    parser.add_argument("--execute", action="store_true", help="drive the arm (default: dry run)")
    args = parser.parse_args(argv)
    cfg = json.loads(TELEOP_CONFIG.read_text())
    robot_map = json.loads(ROBOT_MAP.read_text())
    home = json.loads(HOME_FILE.read_text())["positions"]
    mapper = Mapper(cfg, robot_map, home)
    hand = band_hand(args.url)
    stream = BandStream(args.url)
    time.sleep(1.0)
    if stream.error or stream.latest_time() is None:
        print(f"No band data ({stream.error or 'nothing arriving'}). Check Kinesis and the stream URL.")
        return 1
    print(f"band on the {hand} wrist (from Kinesis). Height range {mapper.heights[0] * 100:.0f}–"
          f"{mapper.heights[-1] * 100:.0f} cm at {cfg['reach_m'] * 100:.0f} cm out.")
    print("\nFIST CALIBRATION (8 s):")
    try:
        meter = calibrate_fist(stream)
    except (ValueError, RuntimeError) as error:
        print(f"\nCalibration failed: {error}")
        return 1
    print(f"fist raises EMG {meter.ratio:.1f}x on ch{meter.channel + 1}.")
    print("\nHold your forearm LEVEL, pointing where you want the robot's middle to be. Starting in 2 s...")
    time.sleep(2)
    air = AirArm(stream, AirAim(hand), meter, mapper, home, robot_map)
    if not args.execute:
        print("DRY RUN: raise/lower, swing and twist your forearm, make a fist. Ctrl+C to quit.")
        try:
            while True:
                air.tick(time.monotonic())
                sys.stdout.write(air.line())
                sys.stdout.flush()
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nDry run ended; nothing moved.")
            return 0
    if not args.port:
        parser.error("--execute needs --port")
    print("\nThe arm will follow your forearm FAST, six motors at once. Start with it straight up, area clear.")
    if input("Type MOVE and press Enter to start: ").strip() != "MOVE":
        print("Not started.")
        return 1
    from .arm_web import ArmController
    from .leader_arm import make_bus
    controller = ArmController(make_bus(args.port), timeout=15.0)
    code = 0
    try:
        follow_band(controller, air)
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
