"""Check each robot joint's direction by moving it by hand, and save the result in robot_map.json.

Reads joint positions from the running read-only dashboard, so it never opens the arm's
serial port and never powers a motor. Stand BEHIND the robot, facing the same way it faces,
so the robot's left and right are your left and right.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from datetime import date
from pathlib import Path

from .kinesis import _consume_sse_line
from .robot_map import MAP_FILE

TICKS_PER_DEGREE = 4096 / 360
MIN_MOVE_TICKS = 80            # about 7 degrees

# joint: (what to do, how a positive tick change relates to the human signal's positive direction)
STEPS = {
    "shoulder_lift": ("Start with the arm straight up. Tilt the WHOLE ARM FORWARD (away from you), about 30°.",
                      "Raising your arm increases its lift angle; tilting the robot forward is the opposite."),
    "shoulder_pan": ("Turn the BASE so the arm swings to YOUR RIGHT, about 30°.",
                     "Your band arm moving outward (to your right) should swing the robot right."),
    "elbow_flex": ("Keep the upper arm still. BEND THE ELBOW FORWARD, about 30°.",
                   "Bending your elbow should bend the robot's elbow forward."),
    "wrist_flex": ("Keep the forearm still. BEND THE WRIST FORWARD, about 30°.",
                   "Your wrist bend (size only) will bend the robot's wrist forward."),
    "wrist_roll": ("TWIST THE WRIST CLOCKWISE as seen looking down the arm toward the base, about 45°.",
                   "Recorded for the robot; the band's roll direction still needs its own check."),
    "gripper": ("SQUEEZE / CLOSE the handle.",
                "Clenching your fist should close the handle."),
}


def sign_for(joint: str, delta: int, side: str = "right") -> int:
    """Map-file sign so that the human's positive direction produces this tick change."""
    moved = 1 if delta > 0 else -1
    if joint == "shoulder_lift":
        return -moved                  # human lift up = robot back toward straight up
    if joint == "shoulder_pan" and side == "left":
        return -moved                  # a left band arm moves outward to the left
    return moved


def check_move(start: dict, end: dict, joint: str) -> str | None:
    """None if the requested joint clearly moved most; otherwise what went wrong."""
    deltas = {j: end[j] - start[j] for j in start}
    if abs(deltas[joint]) < MIN_MOVE_TICKS:
        return (f"{joint} only moved {abs(deltas[joint]) / TICKS_PER_DEGREE:.0f}°. Move it further "
                "(if it is at its stop, move it toward the middle first, then retry).")
    biggest = max(deltas, key=lambda j: abs(deltas[j]))
    if biggest != joint and abs(deltas[biggest]) > abs(deltas[joint]):
        return f"{biggest} moved more than {joint}. Hold the other joints still and retry."
    return None


def apply_results(config: dict, results: dict, side: str) -> dict:
    today = date.today().isoformat()
    for joint, delta in results.items():
        spec = config["joints"][joint]
        if joint == "wrist_roll":
            spec["robot_clockwise_tick_sign"] = 1 if delta > 0 else -1
            spec["note"] = ("Robot direction checked by hand on " + today +
                            "; the band's roll direction is still unchecked, so this stays experimental.")
            continue
        spec["sign"] = sign_for(joint, delta, side)
        spec["verified"] = True
        spec["checked"] = f"by hand on {today}, band arm {side}"
    return config


class Feed:
    def __init__(self, url: str):
        self.url, self.latest, self.error = url, None, None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(urllib.request.Request(self.url, headers={"Accept": "text/event-stream"}),
                             timeout=5) as response:
                event, data = None, []
                for raw in response:
                    event, data, payload = _consume_sse_line(raw.decode("utf-8"), event, data)
                    if payload is not None and "positions" in payload:
                        self.latest = payload
                    elif payload is not None and payload.get("connected") is False:
                        self.error = payload.get("error", "arm not connected")
        except Exception as error:   # noqa: BLE001 - shown to the user
            self.error = str(error)

    def wait(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        while self.latest is None and time.monotonic() < deadline and not self.error:
            time.sleep(0.05)
        if self.latest is None:
            raise SystemExit(f"No arm data from the dashboard ({self.error or 'timeout'}). "
                             "Start it with scripts/dashboard --port … first.")


def live_until_enter(feed: Feed, joint: str) -> dict:
    stop = threading.Event()

    def show():
        while not stop.wait(0.15):
            p = feed.latest["positions"][joint]
            sys.stdout.write(f"\r   {joint}: {p:5d} ticks   (press Enter when you're holding the new position) ")
            sys.stdout.flush()

    threading.Thread(target=show, daemon=True).start()
    input()
    stop.set()
    return dict(feed.latest["positions"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dashboard", default="http://127.0.0.1:8768/events")
    parser.add_argument("--side", choices=("left", "right"), default="right", help="the arm wearing the band")
    parser.add_argument("--joints", nargs="*", choices=list(STEPS), default=list(STEPS))
    args = parser.parse_args(argv)
    feed = Feed(args.dashboard)
    feed.wait()
    print(__doc__.strip().split("\n\n", 1)[1])
    results = {}
    for n, joint in enumerate(args.joints, 1):
        what, why = STEPS[joint]
        while True:
            if any(j["torque_enabled"] for j in feed.latest["joints"].values()):
                raise SystemExit("A motor is powered. Turn all motors off before moving the arm by hand.")
            print(f"\n[{n}/{len(args.joints)}] {joint.replace('_', ' ').upper()}: {what}\n   ({why})")
            input("   Hold still at the starting position and press Enter… ")
            start = dict(feed.latest["positions"])
            print("   Now move it.")
            end = live_until_enter(feed, joint)
            problem = check_move(start, end, joint)
            if problem:
                print(f"\n   ✗ {problem}")
                continue
            delta = end[joint] - start[joint]
            results[joint] = delta
            print(f"\n   ✓ ticks went {'UP' if delta > 0 else 'DOWN'} by {abs(delta)} ({abs(delta) / TICKS_PER_DEGREE:.0f}°)")
            break
    config = apply_results(json.loads(MAP_FILE.read_text()), results, args.side)
    MAP_FILE.write_text(json.dumps(config, indent=2) + "\n")
    print("\nSaved to config/robot_map.json:")
    for joint, delta in results.items():
        spec = config["joints"][joint]
        extra = "" if joint != "wrist_roll" else "  (band roll direction still unchecked)"
        print(f"   {joint:14} sign {spec['sign']:+d}  verified {spec['verified']}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
