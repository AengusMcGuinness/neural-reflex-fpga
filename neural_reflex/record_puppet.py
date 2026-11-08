"""Record the leader arm's joint encoders while you move it by hand (torque off): exact labels for the decoder.

    ./scripts/record_puppet_block.sh SESSION_DIR BLOCK_NAME 'STREAM_URL' /dev/cu.YOUR_ARM_PORT

That script runs this recorder next to record-emg. This module only reads: it refuses to start if
any joint has torque on, and it never writes a register. Each row of arm_encoders.jsonl is
    {"host_time_s": monotonic seconds (same clock as Kinesis and the pose recorder),
     "block", "cue", "positions": {joint: raw tick}, "read_ms"}
The terminal walks you through a ~2.5 minute routine of slow sweeps and holds, which the robot can
follow (its motors cap at ~16-32 deg/s) and which give clean labels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .leader_arm import JOINTS, inspect, make_bus

# (cue, seconds, instruction). Start with the arm straight up (home pose).
ROUTINE = (
    ("ready", 5, "Hold the handle, arm straight up (home). Relax."),
    ("hold-home", 4, "Hold still at home."),
    ("shoulder-sweep", 20, "Slowly tilt the shoulder forward and back, elbow straight. ~4 s each way."),
    ("shoulder-holds", 15, "Shoulder: stop and hold 3 s at a few different angles."),
    ("elbow-sweep", 20, "Shoulder still. Slowly bend and straighten the elbow."),
    ("elbow-holds", 15, "Elbow: stop and hold 3 s at a few different bends."),
    ("reach", 30, "Slow reaches: move shoulder and elbow together to different points."),
    ("free-slow", 30, "Wander anywhere, slowly. Pause now and then."),
    ("hold-home-end", 5, "Back to home and hold still."),
)


def read_positions(bus) -> dict[str, int]:
    return {j: int(bus.read("Present_Position", j, normalize=False)) for j in JOINTS}


def record(bus, output: Path, block: str, rate_hz: float, routine=ROUTINE, log=print, clock=time.monotonic) -> int:
    states = inspect(bus)
    on = [j for j, s in states.items() if s.torque_enabled]
    if on:
        raise RuntimeError(f"torque is on for {', '.join(on)}; turn it off before puppet recording")
    period, count = 1.0 / rate_hz, 0
    with output.open("x", encoding="utf-8") as sink:
        for cue, seconds, instruction in routine:
            log(f"\n[{cue}] {instruction} ({seconds} s)")
            end = clock() + seconds
            while (now := clock()) < end:
                positions = read_positions(bus)
                done = clock()
                sink.write(json.dumps({"host_time_s": (now + done) / 2, "block": block, "cue": cue,
                                       "positions": positions, "read_ms": round(1000 * (done - now), 2)}) + "\n")
                count += 1
                left = int(end - done) + 1
                log(f"\r  {left:3d} s left  shoulder_lift {positions['shoulder_lift']:4d}  "
                    f"elbow_flex {positions['elbow_flex']:4d}   ", end="")
                time.sleep(max(0.0, period - (clock() - now)))
            sink.flush()
    log(f"\nWrote {count} encoder rows to {output}")
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", required=True, help="arm USB serial port")
    parser.add_argument("--output", required=True, help="arm_encoders.jsonl (must not exist)")
    parser.add_argument("--block", default="puppet")
    parser.add_argument("--rate", type=float, default=50.0, help="encoder reads per second")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bus = make_bus(args.port)
    bus.connect()
    try:
        record(bus, output, args.block, args.rate, log=lambda *a, **k: print(*a, **k, flush=True))
    except KeyboardInterrupt:
        print("\nStopped early; the rows so far are saved.")
    finally:
        bus.disconnect(disable_torque=False)   # read-only: leave torque exactly as it was
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
