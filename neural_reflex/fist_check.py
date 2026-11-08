"""20-second band check before recording: does EMG clearly respond when you make a fist?

Prompts you in the terminal: relax, fist, relax, fist (no keys to press). Reads the live Kinesis
stream meanwhile and reports, per EMG channel, fist strength / relaxed strength. Also checks the
band stayed on one connection, the stream rate, and saturation. Exits 1 if the band would not
give usable grip data, so record_arm_block.sh stops before recording. Writes fist_check.json.
"""

from __future__ import annotations

import argparse
import io
import json
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE_HZ
from .kinesis import assert_local_stream_url, iter_emg_records
from .signal import condition_emg

PHASES = (("relax", "Relax your band hand completely, resting on the table", 4),
          ("fist", "Make a FIRM FIST and hold it", 3),
          ("relax", "Relax again", 4),
          ("fist", "FIRM FIST again, hold it", 3),
          ("relax", "Relax", 3))
PASS_RATIO = 3.0           # best channel's fist / relaxed strength
WEAK_RATIO = 1.8
MAX_RAIL_PCT = 5.0
MIN_RATE = 100             # EMG batches per second (Kinesis sends ~128)


def evaluate(records: list[dict], phases: list[tuple[str, float, float]]) -> dict:
    """records: EMG batches with host_time_s; phases: (kind, start, end) on the same clock."""
    if len(records) < 50:
        return {"ok": False, "problems": ["no EMG arrived: is the local sensor feed on in Kinesis?"]}
    problems = []
    connections = {r.get("band_connection_id") for r in records}
    if len(connections) > 1:
        problems.append("the band reconnected during the check: keep it close and try again")
    t_batch = np.array([r["host_time_s"] for r in records])
    rate = len(records) / max(t_batch[-1] - t_batch[0], 1e-9)
    if rate < MIN_RATE:
        problems.append(f"EMG is arriving at only {rate:.0f} batches/s (expected about 128)")
    raw = np.array([row for r in records for row in r["values"]], dtype=float)
    t = np.repeat(t_batch, 16) + np.tile(np.arange(16) / SAMPLE_RATE_HZ, len(records))
    rail = 100 * ((raw < 256) | (raw > 65279)).mean(axis=0)
    emg = condition_emg(raw, SAMPLE_RATE_HZ)
    strength = {}
    for kind in ("relax", "fist"):
        mask = np.zeros(len(t), dtype=bool)
        for k, start, end in phases:
            if k == kind:
                # Skip the first 0.7 s of each phase: you are still reacting to the prompt.
                mask |= (t >= start + 0.7) & (t <= end)
        if mask.sum() < SAMPLE_RATE_HZ:
            problems.append(f"not enough EMG during the {kind} phases")
            return {"ok": False, "problems": problems}
        strength[kind] = np.sqrt(np.mean(emg[mask] ** 2, axis=0))
    ratio = strength["fist"] / np.maximum(strength["relax"], 1e-9)
    order = [int(i) for i in np.argsort(-ratio)]
    best = float(ratio[order[0]])
    saturated = [c + 1 for c in range(8) if rail[c] > MAX_RAIL_PCT]
    if saturated:
        problems.append(f"channel(s) {saturated} saturated: tighten the band so every electrode touches skin")
    if best < WEAK_RATIO:
        problems.append(f"no clear fist response (best channel {best:.1f}x): tighten or reseat the band, "
                        "make a firmer fist, and check Kinesis shows the band connected")
    verdict = "good" if best >= PASS_RATIO else "weak" if best >= WEAK_RATIO else "none"
    return {"ok": not problems and best >= WEAK_RATIO, "verdict": verdict, "best_ratio": best,
            "best_channels": [c + 1 for c in order[:3]], "ratio_by_channel": [round(float(r), 2) for r in ratio],
            "relaxed_rms": [round(float(v), 1) for v in strength["relax"]],
            "fist_rms": [round(float(v), 1) for v in strength["fist"]],
            "rail_pct": [round(float(v), 2) for v in rail], "batches_per_s": round(rate, 1),
            "connections": len(connections), "problems": problems}


def band_hand(url: str, timeout: float = 3.0) -> str | None:
    """Which wrist Kinesis says the band is on ('left'/'right'), from the stream's hello event."""
    from .kinesis import _consume_sse_line
    request = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            event, data = None, []
            for raw in response:
                event, data, payload = _consume_sse_line(raw.decode("utf-8"), event, data)
                if payload and payload.get("type") in ("hello", "status"):
                    status = payload.get("payload", {})
                    return (status.get("status") or status).get("hand")
    except (OSError, socket.timeout, ValueError):
        return None
    return None


def collect(url: str, records: list, stop: threading.Event, errors: list, motion: list | None = None):
    request = urllib.request.Request(url, headers={"Accept": "text/event-stream", "Cache-Control": "no-cache"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=5) as response:
            on_motion = motion.append if motion is not None else None
            for record in iter_emg_records(io.TextIOWrapper(response, encoding="utf-8"), on_motion):
                records.append(record)
                if stop.is_set():
                    return
    except (OSError, socket.timeout, ValueError) as error:
        errors.append(str(error))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="Kinesis stream URL")
    parser.add_argument("--output", help="where to save fist_check.json (e.g. the block folder)")
    parser.add_argument("--side", choices=("left", "right"),
                        help="the arm this recording tracks; fails if Kinesis says the band is on the other wrist")
    args = parser.parse_args(argv)
    url = assert_local_stream_url(args.url)
    hand = band_hand(url)
    records, errors, stop, motion = [], [], threading.Event(), []
    reader = threading.Thread(target=collect, args=(url, records, stop, errors, motion), daemon=True)
    reader.start()
    print("\nBAND CHECK (about 20 s), here in the Terminal. Follow the prompts; no keys to press.")
    print("The camera window with the target arm opens AFTER this check passes.\n")
    time.sleep(1.0)
    if errors or not records:
        print(f"Can't read the band stream ({errors[0] if errors else 'nothing arrived'}). "
              "Is Kinesis open with the local sensor feed on, and is the stream URL current?")
        return 1
    phases = []
    for kind, text, seconds in PHASES:
        start = time.monotonic()                     # same clock as Kinesis host_time_s
        for left in range(seconds, 0, -1):
            sys.stdout.write(f"\r  {'>>> ' if kind == 'fist' else '    '}{text}  {left} ")
            sys.stdout.flush()
            time.sleep(max(0.0, start + seconds - left + 1 - time.monotonic()))
        phases.append((kind, start, time.monotonic()))
        print()
    stop.set()
    result = evaluate(list(records), phases)
    kinds = {m.get("type") for m in motion}
    if not {"orientation", "gyro"} <= kinds:
        missing = sorted({"orientation", "gyro"} - kinds)
        result["problems"].append(f"Kinesis is not sending band {' or '.join(missing)}: turn 'local sensor feed' "
                                  "off and on in Kinesis (or reconnect the band), then copy the new stream URL")
        result["ok"] = False
    result["motion_events"] = len(motion)
    result["band_hand"] = hand
    if args.side and hand and hand != args.side:
        result["problems"].append(f"Kinesis says the band is on your {hand.upper()} wrist, but this recording tracks "
                                  f"your {args.side.upper()} arm. Move the band, or record with '{hand}' as the side "
                                  f"(and set the wrist in Kinesis if it is wrong)")
        result["ok"] = False
    print()
    if "best_ratio" in result:
        print(f"Fist response: {result['best_ratio']:.1f}x on the strongest channel "
              f"(ch{result['best_channels'][0]}; next ch{result['best_channels'][1]}, ch{result['best_channels'][2]})  "
              f"-> {result['verdict'].upper()}")
        print("  per channel: " + "  ".join(f"ch{c + 1} {r:.1f}x" for c, r in enumerate(result["ratio_by_channel"])))
        print(f"  stream {result['batches_per_s']:.0f} batches/s, band connections: {result['connections']}, "
              f"Kinesis says band on: {hand or 'unknown'} wrist")
    for problem in result["problems"]:
        print(f"  ✗ {problem}")
    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        (Path(args.output) / "fist_check.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["ok"]:
        print("\n✓ Band looks good." + ("" if result["verdict"] == "good" else
              " The fist response is weak: make firm fists during the recording."))
        return 0
    print("\nFix the problem above and run the check again before recording.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
