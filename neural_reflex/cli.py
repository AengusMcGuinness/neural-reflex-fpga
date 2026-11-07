"""Run with python3 -m neural_reflex --help."""

import argparse
import json
import math
import random
import sys
import time
from collections import deque
from itertools import groupby
from pathlib import Path

import numpy as np

from .pipeline import (fit_decoder, load_rows, metrics, predict, save_json,
                       split_windows, validate_emg, window_features)
from .fpga_packet import pack_batch


def simulate(args):
    randomizer = random.Random(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    period_ns = round(1_000_000_000 / args.sample_rate)
    samples_per_phase = round(args.seconds_per_phase * args.sample_rate)
    with output.open("w", encoding="utf-8") as sink:
        for round_id in range(args.rounds):
            for label in (0, 1):
                for sample_id in range(samples_per_phase):
                    index = ((round_id * 2 + label) * samples_per_phase + sample_id)
                    amplitudes = [25 + (50 if label else 0) * (1 if channel < 4 else 0.45)
                                  for channel in range(8)]
                    sample = [round(randomizer.gauss(0, amplitude), 3)
                              for amplitude in amplitudes]
                    sink.write(json.dumps({"t_ns": index * period_ns, "emg": sample,
                                           "label": label, "round": round_id}) + "\n")
    print(f"Wrote {args.rounds * 2 * samples_per_phase} SYNTHETIC samples to {output}")


def parse_schedule(specification):
    phases = []
    for item in specification.split(","):
        try:
            name, seconds = item.split(":", 1)
            duration = float(seconds)
        except ValueError as error:
            raise ValueError("schedule format is rest:3,pinch:3") from error
        if name not in ("rest", "pinch") or not math.isfinite(duration) or duration <= 0:
            raise ValueError("schedule needs positive rest and pinch durations")
        phases.append((name, duration))
    if {name for name, _ in phases} != {"rest", "pinch"}:
        raise ValueError("schedule needs both rest and pinch phases")
    return phases


def capture(args):
    phases = parse_schedule(args.schedule) if args.schedule else None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    start_ns = None
    last_phase = None
    count = 0
    try:
        with output.open("w", encoding="utf-8") as sink:
            for line_number, line in enumerate(sys.stdin, 1):
                if not line.strip():
                    continue
                try:
                    incoming = json.loads(line)
                    emg = validate_emg(incoming["emg"])
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f"stdin:{line_number}: {error}") from error
                now_ns = time.monotonic_ns()
                if start_ns is None:
                    start_ns = now_ns
                row = {"t_ns": now_ns, "emg": emg}
                if "t_ns" in incoming:
                    row["source_t_ns"] = incoming["t_ns"]
                for field in ("band_timestamp_us", "sample_index", "batch_sequence",
                              "host_time_s", "stream_session_id", "band_connection_id"):
                    if field in incoming:
                        row[field] = incoming[field]
                if phases:
                    elapsed = (now_ns - start_ns) / 1e9
                    cycle_seconds = sum(seconds for _, seconds in phases)
                    round_id = int(elapsed // cycle_seconds)
                    if round_id >= args.rounds:
                        continue  # drain the upstream reader instead of breaking its pipe
                    position = elapsed % cycle_seconds
                    for name, seconds in phases:
                        if position < seconds:
                            break
                        position -= seconds
                    row["label"] = 0 if name == "rest" else 1
                    row["round"] = round_id
                    if (round_id, name) != last_phase:
                        print(f"Round {round_id + 1}/{args.rounds}: {name.upper()}",
                              file=sys.stderr, flush=True)
                        last_phase = (round_id, name)
                sink.write(json.dumps(row) + "\n")
                count += 1
    except KeyboardInterrupt:
        pass
    print(f"Captured {count} raw EMG samples to {output}", file=sys.stderr)


def inspect(args):
    rows = load_rows(args.input)
    intervals = np.diff([row["t_ns"] for row in rows]) / 1e9
    rates = 1 / intervals if len(intervals) else np.array([])
    result = {"samples": len(rows),
              "duration_seconds": round((rows[-1]["t_ns"] - rows[0]["t_ns"]) / 1e9, 3),
              "median_arrival_rate_hz": round(float(np.median(rates)), 2) if len(rates) else None,
              "labeled_samples": sum("label" in row for row in rows),
              "channel_min": np.min([row["emg"] for row in rows], axis=0).tolist(),
              "channel_max": np.max([row["emg"] for row in rows], axis=0).tolist()}
    print(json.dumps(result, indent=2))


def train(args):
    rows = load_rows(args.input)
    window_samples = round(args.sample_rate * args.window_ms / 1000)
    hop_samples = round(args.sample_rate * args.hop_ms / 1000)
    features, targets, rounds, starts = window_features(rows, window_samples,
                                                         hop_samples, labeled=True)
    train_mask, test_mask = split_windows(rounds, starts, window_samples)
    model = fit_decoder(features, targets, train_mask, args.regularization)
    model.update({"sample_rate_hz": args.sample_rate,
                  "window_samples": window_samples, "hop_samples": hop_samples,
                  "target": "pinch=1, rest=0", "algorithm": "ridge regression"})
    report = {"train": metrics(targets[train_mask], predict(model, features[train_mask])),
              "held_out": metrics(targets[test_mask], predict(model, features[test_mask]))}
    save_json(args.output, model)
    print(json.dumps(report, indent=2))
    print(f"Saved model to {args.output}")


def replay(args):
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    rows = load_rows(args.input)
    features, targets, _, _ = window_features(rows, model["window_samples"],
                                               model["hop_samples"], labeled=False)
    predictions = predict(model, features)
    for index, value in enumerate(predictions):
        filled = round(float(value) * 20)
        print(f"{index:04d} |{'#' * filled}{'.' * (20 - filled)}| "
              f"pinch={value:.2f}")
        if args.live:
            time.sleep(model["hop_samples"] / model["sample_rate_hz"])
    print(f"Replayed {len(predictions)} windows; output is a virtual gripper only")


def live(args):
    """Apply a trained CPU decoder to incoming raw EMG; optionally record it."""
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    window_samples = model["window_samples"]
    hop_samples = model["hop_samples"]
    recent = deque(maxlen=window_samples)
    output = None
    if args.capture:
        path = Path(args.capture)
        path.parent.mkdir(parents=True, exist_ok=True)
        output = path.open("w", encoding="utf-8")
    count = 0
    try:
        for line_number, line in enumerate(sys.stdin, 1):
            if not line.strip():
                continue
            try:
                incoming = json.loads(line)
                emg = validate_emg(incoming["emg"])
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"stdin:{line_number}: {error}") from error
            now_ns = time.monotonic_ns()
            row = {"t_ns": now_ns, "emg": emg}
            if "t_ns" in incoming:
                row["source_t_ns"] = incoming["t_ns"]
            for field in ("band_timestamp_us", "sample_index", "batch_sequence",
                          "host_time_s", "stream_session_id", "band_connection_id"):
                if field in incoming:
                    row[field] = incoming[field]
            if output:
                output.write(json.dumps(row) + "\n")
            recent.append(row)
            count += 1
            if len(recent) == window_samples and (count - window_samples) % hop_samples == 0:
                features, _, _, _ = window_features(list(recent), window_samples,
                                                     hop_samples)
                value = float(predict(model, features)[0])
                filled = round(value * 20)
                print(f"\rvirtual gripper |{'#' * filled}{'.' * (20 - filled)}| "
                      f"pinch={value:.2f}", end="", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if output:
            output.close()
        print(f"\nProcessed {count} raw samples; CPU baseline only", file=sys.stderr)


def normalize_kinesis(args):
    """Flatten Axo Kinesis schema-v1 EMG batches into raw per-sample JSONL."""
    count = 0
    connection_id = None
    for line_number, line in enumerate(sys.stdin, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if event["schema_version"] != 1 or not isinstance(event["payload"], dict):
                raise ValueError("unsupported Kinesis event schema")
            kind = event["type"]
            if kind == "hello":
                connection_id = event["payload"].get("status", {}).get("band_connection_id")
            elif kind == "status":
                connection_id = event["payload"].get("band_connection_id")
            elif kind == "emg":
                payload = event["payload"]
                values = payload["values"]
                if not isinstance(values, list) or not values:
                    raise ValueError("EMG batch is empty or malformed")
                for sample_index, sample in enumerate(values):
                    validate_emg(sample)
                    row = {"emg": sample, "band_timestamp_us": event["band_timestamp_us"],
                           "sample_index": sample_index,
                           "batch_sequence": payload["sequence"],
                           "host_time_s": event["host_time_s"],
                           "stream_session_id": event["stream_session_id"],
                           "band_connection_id": connection_id}
                    print(json.dumps(row, separators=(",", ":")), flush=True)
                    count += 1
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"stdin:{line_number}: {error}") from error
    print(f"Normalized {count} Kinesis raw EMG samples", file=sys.stderr)


def packetize(args):
    """Turn a complete Kinesis capture into fixed-size USB PipeIn packets."""
    rows = load_rows(args.input)
    epochs = {row.get("band_connection_id") for row in rows}
    if len(epochs) != 1 or None in epochs:
        raise ValueError("capture must contain one known band connection epoch")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    batches = 0
    key = lambda row: (row.get("batch_sequence"), row.get("band_timestamp_us"))
    previous_sequence = None
    with output.open("wb") as sink:
        for (sequence, timestamp), grouped in groupby(rows, key):
            group = list(grouped)
            if sequence is None or timestamp is None:
                raise ValueError("input is missing Kinesis batch metadata")
            if [row.get("sample_index") for row in group] != list(range(16)):
                raise ValueError(f"batch {sequence} is incomplete or out of order")
            if previous_sequence is not None and (sequence - previous_sequence) % (2**32) != 1:
                raise ValueError(f"gap or repeated batch after sequence {previous_sequence}")
            sink.write(pack_batch(sequence, timestamp, [row["emg"] for row in group]))
            batches += 1
            previous_sequence = sequence
    print(f"Packed {batches} batches ({batches * 16} raw samples) to {output}")


def session_start(args):
    from .session import create_session
    path = create_session(args.root, args.subject, args.wrist)
    print(path)
    print(f"scripts/record_block.sh {path} clench 'PASTE_STREAM_URL'")


def preflight(args):
    from .pose import ensure_hand_model
    from .preflight import report
    try:
        ensure_hand_model(args.model)
    except OSError as error:
        print(f"could not download hand landmarker: {error}", file=sys.stderr)
    ready, lines = report(args.model)
    print("\n".join(lines))
    if not ready:
        raise SystemExit(2)


def record_emg(args):
    import io
    import socket
    import urllib.request
    from .kinesis import assert_local_stream_url, iter_emg_records
    url = assert_local_stream_url(args.url)
    socket.setdefaulttimeout(5)
    request = urllib.request.Request(
        url, headers={"Accept": "text/event-stream", "Cache-Control": "no-cache"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    motion_count = 0
    motion_path = Path(args.motion_output) if args.motion_output else None
    motion_sink = motion_path.open("w", encoding="utf-8") if motion_path else None

    def on_motion(event):
        nonlocal motion_count
        event["block"] = args.block
        motion_sink.write(json.dumps(event) + "\n")
        motion_count += 1

    try:
        with opener.open(request, timeout=5) as response, output.open("w", encoding="utf-8") as sink:
            for record in iter_emg_records(io.TextIOWrapper(response, encoding="utf-8"),
                                           on_motion if motion_sink else None):
                record["block"] = args.block
                sink.write(json.dumps(record) + "\n")
                sink.flush()
                if motion_sink:
                    motion_sink.flush()
                count += 1
    except (KeyboardInterrupt, TimeoutError, socket.timeout):
        pass
    finally:
        if motion_sink:
            motion_sink.close()
    print(f"Wrote {count} raw EMG batches to {output}", file=sys.stderr)
    if motion_path:
        print(f"Wrote {motion_count} orientation/gyro/gesture events to {motion_path}", file=sys.stderr)


def record_arm(args):
    from .arm_pose import record_arm_pose
    count = record_arm_pose(args.output, args.camera, args.block, args.model, args.side)
    print(f"Wrote {count} arm pose frames to {args.output}", file=sys.stderr)


def record_pose(args):
    from .pose import record_pose as capture_pose
    count = capture_pose(args.output, args.camera, args.block, args.model)
    print(f"Wrote {count} pose frames to {args.output}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Raw EMG capture and virtual gripper baseline")
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser("simulate", help="make clearly synthetic test data")
    synthetic.add_argument("--output", default="data/raw/synthetic.jsonl")
    synthetic.add_argument("--sample-rate", type=int, default=200)
    synthetic.add_argument("--seconds-per-phase", type=float, default=2.0)
    synthetic.add_argument("--rounds", type=int, default=4)
    synthetic.add_argument("--seed", type=int, default=7)
    synthetic.set_defaults(func=simulate)
    recorder = commands.add_parser("capture", help="read one raw 8-channel JSON sample per stdin line")
    recorder.add_argument("--output", required=True)
    recorder.add_argument("--schedule", help="optional rough labels, e.g. rest:3,pinch:3")
    recorder.add_argument("--rounds", type=int, default=4)
    recorder.set_defaults(func=capture)
    probe = commands.add_parser("inspect", help="summarize a raw recording")
    probe.add_argument("--input", required=True)
    probe.set_defaults(func=inspect)
    learner = commands.add_parser("train", help="fit a CPU baseline on labeled raw EMG")
    learner.add_argument("--input", required=True)
    learner.add_argument("--output", default="data/models/decoder.json")
    learner.add_argument("--sample-rate", type=float, required=True,
                         help="actual sample rate reported by the band SDK")
    learner.add_argument("--window-ms", type=float, default=100)
    learner.add_argument("--hop-ms", type=float, default=20)
    learner.add_argument("--regularization", type=float, default=1.0)
    learner.set_defaults(func=train)
    player = commands.add_parser("replay", help="show virtual gripper from recorded EMG")
    player.add_argument("--input", required=True)
    player.add_argument("--model", required=True)
    player.add_argument("--live", action="store_true", help="pace output at the model's hop rate")
    player.set_defaults(func=replay)
    controller = commands.add_parser("live", help="show live CPU virtual gripper from stdin")
    controller.add_argument("--model", required=True)
    controller.add_argument("--capture", help="optional local JSONL path for raw recording")
    controller.set_defaults(func=live)
    normalizer = commands.add_parser("kinesis-normalize",
                                    help="flatten Axo Kinesis event JSONL to raw eight-channel samples")
    normalizer.set_defaults(func=normalize_kinesis)
    packetizer = commands.add_parser("packetize", help="pack recorded Kinesis batches for FPGA PipeIn")
    packetizer.add_argument("--input", required=True)
    packetizer.add_argument("--output", default="data/private/fpga_packets.bin")
    packetizer.set_defaults(func=packetize)
    starter = commands.add_parser("session", help="create an empty recording session")
    starter.add_argument("--subject", required=True)
    starter.add_argument("--wrist", required=True, choices=("left", "right"))
    starter.add_argument("--root", default="data/sessions")
    starter.set_defaults(func=session_start)
    checker = commands.add_parser("preflight", help="check recording dependencies")
    checker.add_argument("--model", default="models/hand_landmarker.task")
    checker.set_defaults(func=preflight)
    emg_recorder = commands.add_parser("record-emg", help="save raw Kinesis EMG batches")
    emg_recorder.add_argument("--url", required=True)
    emg_recorder.add_argument("--output", required=True)
    emg_recorder.add_argument("--block", default="unspecified")
    emg_recorder.add_argument("--motion-output",
                              help="also save band orientation, gyro and gestures to this JSONL")
    emg_recorder.set_defaults(func=record_emg)
    pose_recorder = commands.add_parser("record-pose", help="save webcam hand angles")
    pose_recorder.add_argument("--output", required=True)
    pose_recorder.add_argument("--block", default="unspecified")
    pose_recorder.add_argument("--camera", type=int, default=0)
    pose_recorder.add_argument("--model", default="models/hand_landmarker.task")
    pose_recorder.set_defaults(func=record_pose)
    arm_recorder = commands.add_parser("record-arm", help="save webcam arm angles (MediaPipe Pose)")
    arm_recorder.add_argument("--output", required=True)
    arm_recorder.add_argument("--block", default="unspecified")
    arm_recorder.add_argument("--camera", type=int, default=0)
    arm_recorder.add_argument("--side", choices=("left", "right"), default="right",
                              help="the arm wearing the band")
    arm_recorder.add_argument("--model", default="models/pose_landmarker_full.task")
    arm_recorder.set_defaults(func=record_arm)
    args = parser.parse_args()
    try:
        if hasattr(args, "sample_rate") and args.sample_rate <= 0:
            raise ValueError("sample rate must be positive")
        if hasattr(args, "rounds") and args.rounds < 1:
            raise ValueError("rounds must be positive")
        if hasattr(args, "window_ms") and (args.window_ms <= 0 or args.hop_ms <= 0):
            raise ValueError("window and hop must be positive")
        if hasattr(args, "regularization") and args.regularization < 0:
            raise ValueError("regularization must be nonnegative")
        args.func(args)
    except (ValueError, OSError, KeyError, np.linalg.LinAlgError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
