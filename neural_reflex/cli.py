"""Run with python3 -m neural_reflex --help."""

import argparse
import json
import math
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from .pipeline import (fit_decoder, load_rows, metrics, predict, save_json,
                       split_windows, validate_emg, window_features)


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
                if phases:
                    elapsed = (now_ns - start_ns) / 1e9
                    cycle_seconds = sum(seconds for _, seconds in phases)
                    round_id = int(elapsed // cycle_seconds)
                    if round_id >= args.rounds:
                        break
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
