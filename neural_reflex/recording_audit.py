"""Aggregate recording quality without disclosing raw samples or stream identities."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


SEQ_MODULUS = 2**32
ARM_METRICS = ("shoulder_lift", "shoulder_pan", "elbow_flex", "wrist_flex")
KNOWN_CUES = frozenset({"ready", "sync", "rest", "lift_forward", "sweep_side",
                        "elbow", "wrist_flex", "wrist_roll", "free", "home", "rest_end"})
KNOWN_MOTION = frozenset({"gyro", "orientation", "gesture"})


def _rows(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{line_number}: expected an object")
            yield row


def _number(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _integer(value, field: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{field} must be an integer from {lower} to {upper}")
    return value


def _percentiles(values, *, multiplier=1.0):
    if not values:
        return None
    ordered = sorted(values)

    def at(fraction):
        position = (len(ordered) - 1) * fraction
        low = math.floor(position)
        high = math.ceil(position)
        return round((ordered[low] + (ordered[high] - ordered[low]) * (position - low))
                     * multiplier, 3)

    return {"min": at(0), "p05": at(0.05), "median": at(0.5),
            "p95": at(0.95), "max": at(1)}


def _intervals(times):
    return [later - earlier for earlier, later in zip(times, times[1:])]


def _duration(times):
    return round(times[-1] - times[0], 3) if len(times) > 1 else 0.0


def audit_block(block: Path) -> dict:
    """Summarize one recording folder, including camera-only folders.

    Stream and band IDs are used only to count epochs and are never returned.
    The report contains no raw EMG, landmarks, subject, or capability URL.
    """
    block = Path(block)
    if not block.is_dir():
        raise ValueError("recording block directory does not exist")

    emg_times, band_intervals, arm_times, motion_times = [], [], [], []
    rates = Counter()
    session_ids, connection_ids, epochs = set(), set(), set()
    last_sequence = {}
    sequence = {"gap_boundaries": 0, "missing_batches_estimate": 0,
                "duplicates": 0, "backsteps_or_resets": 0, "wraps": 0,
                "epoch_transitions": 0}
    previous_epoch = previous_band_time = None
    unknown_connection_batches = 0
    emg_batches = emg_samples = 0
    for row in _rows(block / "emg.jsonl"):
        t = _number(row.get("host_time_s"), "EMG host_time_s")
        band_t = _integer(row.get("band_timestamp_us"), "band_timestamp_us", 0, 2**64 - 1)
        seq = _integer(row.get("sequence"), "EMG sequence", 0, SEQ_MODULUS - 1)
        rate = _integer(row.get("sample_rate_hz"), "sample_rate_hz", 1, 100_000)
        values = row.get("values")
        if not isinstance(values, list) or len(values) != 16:
            raise ValueError("EMG batch must contain 16 sample rows")
        for sample in values:
            if not isinstance(sample, list) or len(sample) != 8:
                raise ValueError("EMG sample must contain eight channels")
            for value in sample:
                _integer(value, "EMG ADC count", 0, 65535)
        stream_id, connection_id = row.get("stream_session_id"), row.get("band_connection_id")
        epoch = (stream_id, connection_id)
        if previous_epoch is not None and epoch != previous_epoch:
            sequence["epoch_transitions"] += 1
        if epoch == previous_epoch and previous_band_time is not None:
            band_intervals.append(band_t - previous_band_time)
        previous_epoch = epoch
        previous_band_time = band_t
        if connection_id is None:
            unknown_connection_batches += 1
        session_ids.add(stream_id)
        connection_ids.add(connection_id)
        epochs.add(epoch)
        if epoch in last_sequence:
            previous = last_sequence[epoch]
            delta = (seq - previous) % SEQ_MODULUS
            if delta == 0:
                sequence["duplicates"] += 1
            elif delta >= SEQ_MODULUS // 2:
                sequence["backsteps_or_resets"] += 1
            else:
                if seq < previous:
                    sequence["wraps"] += 1
                if delta > 1:
                    sequence["gap_boundaries"] += 1
                    sequence["missing_batches_estimate"] += delta - 1
        last_sequence[epoch] = seq
        emg_times.append(t)
        rates[rate] += 1
        emg_batches += 1
        emg_samples += len(values)

    angles = defaultdict(list)
    cue_times = defaultdict(list)
    side_validity = defaultdict(Counter)
    pose_frames = world_landmark_frames = 0
    for row in _rows(block / "arm.jsonl"):
        t = _number(row.get("host_time_s"), "arm host_time_s")
        arm_times.append(t)
        pose_frames += 1
        if row.get("world_landmarks") is not None:
            world_landmark_frames += 1
        cue = row.get("cue")
        cue_times[cue if isinstance(cue, str) and cue in KNOWN_CUES else "other"].append(t)
        listed = row.get("angles")
        if listed is not None:
            if not isinstance(listed, dict):
                raise ValueError("arm angles must be an object or null")
            for name, value in listed.items():
                if name in ARM_METRICS and value is not None:
                    angles[name].append(_number(value, "arm angle"))
        sides = row.get("angles_by_side")
        if sides is not None:
            if not isinstance(sides, dict):
                raise ValueError("angles_by_side must be an object")
            for side, side_angles in sides.items():
                if side not in ("left", "right") or side_angles is None:
                    continue
                if not isinstance(side_angles, dict):
                    raise ValueError("side angles must be an object or null")
                for name, value in side_angles.items():
                    if name in ARM_METRICS and value is not None:
                        _number(value, "side angle")
                        side_validity[side][name] += 1

    motion_counts = Counter()
    for row in _rows(block / "motion.jsonl"):
        motion_times.append(_number(row.get("host_time_s"), "motion host_time_s"))
        kind = row.get("type")
        motion_counts[kind if kind in KNOWN_MOTION else "other"] += 1

    span_endpoints = [(times[0], times[-1]) for times in (emg_times, arm_times, motion_times) if times]
    duration_s = round(max(end for _, end in span_endpoints) -
                       min(start for start, _ in span_endpoints), 3) if span_endpoints else 0.0
    overlap_s = (round(max(0.0, min(emg_times[-1], arm_times[-1]) -
                           max(emg_times[0], arm_times[0])), 3)
                 if emg_times and arm_times else 0.0)
    pose_intervals = _intervals(arm_times)
    emg_intervals = _intervals(emg_times)
    selected_rate = next(iter(rates)) if len(rates) == 1 else None
    return {
        "duration_s": duration_s,
        "overlap_s": overlap_s,
        "camera_only": bool(pose_frames and not emg_batches),
        "emg_batches": emg_batches,
        "emg_samples": emg_samples,
        "emg_duration_s": _duration(emg_times),
        "source_sample_rate_hz": selected_rate,
        "source_sample_rate_counts": {str(rate): count for rate, count in sorted(rates.items())},
        "effective_sample_rate_hz": (round(emg_samples / _duration(emg_times), 2)
                                     if _duration(emg_times) > 0 else None),
        "epochs": {"stream_sessions": len(session_ids), "band_connections": len(connection_ids),
                   "session_connection_pairs": len(epochs),
                   "unknown_connection_batches": unknown_connection_batches},
        "sequence_anomalies": sequence,
        "band_batch_interval_us": _percentiles(band_intervals),
        "host_arrival": {"interval_ms": _percentiles(emg_intervals, multiplier=1000),
                         "zero_interval_fraction": (round(sum(dt == 0 for dt in emg_intervals) /
                                                          len(emg_intervals), 4) if emg_intervals else None),
                         "backsteps": sum(dt < 0 for dt in emg_intervals)},
        "pose_frames": pose_frames,
        "pose_duration_s": _duration(arm_times),
        "pose_fps": (round((pose_frames - 1) / _duration(arm_times), 2)
                     if pose_frames > 1 and _duration(arm_times) > 0 else None),
        "world_landmark_frames": world_landmark_frames,
        "pose_gap": {"interval_ms": _percentiles(pose_intervals, multiplier=1000),
                     "over_100ms": sum(dt > 0.1 for dt in pose_intervals),
                     "over_200ms": sum(dt > 0.2 for dt in pose_intervals),
                     "backsteps": sum(dt < 0 for dt in pose_intervals)},
        "per_metric_validity": {name: {"valid_frames": len(angles[name]),
                                       "fraction": (round(len(angles[name]) / pose_frames, 4)
                                                    if pose_frames else 0.0)}
                                for name in ARM_METRICS},
        "angle_distributions": {name: _percentiles(angles[name])
                                for name in ARM_METRICS},
        "side_metric_validity": {side: dict(sorted(counts.items()))
                                 for side, counts in sorted(side_validity.items())},
        "cues": {str(cue): {"frames": len(times), "span_s": _duration(times)}
                 for cue, times in sorted(cue_times.items(), key=lambda item: str(item[0]))},
        "motion_events": dict(sorted(motion_counts.items())),
        "motion_duration_s": _duration(motion_times),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("block", type=Path, help="folder containing emg.jsonl, arm.jsonl, motion.jsonl")
    parser.add_argument("--output", type=Path, help="write aggregate report to this JSON file")
    args = parser.parse_args(argv)
    report = audit_block(args.block)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
