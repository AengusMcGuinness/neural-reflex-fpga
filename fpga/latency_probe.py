"""Measure Kinesis delivery gaps and CPU feature time from a private capture.

This reports inter-arrival timing, not absolute sensor-to-host latency: the
band clock and Mac monotonic clock are not synchronized by this capture.
"""

import argparse
import json
import time

import numpy as np

from neural_reflex.pipeline import load_rows, window_features


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {"p50_ms": round(float(np.percentile(values, 50)), 3),
            "p95_ms": round(float(np.percentile(values, 95)), 3),
            "p99_ms": round(float(np.percentile(values, 99)), 3),
            "max_ms": round(float(values.max()), 3)}


def probe(path, iterations):
    rows = load_rows(path)
    batches = [row for row in rows if row.get("sample_index") == 0]
    if len(batches) < 2:
        raise ValueError("capture needs at least two Kinesis EMG batches")
    if len(rows) < 128:
        raise ValueError("capture needs at least 128 EMG samples")
    arrival = np.diff([row["t_ns"] for row in batches]) / 1e6
    band = np.diff([row["band_timestamp_us"] for row in batches]) / 1e3
    window = rows[:128]
    for _ in range(10):
        window_features(window, 128, 32)
    elapsed = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        window_features(window, 128, 32)
        elapsed.append((time.perf_counter_ns() - start) / 1e6)
    return {"batches": len(batches),
            "sensor_batch_interval": distribution(band),
            "mac_batch_arrival_gap": distribution(arrival),
            "arrival_gaps_under_1ms": int(np.count_nonzero(arrival < 1)),
            "arrival_gaps_total": len(arrival),
            "cpu_feature_call": distribution(elapsed),
            "note": "Arrival gaps reveal burstiness, not absolute Bluetooth latency."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="private Kinesis capture JSONL")
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    print(json.dumps(probe(args.input, args.iterations), indent=2))


if __name__ == "__main__":
    main()
