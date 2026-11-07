"""Put EMG batches on the Mac clock and estimate how late the camera is."""

import numpy as np


def fit_band_clock(band_timestamp_us, host_time_s):
    band = np.asarray(band_timestamp_us, dtype=np.float64)
    host = np.asarray(host_time_s, dtype=np.float64)
    if len(band) < 2:
        raise ValueError("clock fit needs at least two EMG batches")
    slope, intercept = np.polyfit(band, host, 1)
    return float(slope), float(intercept)


def mark_sequence_gaps(records):
    last_sequence = {}
    marked = []
    for record in records:
        connection = record.get("band_connection_id")
        previous = last_sequence.get(connection)
        gap = previous is not None and record["sequence"] - previous > 1
        updated = dict(record)
        updated["gap_before"] = gap
        marked.append(updated)
        last_sequence[connection] = record["sequence"]
    return marked


def expand_batch(record, slope, intercept):
    base = record["band_timestamp_us"]
    interval = record["sample_interval_us"]
    samples = []
    for index, values in enumerate(record["values"]):
        band_us = base + index * interval
        samples.append({
            "host_time_s": slope * band_us + intercept,
            "band_timestamp_us": band_us,
            "emg": list(values),
            "sequence": record["sequence"],
            "valid": not (record.get("gap_before") and index == 0),
        })
    return samples


def estimate_pose_lag_s(emg_time_s, envelope, pose_time_s, pose_speed, search_s=0.3, step_s=0.01):
    """Return seconds to add to an EMG time to reach the matching pose sample."""
    emg_time_s = np.asarray(emg_time_s, dtype=np.float64)
    envelope = np.asarray(envelope, dtype=np.float64)
    pose_time_s = np.asarray(pose_time_s, dtype=np.float64)
    pose_speed = np.asarray(pose_speed, dtype=np.float64)
    best_lag = 0.0
    best_score = -np.inf
    lag = -search_s
    while lag <= search_s + step_s / 2:
        shifted = np.interp(emg_time_s + lag, pose_time_s, pose_speed, left=0.0, right=0.0)
        score = float(np.dot(envelope, shifted))
        if score > best_score:
            best_score = score
            best_lag = lag
        lag += step_s
    return float(best_lag)
