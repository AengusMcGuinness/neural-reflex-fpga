"""Pair conditioned EMG windows with pose angles."""

import numpy as np

from .signal import window_features


def _target_at(poses, label_time, joint_names, min_confidence, max_pose_gap_s):
    if not poses:
        return None
    for pose in poses:
        if abs(pose["host_time_s"] - label_time) <= 1e-6:
            if pose["confidence"] < min_confidence:
                return None
            return [float(pose["angles"][name]) for name in joint_names]
    times = [pose["host_time_s"] for pose in poses]
    after = next((index for index, time_s in enumerate(times) if time_s > label_time), None)
    if after is None or after == 0:
        return None
    before = after - 1
    span = times[after] - times[before]
    if span <= 0 or span > max_pose_gap_s:
        return None
    if (poses[before]["confidence"] < min_confidence or
            poses[after]["confidence"] < min_confidence):
        return None
    weight = (label_time - times[before]) / span
    return [float((1 - weight) * poses[before]["angles"][name] +
                  weight * poses[after]["angles"][name]) for name in joint_names]


def build_examples(samples, poses, window_samples, hop_samples, lags_hops,
                   label_delay_s, min_confidence, max_pose_gap_s, joint_names):
    signal = np.asarray([row["emg"] for row in samples], dtype=np.float64)
    times = np.asarray([row["host_time_s"] for row in samples], dtype=np.float64)
    valid = np.asarray([row["valid"] for row in samples], dtype=bool)
    ordered = sorted(poses, key=lambda pose: pose["host_time_s"])
    ends = list(range(window_samples - 1, len(samples), hop_samples))
    feature_rows = []
    for end in ends:
        start = end - window_samples + 1
        if valid[start:end + 1].all():
            feature_rows.append(window_features(signal[start:end + 1]))
        else:
            feature_rows.append(None)
    examples = []
    oldest = max(lags_hops)
    for position, end in enumerate(ends):
        if position < oldest:
            continue
        parts = []
        for lag in lags_hops:
            row = feature_rows[position - lag]
            if row is None:
                parts = None
                break
            parts.append(row)
        if parts is None:
            continue
        target = _target_at(ordered, float(times[end] + label_delay_s), joint_names,
                            min_confidence, max_pose_gap_s)
        if target is None:
            continue
        examples.append({"window_end_s": float(times[end]),
                         "features": np.concatenate(parts),
                         "target": target})
    return examples


def stack_examples(examples):
    if not examples:
        raise ValueError("no paired windows")
    features = np.stack([row["features"] for row in examples])
    targets = np.asarray([row["target"] for row in examples], dtype=np.float64)
    return features, targets
