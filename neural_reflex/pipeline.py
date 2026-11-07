"""Sensor-independent recording, feature extraction, and baseline decoder."""

import json
import math
from pathlib import Path

import numpy as np


CHANNELS = 8


def validate_emg(values):
    if not isinstance(values, list) or len(values) != CHANNELS:
        raise ValueError("emg must be an array of exactly eight raw channel values")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(value) for value in values):
        raise ValueError("emg values must be finite numbers")
    return values


def load_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                validate_emg(row["emg"])
                timestamp = row["t_ns"]
                if isinstance(timestamp, bool) or not isinstance(timestamp, int):
                    raise ValueError("t_ns must be an integer")
                if rows and timestamp <= rows[-1]["t_ns"]:
                    raise ValueError("t_ns must increase strictly")
                if "label" in row and row["label"] not in (0, 1):
                    raise ValueError("label must be 0 (rest) or 1 (pinch)")
                rows.append(row)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not rows:
        raise ValueError(f"{path}: no samples")
    return rows


def window_features(rows, window_samples, hop_samples, labeled=False):
    """Mean absolute deviation and waveform length for eight raw ADC channels."""
    if window_samples < 2 or hop_samples < 1:
        raise ValueError("window needs at least two samples and a positive hop")
    features, targets, rounds, starts = [], [], [], []
    for start in range(0, len(rows) - window_samples + 1, hop_samples):
        segment = rows[start:start + window_samples]
        if labeled:
            labels = {row.get("label") for row in segment}
            round_ids = {row.get("round") for row in segment}
            if len(labels) != 1 or None in labels or len(round_ids) != 1:
                continue
            targets.append(float(segment[0]["label"]))
            rounds.append(segment[0].get("round"))
        signal = np.asarray([row["emg"] for row in segment], dtype=np.float64)
        # Kinesis sends unsigned ADC counts, so remove each window's DC offset.
        mav = np.mean(np.abs(signal - np.mean(signal, axis=0)), axis=0)
        waveform_length = np.mean(np.abs(np.diff(signal, axis=0)), axis=0)
        features.append(np.concatenate((mav, waveform_length)))
        starts.append(start)
    if not features:
        raise ValueError("no complete usable windows; capture more labeled samples")
    return np.stack(features), np.asarray(targets), rounds, starts


def split_windows(rounds, starts, window_samples):
    """Hold out a whole round, or a non-overlapping chronological tail."""
    if len(set(rounds)) >= 2 and None not in rounds:
        held_out = rounds[-1]
        train = np.asarray([round_id != held_out for round_id in rounds])
        test = ~train
    else:
        cutoff = starts[max(1, int(len(starts) * 0.8)) - 1] + window_samples
        train = np.asarray([start + window_samples <= cutoff for start in starts])
        test = np.asarray([start >= cutoff for start in starts])
    if train.sum() < 2 or test.sum() < 1:
        raise ValueError("need more labeled samples or another complete calibration round")
    return train, test


def fit_decoder(features, targets, train_mask, regularization=1.0):
    train_x = features[train_mask]
    train_y = targets[train_mask]
    if len(set(train_y)) < 2:
        raise ValueError("training data needs both rest and pinch labels")
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    normalized = (train_x - mean) / scale
    augmented = np.column_stack((np.ones(len(normalized)), normalized))
    penalty = np.eye(augmented.shape[1]) * regularization
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(augmented.T @ augmented + penalty,
                              augmented.T @ train_y)
    return {"mean": mean.tolist(), "scale": scale.tolist(),
            "bias": float(weights[0]), "weights": weights[1:].tolist(),
            "features": "8x mean absolute deviation, then 8x mean waveform length"}


def predict(model, features):
    mean = np.asarray(model["mean"])
    scale = np.asarray(model["scale"])
    weights = np.asarray(model["weights"])
    if features.shape[-1] != 16 or len(mean) != 16 or len(scale) != 16 or len(weights) != 16:
        raise ValueError("model and window must each contain sixteen features")
    return np.clip((features - mean) / scale @ weights + model["bias"], 0, 1)


def metrics(targets, predictions):
    return {"windows": len(targets),
            "mae": float(np.mean(np.abs(targets - predictions))),
            "accuracy_at_0_5": float(np.mean((predictions >= 0.5) == targets))}


def save_json(path, value):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
