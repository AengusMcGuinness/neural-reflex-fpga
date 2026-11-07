"""Causal EMG conditioning and the 40-feature window used by the decoder."""

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, tf2sos

from .config import BANDPASS_HZ, NOTCH_HARMONICS, NOTCH_HZ, NOTCH_Q


def design_sos(sample_rate):
    low, high = BANDPASS_HZ
    sos = butter(4, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    notches = []
    for harmonic in NOTCH_HARMONICS:
        frequency = NOTCH_HZ * harmonic
        if frequency >= sample_rate * 0.45:
            continue
        numerator, denominator = iirnotch(frequency, NOTCH_Q, fs=sample_rate)
        notches.append(tf2sos(numerator, denominator))
    if notches:
        sos = np.vstack([sos, *notches])
    return sos


def condition_emg(emg, sample_rate, valid=None):
    signal = np.asarray(emg, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] != 8:
        raise ValueError("emg must have shape (samples, 8)")
    if valid is None:
        valid = np.ones(len(signal), dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
    output = np.zeros_like(signal)
    sos = design_sos(sample_rate)
    padded = np.concatenate(([False], valid, [False]))
    changes = np.diff(padded.astype(np.int8))
    for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        if end - start < 8:
            continue
        output[start:end] = sosfilt(sos, signal[start:end], axis=0)
    return output


def window_features(window):
    segment = np.asarray(window, dtype=np.float64)
    if segment.ndim != 2 or segment.shape[1] != 8 or segment.shape[0] < 3:
        raise ValueError("window must have at least three samples of eight channels")
    mean_absolute = np.mean(np.abs(segment), axis=0)
    waveform_length = np.mean(np.abs(np.diff(segment, axis=0)), axis=0)
    rms = np.sqrt(np.mean(segment ** 2, axis=0))
    signs = np.sign(segment)
    signs[signs == 0] = 1.0
    zero_crossings = np.mean(signs[1:] * signs[:-1] < 0, axis=0)
    slopes = np.sign(np.diff(segment, axis=0))
    slopes[slopes == 0] = 1.0
    slope_changes = np.mean(slopes[1:] * slopes[:-1] < 0, axis=0)
    return np.concatenate((mean_absolute, waveform_length, rms, zero_crossings, slope_changes))


def lagged_features(rows, index, lags_hops):
    parts = []
    for lag in lags_hops:
        source = index - lag
        if source < 0:
            return None
        parts.append(np.asarray(rows[source], dtype=np.float64))
    return np.concatenate(parts)
