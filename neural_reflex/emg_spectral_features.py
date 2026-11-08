"""Causal, gap-aware EMG spectra paired with the existing camera angle labels."""

from __future__ import annotations

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

from .signal import design_sos


FS = 2048
BANDS = ((20, 80), (80, 150), (150, 250), (250, 450))
FEATURE_NAMES = (
    "log_rms", "log_power_20_80", "log_power_80_150",
    "log_power_150_250", "log_power_250_450",
    "relative_power_20_80", "relative_power_80_150",
    "relative_power_150_250", "relative_power_250_450",
    "mean_frequency", "median_frequency", "spectral_entropy",
)


def periodograms(windows, fs=FS):
    """Periodic-Hann, demeaned, one-sided power spectral density in ADC²/Hz.

    Input has shape (examples, samples, 8); output frequency axis is one-sided.
    Integrating the PSD over all bins gives the Hann-weighted demeaned power.
    """
    windows = np.asarray(windows, dtype=float)
    if windows.ndim != 3 or windows.shape[2] != 8 or windows.shape[1] < 4:
        raise ValueError("windows must have shape (examples, at least 4 samples, 8)")
    if not np.isfinite(windows).all() or not np.isfinite(fs) or fs <= 0:
        raise ValueError("windows and sampling frequency must be finite")
    count = windows.shape[1]
    freqs = np.fft.rfftfreq(count, d=1 / fs)
    if len(windows) == 0:
        return freqs, np.empty((0, len(freqs), 8))
    hann = np.hanning(count + 1)[:-1]  # periodic Hann, not symmetric window
    centered = windows - windows.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered * hann[None, :, None], axis=1)
    psd = np.abs(spectrum) ** 2 / (fs * np.sum(hann**2))
    psd[:, 1:-1 if count % 2 == 0 else None] *= 2
    return freqs, psd


def features_from_windows(windows):
    """Return examples × channels × 12 absolute and gain-invariant features.

    The first five are log1p of unweighted raw RMS and rectangular PSD band
    integrals. Relative powers and spectral shape are NaN when 20–450 Hz power
    is exactly zero, while absolute zero-power features stay valid.
    """
    windows = np.asarray(windows, dtype=float)
    freqs, psd = periodograms(windows, FS)
    if len(windows) == 0:
        return np.empty((0, 8, len(FEATURE_NAMES)))
    df = freqs[1] - freqs[0]
    rms = np.sqrt(np.mean(windows**2, axis=1))
    masks = [(freqs >= low) & ((freqs < high) if high < 450 else (freqs <= high))
             for low, high in BANDS]
    bands = np.stack([np.sum(psd[:, mask], axis=1) * df for mask in masks], axis=2)
    total = bands.sum(axis=2)
    relative = np.divide(bands, total[:, :, None], out=np.full_like(bands, np.nan),
                         where=total[:, :, None] > 0)
    in_range = (freqs >= 20) & (freqs <= 450)
    selected = psd[:, in_range]
    selected_freqs = freqs[in_range]
    if len(selected_freqs) < 2:
        raise ValueError("spectral window is too short for 20–450 Hz shape features")
    mean = np.divide(np.sum(selected * selected_freqs[None, :, None], axis=1) * df, total,
                     out=np.full_like(total, np.nan), where=total > 0)
    cumulative = np.cumsum(selected, axis=1)
    median_indices = np.argmax(cumulative >= total[:, None, :] / (2 * df), axis=1)
    median = selected_freqs[median_indices].astype(float)
    median[total == 0] = np.nan
    probabilities = np.divide(selected, selected.sum(axis=1, keepdims=True),
                              out=np.zeros_like(selected), where=total[:, None, :] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probabilities > 0, probabilities * np.log(probabilities), 0)
    entropy = -np.sum(terms, axis=1) / np.log(len(selected_freqs))
    entropy[total == 0] = np.nan
    return np.concatenate((np.log1p(rms)[:, :, None], np.log1p(bands), relative,
                           mean[:, :, None], median[:, :, None], entropy[:, :, None]), axis=2)


def paired_spectral_segments(segments, window_samples=512, hop_samples=41,
                             warmup_samples=2048, label_delay_s=.05):
    """Reset filtering at every acquisition segment; pair trailing spectra to pose.

    Each output feature sees only samples at or before ``times``. Camera labels
    use the benchmark's existing angle geometry and fixed label offset; camera
    interpolation gaps longer than 150 ms are excluded.
    """
    if (window_samples < 4 or hop_samples < 1 or warmup_samples < 0 or
            not np.isfinite(label_delay_s)):
        raise ValueError("invalid spectral pairing configuration")
    parts, coverage = [], []
    sos = design_sos(FS)
    for segment in segments:
        times = np.asarray(segment["times"], float)
        raw = np.asarray(segment["raw"], float)
        pose_times = np.asarray(segment["pose_times"], float)
        pose_targets = np.asarray(segment["pose_targets"], float)
        if (raw.shape != (len(times), 8) or pose_targets.shape != (len(pose_times), 3) or
                not np.isfinite(raw).all() or np.any(np.diff(times) <= 0) or
                np.any(np.diff(pose_times) <= 0)):
            raise ValueError("invalid segment samples or camera labels")
        lo = max(times[0], pose_times[0])
        hi = min(times[-1] + 1 / FS, pose_times[-1])
        index = np.flatnonzero((times >= lo) & (times < hi))
        candidate = np.arange(warmup_samples + window_samples - 1, len(index), hop_samples)
        entry = {"segment": int(segment["segment_id"]), "raw_samples": len(raw),
                 "candidate_windows": len(candidate), "paired_windows": 0,
                 "label_excluded_windows": len(candidate)}
        if not len(candidate):
            coverage.append(entry)
            continue
        local_raw, local_times = raw[index], times[index]
        zi = sosfilt_zi(sos)[:, :, None] * local_raw[0][None, None, :]
        filtered, _ = sosfilt(sos, local_raw, axis=0, zi=zi)
        at = local_times[candidate] + label_delay_s
        after = np.searchsorted(pose_times, at)
        bounded = (after > 0) & (after < len(pose_times))
        after = np.clip(after, 1, len(pose_times) - 1)
        before = after - 1
        span = pose_times[after] - pose_times[before]
        valid = (bounded & (pose_times[before] >= lo) & (pose_times[after] < hi) &
                 (span <= .15))
        weight = (at - pose_times[before]) / span
        labels = (1 - weight[:, None]) * pose_targets[before] + weight[:, None] * pose_targets[after]
        valid &= np.isfinite(labels[:, :2]).all(axis=1)
        ends, before, after, labels = candidate[valid], before[valid], after[valid], labels[valid]
        entry["paired_windows"] = len(ends)
        entry["label_excluded_windows"] -= len(ends)
        coverage.append(entry)
        if not len(ends):
            continue
        feature_blocks = []
        for chunk in np.array_split(ends, max(1, int(np.ceil(len(ends) / 256)))):
            windows = np.stack([filtered[end - window_samples + 1:end + 1] for end in chunk])
            feature_blocks.append(features_from_windows(windows))
        features = np.concatenate(feature_blocks)
        starts = ends - window_samples + 1
        rail = ((local_raw <= 100) | (local_raw >= 65435)).astype(int)
        cumsum = np.vstack((np.zeros((1, 8), int), np.cumsum(rail, axis=0)))
        clipped = (cumsum[ends + 1] - cumsum[starts]).sum(axis=1) / (window_samples * 8)
        rounds = segment.get("rounds", np.full(len(pose_times), None))
        parts.append({"features": features, "y": labels, "times": local_times[ends],
                      "input_start": local_times[starts], "label_end": pose_times[after],
                      "cues": np.asarray(segment["cues"])[before],
                      "rounds": np.array([int(rounds[i]) if rounds[i] is not None else 0 for i in before]),
                      "segments": np.full(len(ends), segment["segment_id"], int),
                      "clip_fraction": clipped})
    if not parts:
        raise ValueError("no spectral windows have valid shoulder and elbow labels")
    keys = ("features", "y", "times", "input_start", "label_end", "cues", "rounds",
            "segments", "clip_fraction")
    merged = {key: np.concatenate([part[key] for part in parts]) for key in keys}
    if np.any(np.diff(merged["times"]) <= 0):
        raise ValueError("mapped segment times overlap or run backward")
    merged["segment_coverage"] = coverage
    return merged
