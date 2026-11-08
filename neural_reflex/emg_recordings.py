"""Load EMG recordings without joining samples across loss or reconnection.

The camera labels retain the benchmark's original angle definitions. Segments
share those labels but have separate raw arrays and independently mapped clocks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .emg_benchmark import CONFIG, label_values, read_rows


SEQUENCE_MODULUS = 2**32
BATCH_SAMPLES = 16
TIMESTAMP_TOLERANCE_S = 2e-6


def _validate_batches(rows):
    if len(rows) < 2:
        raise ValueError("at least two EMG batches are required")
    raw = np.asarray([row.get("values") for row in rows], dtype=float)
    if (raw.shape != (len(rows), BATCH_SAMPLES, 8) or not np.isfinite(raw).all()
            or np.any((raw < 0) | (raw > 65535)) or np.any(raw != np.floor(raw))):
        raise ValueError("expected unsigned 16 by 8 ADC batches")
    if any(type(row.get("sequence")) is not int or
           not 0 <= row["sequence"] < SEQUENCE_MODULUS for row in rows):
        raise ValueError("invalid EMG sequence number")
    if any(row.get("sample_rate_hz") != CONFIG["sample_rate"] for row in rows):
        raise ValueError("unsupported EMG sample rate")
    band = np.asarray([row["band_timestamp_us"] for row in rows], dtype=float) / 1e6
    host = np.asarray([row["host_time_s"] for row in rows], dtype=float)
    interval = np.asarray([row["sample_interval_us"] for row in rows], dtype=float) / 1e6
    if (not np.isfinite(band).all() or not np.isfinite(host).all() or
            not np.isfinite(interval).all() or np.any(np.diff(host) < 0) or
            not np.allclose(interval, 1 / CONFIG["sample_rate"], atol=1e-9)):
        raise ValueError("invalid EMG acquisition or host clock")
    return raw, band, host, interval


def _clock_for_epoch(band, host, clock_mode):
    """Fit only receipt/device timestamps; report the OLS candidate in either mode."""
    if clock_mode not in ("ols", "unit"):
        raise ValueError("clock_mode must be 'ols' or 'unit'")
    if len(band) == 1:
        clock = {"fit": "single-batch unit slope; host receipt anchor", "slope": 1.0,
                 "receipt_residual_p05_p50_p95_ms": [0.0, 0.0, 0.0],
                 "candidate_ols_slope": None,
                 "candidate_ols_residual_p05_p50_p95_ms": None}
        return host.copy(), 1.0, clock
    if band[-1] <= band[0]:
        raise ValueError("clock epoch needs increasing timestamps")
    fit = band < band[0] + .6 * (band[-1] - band[0])
    if fit.sum() < 2:
        fit[:2] = True
    relative_band, relative_host = band[fit] - band[0], host[fit] - host[0]
    ols_slope, ols_offset = np.polyfit(relative_band, relative_host, 1)
    ols_residual = host - (host[0] + ols_offset + ols_slope * (band - band[0]))
    if clock_mode == "unit":
        slope = 1.0
        offset = float(np.median(relative_host - relative_band))
        method = "unit slope; first 60% host receipt median offset"
    elif .99 < ols_slope < 1.01:
        slope, offset = ols_slope, ols_offset
        method = "first 60% of EMG device time; host receipt timestamps"
    else:
        slope = 1.0
        offset = float(np.median(relative_host - relative_band))
        method = "unit slope fallback; implausible first 60% OLS receipt drift"
    residual = host - (host[0] + offset + slope * (band - band[0]))
    clock = {"fit": method, "slope": float(slope),
             "receipt_residual_p05_p50_p95_ms":
             (np.percentile(residual, [5, 50, 95]) * 1000).tolist(),
             "candidate_ols_slope": float(ols_slope),
             "candidate_ols_residual_p05_p50_p95_ms":
             (np.percentile(ols_residual, [5, 50, 95]) * 1000).tolist()}
    return host[0] + offset + slope * (band - band[0]), slope, clock


def load_segments(block, side="right", clock_mode="ols"):
    """Return ``(segments, metadata)`` in acquisition order.

    A segment ends at a sequence loss, device clock discontinuity, or connection
    transition. No missing ADC samples are synthesized. A clock fit is shared by
    all contiguous segments in one connection/clock epoch. Each segment's
    ``times`` and ``raw`` align one-to-one; all pose fields refer to the full
    camera block and must be restricted by the caller's pairing interval.
    """
    block = Path(block)
    if clock_mode not in ("ols", "unit"):
        raise ValueError("clock_mode must be 'ols' or 'unit'")
    rows, pose = read_rows(block / "emg.jsonl"), read_rows(block / "arm.jsonl")
    raw, band, host, interval = _validate_batches(rows)
    if len(pose) < 2:
        raise ValueError("at least two pose frames are required")
    pose_times = np.asarray([row["host_time_s"] for row in pose], float)
    if not np.isfinite(pose_times).all() or np.any(np.diff(pose_times) <= 0):
        raise ValueError("pose timestamps must be finite and increasing")
    pose_targets = np.asarray([label_values(row, side) for row in pose])
    cues = np.asarray([row.get("cue") or "unlabelled" for row in pose])
    rounds = np.asarray([row.get("round") for row in pose], dtype=object)
    steps = np.asarray([(row.get("target") or {}).get("step", row.get("cue")) for row in pose], dtype=object)
    spots = np.asarray([(row.get("target") or {}).get("spot") for row in pose], dtype=object)

    # Tuple identity is compared only in memory; reports expose ordinal epochs.
    keys = [(row.get("stream_session_id"), row.get("band_connection_id")) for row in rows]
    segment_starts, clock_starts = [0], [0]
    boundaries = []
    missing = 0
    for i in range(1, len(rows)):
        changed = keys[i] != keys[i - 1]
        reasons = ["connection_transition"] if changed else []
        lost = 0
        clock_break = changed
        if not changed:
            delta = (rows[i]["sequence"] - rows[i - 1]["sequence"]) % SEQUENCE_MODULUS
            if delta == 0 or delta >= SEQUENCE_MODULUS // 2:
                raise ValueError(f"duplicate or backward EMG sequence at batch {i}")
            if delta > 1:
                lost = delta - 1
                missing += lost
                reasons.append("sequence_gap")
            expected = delta * BATCH_SAMPLES * interval[i - 1]
            if band[i] <= band[i - 1] or abs((band[i] - band[i - 1]) - expected) > TIMESTAMP_TOLERANCE_S:
                reasons.append("device_clock_discontinuity")
                clock_break = True
        if reasons:
            segment_starts.append(i)
            boundaries.append({"after_batch": i - 1, "before_batch": i,
                               "reasons": reasons, "missing_batches_estimate": lost,
                               "missing_sample_duration_s": lost * BATCH_SAMPLES / CONFIG["sample_rate"],
                               "host_gap_s": float(host[i] - host[i - 1]),
                               "device_gap_s": None if changed else float(band[i] - band[i - 1])})
        if clock_break:
            clock_starts.append(i)
    segment_starts.append(len(rows))
    clock_starts.append(len(rows))

    batch_times = np.empty(len(rows), float)
    clocks = []
    for clock_id, (a, b) in enumerate(zip(clock_starts, clock_starts[1:])):
        batch_times[a:b], slope, clock = _clock_for_epoch(band[a:b], host[a:b], clock_mode)
        clocks.append({"clock_epoch": clock_id, "batch_start": a, "batch_end": b,
                       "batch_count": b - a, **clock})

    segments, summaries = [], []
    first_host = float(host[0])
    for segment_id, (a, b) in enumerate(zip(segment_starts, segment_starts[1:])):
        clock_id = int(np.searchsorted(clock_starts, a, side="right") - 1)
        slope = clocks[clock_id]["slope"]
        sample_times = (batch_times[a:b, None] +
                        slope * interval[a:b, None] * np.arange(BATCH_SAMPLES)).ravel()
        segment_raw = raw[a:b].reshape(-1, 8)
        near_rail = ((segment_raw <= 100) | (segment_raw >= 65435)).sum(axis=0)
        segments.append({"raw": segment_raw, "times": sample_times,
                         "pose_times": pose_times, "pose_targets": pose_targets,
                         "cues": cues, "rounds": rounds, "steps": steps, "spots": spots,
                         "clock": clocks[clock_id], "side": side,
                         "segment_id": segment_id, "clock_epoch": clock_id})
        summaries.append({"segment_id": segment_id, "clock_epoch": clock_id,
                          "batch_count": b - a, "sample_count": len(segment_raw),
                          "start_s": float(sample_times[0] - first_host),
                          "end_s": float(sample_times[-1] - first_host),
                          "duration_s": float(sample_times[-1] - sample_times[0]),
                          "near_rail_per_channel": near_rail.astype(int).tolist()})
    metadata = {"batch_count": len(rows), "sample_count": len(rows) * BATCH_SAMPLES,
                "clock_mode": clock_mode,
                "retained_sample_duration_s": len(rows) * BATCH_SAMPLES / CONFIG["sample_rate"],
                "elapsed_host_s": float(host[-1] - host[0]),
                "pose_frames": len(pose), "segment_count": len(segments),
                "connection_epoch_count": 1 + sum("connection_transition" in b["reasons"] for b in boundaries),
                "clock_epoch_count": len(clocks), "sequence_missing_batches": missing,
                "sequence_missing_duration_s": missing * BATCH_SAMPLES / CONFIG["sample_rate"],
                "boundaries": boundaries, "segments": summaries, "clocks": clocks,
                "near_rail_per_channel": np.sum([s["near_rail_per_channel"] for s in summaries], axis=0).astype(int).tolist()}
    return segments, metadata
