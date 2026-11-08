"""Descriptive channel-by-feature associations with camera-derived arm angles.

The block-bootstrap intervals describe resampling of this recording's intact
acquisition segments and rounds. They are not population confidence intervals.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .emg_transfer_stats import block_summary, bootstrap_weights, temporal_blocks


CHANNELS = 8
FEATURES = 12
TARGETS = 3


def _arrays(pair):
    features = np.asarray(pair["features"], dtype=float)
    target = np.asarray(pair["y"], dtype=float)
    if features.ndim != 3 or features.shape[1:] != (CHANNELS, FEATURES):
        raise ValueError("features must have shape (windows, 8, 12)")
    n = len(features)
    if target.shape != (n, TARGETS):
        raise ValueError("y must have shape (windows, 3)")
    cues = np.asarray(pair["cues"], dtype=str)
    if cues.shape != (n,):
        raise ValueError("cues must have one value per window")
    return features, target, cues


def _pearson(x, y):
    """Pearson r for already pairwise-masked finite vectors."""
    if len(x) < 3:
        return np.nan
    xc = x - x.mean()
    yc = y - y.mean()
    xx, yy = xc @ xc, yc @ yc
    # Repeated decimal values can acquire tiny nonzero variance on centering.
    if np.sqrt(xx / len(x)) <= 1e-12 or np.sqrt(yy / len(y)) <= 1e-12:
        return np.nan
    return float((xc @ yc) / np.sqrt(xx * yy))


def _within_cue(x, y, cues):
    """Remove cue means using exactly the rows valid for this one pair."""
    xr, yr = x.copy(), y.copy()
    for cue in np.unique(cues):
        at = cues == cue
        xr[at] -= x[at].mean()
        yr[at] -= y[at].mean()
    return _pearson(xr, yr)


def correlation_summary(pair):
    """Return 8×12×3 Pearson, Spearman, and cue-demeaned correlations.

    Every channel/feature/target pair has its own finite mask. An undefined
    coefficient remains NaN for the caller's JSON-safe conversion to null.
    """
    features, target, cues = _arrays(pair)
    shape = (CHANNELS, FEATURES, TARGETS)
    pearson = np.full(shape, np.nan)
    spearman = np.full(shape, np.nan)
    within = np.full(shape, np.nan)
    counts = np.zeros(shape, dtype=int)
    for channel in range(CHANNELS):
        for feature in range(FEATURES):
            x = features[:, channel, feature]
            for joint in range(TARGETS):
                y = target[:, joint]
                valid = np.isfinite(x) & np.isfinite(y)
                counts[channel, feature, joint] = int(valid.sum())
                if valid.sum() < 3:
                    continue
                a, b, c = x[valid], y[valid], cues[valid]
                pearson[channel, feature, joint] = _pearson(a, b)
                spearman[channel, feature, joint] = _pearson(rankdata(a), rankdata(b))
                within[channel, feature, joint] = _within_cue(a, b, c)
    with np.errstate(invalid="ignore"):
        target_std = np.nanstd(target, axis=0)
    return {"windows": len(features), "pearson_r": pearson,
            "spearman_rho": spearman, "cue_demeaned_r": within,
            "pair_counts": counts, "target_std_deg": target_std}


def _block_cue_summaries(x, y, cues, blocks):
    """Six sufficient statistics for each block and cue, after pairwise masking."""
    cue_values = np.unique(cues)
    sums = np.zeros((len(blocks), len(cue_values), 6), dtype=float)
    valid_all = np.isfinite(x) & np.isfinite(y)
    # Center once before x² - (Σx)²/n. This avoids cancellation when raw values
    # have a large offset; weighted cue means still change on every draw.
    origin_x = x[valid_all].mean() if valid_all.any() else 0.0
    origin_y = y[valid_all].mean() if valid_all.any() else 0.0
    for bi, block in enumerate(blocks):
        valid = np.isfinite(x[block]) & np.isfinite(y[block])
        for ci, cue in enumerate(cue_values):
            at = block[valid & (cues[block] == cue)]
            if not len(at):
                continue
            a, b = x[at] - origin_x, y[at] - origin_y
            sums[bi, ci] = (len(at), a.sum(), b.sum(), a @ a, b @ b, a @ b)
    return sums


def _coefficients_from_summaries(sums):
    """Pool centered cue-specific sums separately in every bootstrap draw."""
    n, sx, sy, sxx, syy, sxy = np.moveaxis(sums, -1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        covariance = np.where(n > 0, sxy - sx * sy / n, 0).sum(axis=-1)
        xx = np.where(n > 0, sxx - sx * sx / n, 0).sum(axis=-1)
        yy = np.where(n > 0, syy - sy * sy / n, 0).sum(axis=-1)
        denominator = np.sqrt(np.maximum(xx, 0) * np.maximum(yy, 0))
        coefficient = np.where((n.sum(axis=-1) >= 3) & (denominator > 1e-10),
                               covariance / denominator, np.nan)
    return coefficient


def _bootstrap_pair_coefficients(x, y, cues, blocks, weights):
    """Overall and within-cue r for given whole-block bootstrap weights.

    Cue means are recomputed from weighted sufficient statistics on each draw;
    residuals from the original recording are never reused as fixed values.
    """
    x, y, cues = np.asarray(x, float), np.asarray(y, float), np.asarray(cues, dtype=str)
    summaries = _block_cue_summaries(x, y, cues, blocks)
    weighted = np.einsum("rb,bck->rck", weights, summaries)
    overall = _coefficients_from_summaries(weighted.sum(axis=1, keepdims=True))
    within = _coefficients_from_summaries(weighted)
    return overall, within


def _interval(values):
    finite = values[np.isfinite(values)]
    return np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else [None, None]


def selected_intervals(pair, selections, replicates=1000, seconds=(2, 5, 10)):
    """Descriptive pointwise intervals for fixed channel-feature-target pairs.

    Whole wall-time blocks are resampled within acquisition segment and round,
    using the same draw for all selected pairs at a given block size.
    """
    features, target, cues = _arrays(pair)
    n = len(features)
    if n < 2:
        raise ValueError("at least two paired windows are required")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    for key in ("times", "segments", "rounds"):
        if np.asarray(pair[key]).shape != (n,):
            raise ValueError(f"{key} must have one value per window")
    parsed = []
    for selection in selections:
        identity = {key: selection[key] for key in
                    ("label", "channel", "feature_index", "target_index")}
        for key, upper in (("channel", CHANNELS), ("feature_index", FEATURES),
                           ("target_index", TARGETS)):
            value = identity[key]
            if type(value) is not int or not 0 <= value < upper:
                raise ValueError(f"{key} must be an integer from 0 to {upper - 1}")
        parsed.append(identity)
    result = {}
    for block_seconds in seconds:
        blocks, strata = temporal_blocks(pair["times"], pair["segments"],
                                         pair["rounds"], block_seconds)
        weights = bootstrap_weights(strata, replicates=replicates)
        rows = []
        for identity in parsed:
            channel, feature, joint = (identity[key] for key in
                                       ("channel", "feature_index", "target_index"))
            x, y = features[:, channel, feature], target[:, joint]
            valid = np.isfinite(x) & np.isfinite(y)
            observed = _pearson(x[valid], y[valid])
            observed_within = _within_cue(x[valid], y[valid], cues[valid]) if valid.sum() >= 3 else np.nan
            bootstrap, bootstrap_within = _bootstrap_pair_coefficients(x, y, cues, blocks, weights)
            rows.append({**identity, "pearson_r": observed,
                         "pearson_ci95": _interval(bootstrap),
                         "cue_demeaned_r": observed_within,
                         "cue_demeaned_ci95": _interval(bootstrap_within)})
        result[str(block_seconds)] = {"blocks": block_summary(pair["times"], blocks, strata,
                                                               block_seconds), "pairs": rows}
    return result
