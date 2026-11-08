"""Text -> pen strokes in metres, from the Hershey single-stroke fonts.

The Hershey-Fonts package draws glyphs in integer font units with y pointing
DOWN (cap line at -12, baseline at +9 for the default ``futural`` font).  Here
the glyphs are flipped so y points up, scaled so the cap height equals
``height_m``, laid out left to right with the left edge at x = 0 and the
baseline at y = 0, and every polyline is resampled to at most
``max_spacing_m`` between consecutive points (2 mm by default) so the IK
solver can seed each waypoint from the previous one.
"""
from __future__ import annotations

import numpy as np

DEFAULT_FONT = "futural"
DEFAULT_HEIGHT_M = 0.03
MAX_SPACING_M = 0.002


def _load_font(font: str):
    from HersheyFonts import HersheyFonts

    hf = HersheyFonts()
    hf.load_default_font(font)
    return hf


def resample_polyline(points: np.ndarray, max_spacing_m: float = MAX_SPACING_M) -> np.ndarray:
    """Insert points so no two consecutive points are more than ``max_spacing_m`` apart."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return pts
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        length = float(np.linalg.norm(b - a))
        n = max(1, int(np.ceil(length / max_spacing_m - 1e-9)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
    return np.asarray(out, dtype=np.float64)


def text_to_strokes(text: str, height_m: float = DEFAULT_HEIGHT_M, font: str = DEFAULT_FONT,
                    max_spacing_m: float = MAX_SPACING_M) -> list[np.ndarray]:
    """Return a list of (K,2) arrays in metres: x to the right, y up, baseline y = 0."""
    if not text:
        return []
    hf = _load_font(font)
    opts = hf.render_options
    cap_units = float(opts["base_line"] - opts["cap_line"])
    if cap_units <= 0:
        raise ValueError(f"font {font!r} reports a non-positive cap height")
    scale = float(height_m) / cap_units
    raw = []
    for stroke in hf.strokes_for_text(text):
        pts = np.asarray(list(stroke), dtype=np.float64).reshape(-1, 2)
        if len(pts) < 2:
            continue
        xy = np.empty_like(pts)
        xy[:, 0] = pts[:, 0] * scale
        xy[:, 1] = (opts["base_line"] - pts[:, 1]) * scale      # flip: y up, baseline at 0
        raw.append(xy)
    if not raw:
        return []
    x0 = min(float(s[:, 0].min()) for s in raw)
    return [resample_polyline(s - np.array([x0, 0.0]), max_spacing_m) for s in raw]


def strokes_bbox(strokes: list[np.ndarray]) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) in metres over every point of every stroke."""
    if not strokes:
        return (0.0, 0.0, 0.0, 0.0)
    allpts = np.concatenate([np.asarray(s).reshape(-1, 2) for s in strokes])
    return (float(allpts[:, 0].min()), float(allpts[:, 1].min()),
            float(allpts[:, 0].max()), float(allpts[:, 1].max()))


def strokes_length(strokes: list[np.ndarray]) -> float:
    """Total pen-down path length in metres."""
    return float(sum(np.linalg.norm(np.diff(np.asarray(s), axis=0), axis=1).sum()
                     for s in strokes if len(s) > 1))


__all__ = ["text_to_strokes", "strokes_bbox", "strokes_length", "resample_polyline",
           "DEFAULT_FONT", "DEFAULT_HEIGHT_M", "MAX_SPACING_M"]
