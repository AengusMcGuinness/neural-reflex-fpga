"""Create an empty recording session before the band is connected."""

import json
from datetime import datetime
from pathlib import Path

from .config import HOP_MS, JOINT_NAMES, LABEL_DELAY_S, SAMPLE_RATE_HZ, WINDOW_MS


def create_session(root, subject, wrist):
    if wrist not in ("left", "right"):
        raise ValueError("wrist must be left or right")
    path = Path(root) / datetime.now().strftime("%Y%m%d-%H%M%S")
    path.mkdir(parents=True)
    (path / "blocks").mkdir()
    manifest = {
        "subject": subject,
        "wrist": wrist,
        "joints": list(JOINT_NAMES),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "window_ms": WINDOW_MS,
        "hop_ms": HOP_MS,
        "label_delay_s": LABEL_DELAY_S,
        "batch_timestamp_aligns_to": "first_sample",
        "protocol": [
            "Begin with four sharp fist clenches, then two seconds of rest.",
            "Record separate 1-2 minute blocks: single-finger flexion, thumb-to-finger pinches, fist and open hand, wrist movement, different speeds, random movement, and rest.",
            "Change arm posture between blocks.",
            "Take the band off and put it back on for the held-out session.",
        ],
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
