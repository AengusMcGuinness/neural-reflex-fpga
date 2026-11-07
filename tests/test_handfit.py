"""End-to-end check of handfit on a synthetic block written in the recorders' formats."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from neural_reflex.angles import joint_angles
from neural_reflex.handfit import fit, overlay_data, pair_block, write_viewer
from neural_reflex.pose import pose_row

FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16),
           "pinky": (17, 18, 19, 20)}


def synthetic_hand(curls):
    """21 world landmarks (metres) for a flat hand whose fingers curl by the given degrees."""
    points = np.zeros((21, 3))
    points[1:5] = [[-0.03, -0.02, 0], [-0.045, -0.04, 0], [-0.055, -0.06, 0], [-0.06, -0.075, 0]]
    for column, (finger, chain) in enumerate(FINGERS.items()):
        base = np.array([-0.02 + 0.015 * column, -0.08, 0.0])
        points[chain[0]] = base
        direction, position = 0.0, base
        for joint in chain[1:]:
            direction += np.radians(curls[finger])
            position = position + 0.025 * np.array([0, -np.cos(direction), -np.sin(direction)])
            points[joint] = position
    return points


def write_block(folder: Path, seconds=60.0, seed=3):
    rng = np.random.default_rng(seed)
    rate, batch = 2048, 16
    t = np.arange(int(seconds * rate)) / rate
    curls = {f: 35 * (1 + np.sin(2 * np.pi * (0.13 + 0.07 * i) * t + i)) for i, f in enumerate(FINGERS)}
    gains = np.zeros((len(t), 8))
    for i, f in enumerate(FINGERS):           # finger i drives channels 2i and 2i+1
        gains[:, 2 * i] += curls[f] / 70
        gains[:, 2 * i + 1] += curls[f] / 140
    emg = 32768 + (40 + 400 * gains) * rng.standard_normal((len(t), 8))
    folder.mkdir(parents=True)
    with (folder / "emg.jsonl").open("w") as out:
        for k, start in enumerate(range(0, len(t) - batch + 1, batch)):
            out.write(json.dumps({
                "host_time_s": 1000 + t[start] + 0.004 * rng.random(), "band_connection_id": "x",
                "band_timestamp_us": int(t[start] * 1e6), "event_sequence": k, "sequence": k,
                "sample_interval_us": 1e6 / rate, "sample_rate_hz": rate,
                "values": emg[start:start + batch].round().astype(int).tolist(), "block": "synthetic"}) + "\n")
    with (folder / "pose.jsonl").open("w") as out:
        for frame_time in np.arange(0, seconds, 1 / 30):
            i = int(frame_time * rate)
            landmarks = synthetic_hand({f: curls[f][i] for f in FINGERS})
            row = pose_row(1000 + frame_time + 0.08, "synthetic", landmarks.tolist(), 0.95)
            out.write(json.dumps(row) + "\n")


class HandfitTests(unittest.TestCase):
    def test_synthetic_hand_angles_follow_curl(self):
        straight = joint_angles(synthetic_hand({f: 0 for f in FINGERS}))
        bent = joint_angles(synthetic_hand({f: 60 for f in FINGERS}))
        self.assertLess(straight["index_pip"], 1)
        self.assertAlmostEqual(bent["index_pip"], 60, delta=1)

    def test_decoder_learns_finger_curl_from_emg(self):
        with tempfile.TemporaryDirectory() as tmp:
            block = Path(tmp) / "blocks" / "synthetic"
            write_block(block)
            paired = pair_block(block)
            result = fit([paired])
            for finger in FINGERS:
                self.assertGreater(result["angles"][f"{finger}_pip"]["r2"], 0.5, finger)
            data = overlay_data([paired], result)
            self.assertEqual(len(data["frames"]), result["test_windows"])
            self.assertEqual(len(data["frames"][0]["predicted_landmarks"]), 63)
            page = Path(tmp) / "overlay.html"
            write_viewer(data, page)
            self.assertNotIn("__HANDFIT_DATA__", page.read_text())


if __name__ == "__main__":
    unittest.main()
