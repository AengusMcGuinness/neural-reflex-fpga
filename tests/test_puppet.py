"""Puppet recording (leader-arm encoders as labels) and encoder-label training on a synthetic block."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neural_reflex import decode_arm
from neural_reflex.leader_arm import JOINTS
from neural_reflex.record_puppet import record


class ReadOnlyBus:
    def __init__(self, torque_on=()):
        self.torque_on = set(torque_on)
        self.reads = 0

    def read(self, register, name, normalize=False):
        self.reads += 1
        return {"Present_Position": 2000 + JOINTS.index(name), "Min_Position_Limit": 500,
                "Max_Position_Limit": 3500, "Torque_Enable": int(name in self.torque_on),
                "Operating_Mode": 0}[register]

    def write(self, *args, **kwargs):
        raise AssertionError("puppet recording must never write to the arm")


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        self.t += 0.01
        return self.t


def synthetic_block(folder: Path, seconds=40.0, seed=0):
    """EMG whose channel-0/1 amplitude tracks two joint trajectories, plus encoder rows of those joints."""
    rng = np.random.default_rng(seed)
    rate, batch = 2048, 16
    lift = lambda t: 2000 + 400 * np.sin(2 * np.pi * t / 8)
    elbow = lambda t: 1500 + 300 * np.sin(2 * np.pi * t / 5 + 1)
    t0 = 1000.0
    with (folder / "emg.jsonl").open("w") as emg, (folder / "motion.jsonl").open("w") as motion:
        for seq in range(int(seconds * rate / batch)):
            ts = (seq * batch + np.arange(batch)) / rate
            amp = np.ones((batch, 8)) * 40
            amp[:, 0] = 40 + 0.5 * (lift(ts) - 1600)
            amp[:, 1] = 40 + 0.5 * (elbow(ts) - 1200)
            values = (32768 + amp * rng.standard_normal((batch, 8))).astype(int).tolist()
            emg.write(json.dumps({"host_time_s": t0 + ts[-1] + 0.002, "band_timestamp_us": int(ts[0] * 1e6),
                                  "sequence": seq, "sample_interval_us": 1e6 / rate, "values": values}) + "\n")
            if seq % 3 == 0:
                for kind, vals in (("orientation", [1, 0, 0, 0]), ("gyro", rng.normal(0, 5, 3).tolist())):
                    motion.write(json.dumps({"host_time_s": t0 + ts[-1], "type": kind,
                                             "payload": {"values": vals}}) + "\n")
    with (folder / "arm_encoders.jsonl").open("w") as enc:
        for t in np.arange(0, seconds, 0.02):
            positions = {j: 2000 for j in JOINTS}
            positions.update(shoulder_lift=int(lift(t)), elbow_flex=int(elbow(t)))
            enc.write(json.dumps({"host_time_s": t0 + t, "cue": "free-slow", "positions": positions}) + "\n")


class PuppetTests(unittest.TestCase):
    def test_records_without_writing(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "arm_encoders.jsonl"
            n = record(ReadOnlyBus(), out, "puppet", rate_hz=1000, routine=(("hold", 0.3, "hold"),),
                       log=lambda *a, **k: None, clock=FakeClock())
            rows = [json.loads(l) for l in out.read_text().splitlines()]
            self.assertEqual(len(rows), n)
            self.assertGreater(n, 5)
            self.assertEqual(rows[0]["positions"]["elbow_flex"], 2002)
            self.assertEqual(rows[0]["cue"], "hold")

    def test_refuses_with_torque_on(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                record(ReadOnlyBus(torque_on={"elbow_flex"}), Path(d) / "a.jsonl", "p", 50,
                       log=lambda *a, **k: None, clock=FakeClock())

    def test_encoder_labels_train_a_decoder(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "puppet"
            folder.mkdir()
            synthetic_block(folder)
            paired = decode_arm.pair_encoders(folder, decode_arm.ARM_TARGETS, "emg+motion", label_delay=0.0)
            self.assertEqual(paired["features"].shape[1], 175)
            self.assertGreater(len(paired["ticks"]), 1500)
            decode_arm.main([str(folder), "--labels", "encoders", "--output", str(Path(d) / "models")])
            model = np.load(Path(d) / "models" / "arm_decoder.npz")
            pred = decode_arm.predict_ticks({k: model[k] for k in ("mean", "scale", "W", "b", "clamp_lo", "clamp_hi")},
                                            paired["features"])
            err = np.abs(pred - paired["ticks"]).mean(axis=0)
            self.assertTrue((err < 40).all(), err)        # amplitude-coded synthetic EMG is decodable
            self.assertTrue((folder / "robot_plan_decoded.json").exists())


if __name__ == "__main__":
    unittest.main()
