"""Core tests for the SO-101 seed-conditioned IK pipeline.

Run with:
    .venv-ik/bin/python -m unittest tests.test_so101_ik_core
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from so101_ik.model import SeedConditionedIKModel, build_model
from so101_ik.representations import (
    encode_sincos,
    decode_sincos,
    decode_sincos_torch,
    normalize_sincos,
    extract_pose_9d,
    rotmat_to_6d,
    r6d_to_rotmat,
    POSE9D_COLS,
    JOINT_COLS,
    SEED_COLS,
    SINCOS_COLS,
)

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestModel(unittest.TestCase):
    def test_param_count_and_shapes(self):
        model = build_model()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[test] SeedConditionedIKModel parameters: {n_params:,}")
        self.assertGreater(n_params, 400_000)
        self.assertEqual(SeedConditionedIKModel.INPUT_DIM, 14)
        self.assertEqual(SeedConditionedIKModel.OUTPUT_DIM, 10)
        self.assertEqual(model.block1.linear.in_features, 14)
        self.assertEqual(model.block1.linear.out_features, 256)
        self.assertEqual(model.block2.linear.out_features, 512)
        self.assertEqual(model.block3.linear.out_features, 512)
        self.assertTrue(model.block3.use_residual)
        self.assertEqual(model.block4_linear.out_features, 256)
        self.assertEqual(model.output_layer.out_features, 10)
        model.eval()
        x = torch.randn(7, 14)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (7, 10))
        self.assertTrue(torch.isfinite(y).all())


class TestRepresentations(unittest.TestCase):
    def test_column_names(self):
        self.assertEqual(len(POSE9D_COLS), 9)
        self.assertEqual(JOINT_COLS, ["q1", "q2", "q3", "q4", "q5"])
        self.assertEqual(SEED_COLS, [f"seed_q{i}" for i in range(1, 6)])
        self.assertEqual(len(SINCOS_COLS), 10)
        self.assertEqual(SINCOS_COLS[:2], ["sin_q1", "cos_q1"])

    def test_normalize_sincos_unit_circle(self):
        torch.manual_seed(0)
        raw = torch.randn(64, 10) * 3.0
        out = normalize_sincos(raw)
        self.assertEqual(tuple(out.shape), (64, 10))
        pairs = out.reshape(-1, 5, 2)
        norms = torch.sqrt((pairs ** 2).sum(-1))
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))

    def test_encode_decode_round_trip(self):
        rng = np.random.default_rng(1)
        q = rng.uniform(-np.pi + 1e-3, np.pi - 1e-3, size=(1000, 5)).astype(np.float32)
        sc = encode_sincos(q)
        self.assertEqual(sc.shape, (1000, 10))
        q_back = decode_sincos(sc.astype(np.float64))
        self.assertLess(np.max(np.abs(q_back - q)), 1e-6)
        q_back_t = decode_sincos_torch(torch.from_numpy(sc).double()).numpy()
        self.assertLess(np.max(np.abs(q_back_t - q)), 1e-6)

    def test_rotation_6d_round_trip(self):
        rng = np.random.default_rng(2)
        A = rng.normal(size=(20, 3, 3))
        R = np.linalg.qr(A)[0]
        R[np.linalg.det(R) < 0, :, 2] *= -1
        r6 = rotmat_to_6d(R)
        self.assertEqual(r6.shape, (20, 6))
        R_back = r6d_to_rotmat(torch.from_numpy(r6)).numpy()
        self.assertLess(np.max(np.abs(R_back - R)), 1e-9)
        T = np.tile(np.eye(4), (20, 1, 1))
        T[:, :3, :3] = R
        T[:, :3, 3] = rng.normal(size=(20, 3))
        p9 = extract_pose_9d(T)
        self.assertEqual(p9.shape, (20, 9))
        np.testing.assert_allclose(p9[:, :3], T[:, :3, 3])
        np.testing.assert_allclose(extract_pose_9d(T[0]), p9[0])


class TestGenerateAndTrain(unittest.TestCase):
    def test_generate_20k_and_train_2_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "so101_20k.npz")
            out_dir = os.path.join(tmp, "run")
            gen = subprocess.run(
                [PY, "-m", "so101_ik.generate_dataset", "--samples", "20000",
                 "--seed_noise", "0.5", "--rng_seed", "42", "--output", data],
                cwd=REPO, capture_output=True, text=True)
            self.assertEqual(gen.returncode, 0, gen.stdout + gen.stderr)
            self.assertTrue(os.path.isfile(data))

            d = np.load(data)
            self.assertEqual(d["pose9d"].shape, (20000, 9))
            self.assertEqual(d["seed"].shape, (20000, 5))
            self.assertEqual(d["q"].shape, (20000, 5))
            self.assertEqual(d["pose9d"].dtype, np.float32)
            meta = json.loads(str(d["meta"]))
            self.assertEqual(meta["num_samples"], 20000)
            self.assertEqual(len(meta["joints"]), 5)
            # 6D rotation columns are unit vectors (first two columns of R)
            n1 = np.linalg.norm(d["pose9d"][:, 3:6], axis=1)
            self.assertTrue(np.allclose(n1, 1.0, atol=1e-5))
            # Seeds respect the limits
            for i, (lo, hi) in enumerate(meta["joint_limits"]):
                self.assertTrue(np.all(d["seed"][:, i] >= lo - 1e-6))
                self.assertTrue(np.all(d["seed"][:, i] <= hi + 1e-6))

            tr = subprocess.run(
                [PY, "-m", "so101_ik.train", "--data", data, "--out_dir", out_dir,
                 "--epochs", "2", "--max_samples", "20000", "--batch_size", "512",
                 "--num_threads", "2"],
                cwd=REPO, capture_output=True, text=True)
            self.assertEqual(tr.returncode, 0, tr.stdout + tr.stderr)

            for name in ("model_best.pt", "pose_scaler.pkl", "seed_scaler.pkl",
                         "training_history.csv", "training_curves.png",
                         "test_indices.npy", "results.json"):
                self.assertTrue(os.path.isfile(os.path.join(out_dir, name)), name)

            with open(os.path.join(out_dir, "results.json")) as f:
                res = json.load(f)
            self.assertEqual(res["epochs_run"], 2)
            self.assertEqual(res["samples_used"], 20000)
            self.assertTrue(np.isfinite(res["best_val_loss"]))
            self.assertTrue(np.isfinite(res["best_val_mae_deg"]))
            self.assertTrue(np.isfinite(res["test_mae_deg"]))
            self.assertEqual(len(res["test_per_joint_mae_deg"]), 5)

            hist = np.genfromtxt(os.path.join(out_dir, "training_history.csv"),
                                 delimiter=",", names=True)
            self.assertEqual(hist.shape[0], 2)
            self.assertTrue(np.all(np.isfinite(hist["val_mae_deg"])))

            # The saved weights load into a fresh model with the right shapes
            model = SeedConditionedIKModel()
            model.load_state_dict(torch.load(os.path.join(out_dir, "model_best.pt"),
                                             map_location="cpu", weights_only=True))


if __name__ == "__main__":
    unittest.main()
