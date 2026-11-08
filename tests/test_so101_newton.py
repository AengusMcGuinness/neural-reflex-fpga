"""Tests for so101_ik.newton / infer / benchmark.

Run: /home/dandi/neural-reflex-fpga/.venv-ik/bin/python -m unittest tests.test_so101_newton
"""

import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from so101_ik.fk import JOINT_LIMITS, N_JOINTS, build_so101_model, forward_kinematics
from so101_ik.newton import axis_angle_error, axis_error, ik_newton_refine

LO = np.array([lo for lo, _ in JOINT_LIMITS])
HI = np.array([hi for _, hi in JOINT_LIMITS])
PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _random_targets(n, rng, robot):
    qs = rng.uniform(LO, HI, size=(n, N_JOINTS))
    Ts = np.stack([forward_kinematics(q, robot) for q in qs])
    starts = np.clip(qs + rng.normal(0.0, 0.05, size=qs.shape), LO, HI)
    return qs, Ts, starts


class TestNewtonSE3(unittest.TestCase):
    def test_se3_converges_from_near_start(self):
        robot = build_so101_model()
        rng = np.random.default_rng(0)
        n = 200
        _, Ts, starts = _random_targets(n, rng, robot)
        n_conv = 0
        for T, q0 in zip(Ts, starts):
            q, iters, pe, re, conv = ik_newton_refine(T, q0, robot, task="se3")
            self.assertEqual(q.shape, (N_JOINTS,))
            self.assertTrue(np.all(q >= LO - 1e-12) and np.all(q <= HI + 1e-12))
            if conv:
                self.assertLess(pe, 1.0)
                self.assertLess(re, 0.573)
                self.assertLess(iters, 200)
                n_conv += 1
        rate = n_conv / n
        print(f"\n[se3] converged {n_conv}/{n} = {100 * rate:.1f}%")
        self.assertGreaterEqual(rate, 0.95)


class TestNewtonPen(unittest.TestCase):
    def test_pen_converges_and_axis_matches(self):
        robot = build_so101_model()
        rng = np.random.default_rng(1)
        n = 200
        _, Ts, starts = _random_targets(n, rng, robot)
        n_conv = 0
        for T, q0 in zip(Ts, starts):
            q, iters, pe, re, conv = ik_newton_refine(T, q0, robot, task="pen")
            if conv:
                n_conv += 1
                T_fin = forward_kinematics(q, robot)
                self.assertLess(np.linalg.norm(T_fin[:3, 3] - T[:3, 3]) * 1e3, 1.0)
                ang = axis_angle_error(T_fin[:3, 2], T[:3, 2])
                self.assertLess(ang, 0.01)
                self.assertAlmostEqual(np.degrees(ang), re, places=6)
        rate = n_conv / n
        print(f"\n[pen] converged {n_conv}/{n} = {100 * rate:.1f}%")
        self.assertGreaterEqual(rate, 0.95)

    def test_pen_roll_is_free(self):
        """Target differing from the start only by a wrist roll: the pen task must not chase the roll.

        The default gripper_frame tool point sits ~8 mm off the wrist-roll axis, so a pure
        roll does move it.  ``tool_offset_m`` is used to put the pen tip exactly on the roll
        axis (gripper_frame_link = Ry(pi) of gripper_link, so the (-0.0079, -0.000218) lateral
        offset is cancelled by (+0.0079, +0.000218) in gripper_link = (-0.0079, +0.000218) in the
        tool frame).  Then a target that differs only by roll is already solved: 1 iteration,
        roll untouched.
        """
        robot = build_so101_model(tool_offset_m=(-0.0079, 0.000218121, 0.0))
        q0 = np.array([0.3, -0.5, 0.8, 0.2, 0.1])
        q1 = q0.copy()
        q1[4] += 0.7
        T = forward_kinematics(q1, robot)
        # sanity: tip on the roll axis -> same tip position and same tool z-axis
        T0 = forward_kinematics(q0, robot)
        # (URDF angles are rounded to 3.14159 etc., leaving a few micro-rad / micro-mm of residual)
        self.assertLess(np.linalg.norm(T0[:3, 3] - T[:3, 3]) * 1e3, 1e-2)
        self.assertLess(axis_angle_error(T0[:3, 2], T[:3, 2]), 1e-4)
        q, iters, pe, re, conv = ik_newton_refine(T, q0, robot, task="pen")
        self.assertTrue(conv)
        self.assertEqual(iters, 1)
        np.testing.assert_allclose(q, q0)
        # se3 on the other hand must move the roll to reach the target rotation
        q_se3, _, _, _, conv_se3 = ik_newton_refine(T, q0, robot, task="se3")
        self.assertTrue(conv_se3)
        self.assertAlmostEqual(q_se3[4], q1[4], places=2)

    def test_pen_roll_stays_exactly_at_seed(self):
        """With the default (off-axis) pen tip the roll column is zeroed: roll never moves, converged or not."""
        robot = build_so101_model(tool_offset_m=(0.0, 0.0, 0.05))
        rng = np.random.default_rng(5)
        _, Ts, starts = _random_targets(100, rng, robot)
        n_conv = 0
        for T, q0 in zip(Ts, starts):
            q, _, _, _, conv = ik_newton_refine(T, q0, robot, task="pen")
            self.assertEqual(q[4], q0[4])
            n_conv += conv
        self.assertGreaterEqual(n_conv, 95)
        # the se3 reference path still uses the roll
        T = forward_kinematics(np.array([0.3, -0.5, 0.8, 0.2, 0.9]), robot)
        q_se3, _, _, _, conv = ik_newton_refine(T, np.array([0.3, -0.5, 0.8, 0.2, 0.1]), robot, task="se3")
        self.assertTrue(conv)
        self.assertNotAlmostEqual(q_se3[4], 0.1, places=2)

    def test_pen_converges_from_inverted_start(self):
        """Pen-down targets from poses whose tool z points UP (nearly antiparallel to the target z).

        The geodesic axis error keeps a full-size pull there, where the raw cross product
        (sin(theta)) nearly vanishes.  The exact straight-up home is excluded on purpose: it
        is a kinematic singularity where the tilt direction is decided by a 0.04 x-component
        and Newton can park the elbow/wrist on their limits, which no error term fixes.
        """
        robot = build_so101_model(tool_offset_m=(0.0, 0.0, 0.05))
        roll = 0.0487
        inverted = [np.array([-0.1833, -1.5264, 1.1001, -1.1125, roll]),
                    np.array([0.155, -0.9947, 0.6626, -1.2256, roll]),
                    np.array([0.0874, -0.8305, -0.0005, -0.6709, roll]),
                    np.array([0.3526, 0.0998, -0.1374, -1.4513, roll])]
        R_down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        for q0 in inverted:
            z_up = forward_kinematics(q0, robot)[:3, 2]
            self.assertGreater(z_up[2], 0.9, q0)                         # pen pointing up
            for x, y in ((0.20, 0.0), (0.22, 0.08), (0.18, -0.09)):
                T = np.eye(4)
                T[:3, :3] = R_down
                T[:3, 3] = (x, y, 0.01)
                self.assertGreater(axis_angle_error(z_up, T[:3, 2]), np.radians(150))
                q, iters, pe, re, conv = ik_newton_refine(T, q0, robot, task="pen")
                self.assertTrue(conv, (q0, x, y, iters, pe, re))
                self.assertLess(pe, 1.0)
                self.assertLess(re, 0.573)
                self.assertEqual(q[4], roll)

    def test_axis_error_geodesic(self):
        z = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(axis_error(z, z), np.zeros(3))
        # 90 deg about +x: cross(z, -y) = +x, theta = pi/2
        e = axis_error(z, np.array([0.0, -1.0, 0.0]))
        np.testing.assert_allclose(e, [np.pi / 2, 0.0, 0.0], atol=1e-12)
        # 150 deg: magnitude theta, not sin(theta)
        z_des = np.array([0.0, np.sin(np.radians(150)), np.cos(np.radians(150))])
        e = axis_error(z, z_des)
        self.assertAlmostEqual(np.linalg.norm(e), np.radians(150), places=12)
        self.assertAlmostEqual(np.dot(e, z), 0.0, places=12)
        # antiparallel: magnitude pi along some axis perpendicular to z_cur
        for z_cur in (z, np.array([1.0, 0.0, 0.0]), np.array([0.6, 0.0, 0.8])):
            e = axis_error(z_cur, -z_cur)
            self.assertAlmostEqual(np.linalg.norm(e), np.pi, places=12)
            self.assertAlmostEqual(np.dot(e, z_cur), 0.0, places=12)


class TestIKSolverUntrained(unittest.TestCase):
    def test_solver_loads_and_returns_shapes(self):
        import torch
        from sklearn.preprocessing import StandardScaler

        from so101_ik.infer import IKSolver
        from so101_ik.model import SeedConditionedIKModel

        self.assertEqual(SeedConditionedIKModel.INPUT_DIM, 14)
        self.assertEqual(SeedConditionedIKModel.OUTPUT_DIM, 10)
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as d:
            torch.save(SeedConditionedIKModel().state_dict(), os.path.join(d, "model_best.pt"))
            ps = StandardScaler().fit(rng.normal(size=(100, 9)).astype(np.float32))
            ss = StandardScaler().fit(rng.uniform(LO, HI, size=(100, N_JOINTS)).astype(np.float32))
            with open(os.path.join(d, "pose_scaler.pkl"), "wb") as f:
                pickle.dump(ps, f)
            with open(os.path.join(d, "seed_scaler.pkl"), "wb") as f:
                pickle.dump(ss, f)

            solver = IKSolver(d, device="cpu")
            robot = build_so101_model()
            q_true = rng.uniform(LO, HI)
            T = forward_kinematics(q_true, robot)
            q_seed = np.clip(q_true + rng.normal(0, 0.5, N_JOINTS), LO, HI)

            q = solver.solve(T, q_seed)
            self.assertEqual(q.shape, (N_JOINTS,))
            self.assertTrue(np.all(np.isfinite(q)))
            self.assertTrue(np.all(np.abs(q) <= np.pi + 1e-6))

            qb = solver.solve_batch(np.stack([T, T, T]), np.stack([q_seed] * 3))
            self.assertEqual(qb.shape, (3, N_JOINTS))
            np.testing.assert_allclose(qb[0], q, atol=1e-5)

            v = solver.verify(T, q)
            for k in ("pos_error_mm", "rot_error_deg", "T_predicted"):
                self.assertIn(k, v)
            self.assertEqual(v["T_predicted"].shape, (4, 4))

            for task in ("se3", "pen"):
                qr, info = solver.solve_refined(T, q_seed, task=task, max_iter=50)
                self.assertEqual(qr.shape, (N_JOINTS,))
                for k in ("converged", "iters", "pos_err_mm", "rot_err_deg", "q_dnn"):
                    self.assertIn(k, info)
                self.assertEqual(info["q_dnn"].shape, (N_JOINTS,))
                self.assertIsInstance(info["converged"], bool)
                self.assertLessEqual(info["iters"], 50)


class TestBenchmarkCLI(unittest.TestCase):
    def test_no_model_runs_and_writes_json(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "bench.json")
            proc = subprocess.run(
                [PY, "-m", "so101_ik.benchmark", "--no_model", "--n", "50", "--out", out],
                cwd=REPO, capture_output=True, text=True, timeout=600,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr[-2000:])
            self.assertTrue(os.path.exists(out))
            with open(out) as f:
                payload = json.load(f)
            s = payload["summary"]
            self.assertEqual(s["num_samples"], 50)
            self.assertIsNone(s["model_dir"])
            self.assertNotIn("dnn", s)
            for key in ("random", "seed_init_baseline"):
                self.assertIn("convergence_percent", s[key])
                self.assertIn("mean_ms", s[key])
            self.assertEqual(len(payload["first_30_examples"]), 30)
            self.assertIn("Seed-Initialized baseline", proc.stdout)


if __name__ == "__main__":
    unittest.main()
