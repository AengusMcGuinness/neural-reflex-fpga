"""Tests for so101_ik.fk (PoE forward kinematics for the SO-101).

Run: .venv-ik/bin/python -m unittest tests.test_so101_fk
"""
import unittest

import numpy as np

from so101_ik.fk import (
    JOINT_LIMITS,
    N_JOINTS,
    build_so101_model,
    fk_batch,
    forward_kinematics,
    space_jacobian,
    urdf_chain_fk,
)


def _random_q(rng, n):
    lo = np.array([l for l, _ in JOINT_LIMITS])
    hi = np.array([h for _, h in JOINT_LIMITS])
    return rng.uniform(lo, hi, size=(n, N_JOINTS))


def _space_twist(T_plus, T_minus, h):
    """Space twist from central difference: V^ = (dT/dq) T^{-1}."""
    dT = (T_plus - T_minus) / (2.0 * h)
    T_mid_inv = np.linalg.inv(0.5 * (T_plus + T_minus))
    return dT, T_mid_inv


class TestSO101FK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_so101_model()
        cls.rng = np.random.default_rng(0)

    def test_poe_matches_urdf_chain(self):
        qs = _random_q(self.rng, 2000)
        max_err = 0.0
        for q in qs:
            err = np.abs(forward_kinematics(q, self.model) - urdf_chain_fk(q)).max()
            max_err = max(max_err, err)
        print(f"\n[fk] PoE vs URDF chain max abs error over 2000 q: {max_err:.3e}")
        self.assertLess(max_err, 1e-9)

    def test_fk_batch_matches_single(self):
        qs = _random_q(self.rng, 500)
        Tb = fk_batch(qs, self.model)
        self.assertEqual(Tb.shape, (500, 4, 4))
        for i, q in enumerate(qs):
            np.testing.assert_allclose(Tb[i], forward_kinematics(q, self.model), atol=1e-12)

    def test_space_jacobian_finite_difference(self):
        qs = _random_q(self.rng, 50)
        h = 1e-6
        worst = 0.0
        for q in qs:
            J = space_jacobian(q, self.model)
            T0 = forward_kinematics(q, self.model)
            T0_inv = np.linalg.inv(T0)
            for i in range(N_JOINTS):
                qp = q.copy(); qp[i] += h
                qm = q.copy(); qm[i] -= h
                dT = (forward_kinematics(qp, self.model) - forward_kinematics(qm, self.model)) / (2 * h)
                V_hat = dT @ T0_inv  # se(3) matrix: [[w^, v],[0,0]]
                w_fd = np.array([V_hat[2, 1], V_hat[0, 2], V_hat[1, 0]])
                v_fd = V_hat[:3, 3]
                err = max(np.abs(J[:3, i] - v_fd).max(), np.abs(J[3:, i] - w_fd).max())
                worst = max(worst, err)
        print(f"[fk] space Jacobian vs finite differences max abs error: {worst:.3e}")
        self.assertLess(worst, 1e-5)

    def test_tool_offset_shifts_by_rotated_offset(self):
        offset = np.array([0.01, -0.02, 0.05])
        m_off = build_so101_model(tool_offset_m=offset)
        qs = _random_q(self.rng, 100)
        for q in qs:
            T0 = forward_kinematics(q, self.model)
            T1 = forward_kinematics(q, m_off)
            np.testing.assert_allclose(T1[:3, :3], T0[:3, :3], atol=1e-12)
            np.testing.assert_allclose(T1[:3, 3], T0[:3, 3] + T0[:3, :3] @ offset, atol=1e-12)
            # also matches the naive chain with the same offset
            np.testing.assert_allclose(T1, urdf_chain_fk(q, tool_offset_m=offset), atol=1e-9)

    def test_zero_config_and_reach(self):
        T0 = forward_kinematics(np.zeros(N_JOINTS), self.model)
        self.assertTrue(np.all(np.isfinite(T0)))
        qs = _random_q(self.rng, 20000)
        P = fk_batch(qs, self.model)[:, :3, 3]
        # Reach = tool distance from the shoulder_lift joint. That joint is
        # 0.035 m off the pan axis, so compute its position at each sample's
        # pan angle: exp([S1] q1) applied to its q=0 location.
        w1, v1 = self.model["w"][0], self.model["v"][0]
        shoulder0 = np.array([0.0692, -0.0183, 0.1166])  # shoulder_lift origin at q=0
        # exp of a pure rotation about the pan axis (through point p1 on the axis)
        # p(q1) = p1 + R(q1) (shoulder0 - p1), with p1 = cross(w1, v1) a point on the axis
        p1 = np.cross(w1, v1)
        wh = np.array([[0, -w1[2], w1[1]], [w1[2], 0, -w1[0]], [-w1[1], w1[0], 0]])
        c, s = np.cos(qs[:, 0]), np.sin(qs[:, 0])
        R = np.eye(3)[None] + s[:, None, None] * wh[None] + (1 - c)[:, None, None] * (wh @ wh)[None]
        shoulder = p1 + np.einsum("nij,j->ni", R, shoulder0 - p1)
        r = np.linalg.norm(P - shoulder, axis=1)
        r_base = np.linalg.norm(P, axis=1)
        print(f"[fk] tool position at q=0 (base_link): {T0[:3, 3]}")
        print(f"[fk] reach from shoulder_lift joint: max={r.max():.4f} m  "
              f"median={np.median(r):.4f} m  mean={r.mean():.4f} m  (link-length sum 0.4132 m)")
        print(f"[fk] distance from base_link origin: max={r_base.max():.4f} m  median={np.median(r_base):.4f} m")
        self.assertGreater(r.max(), 0.25)
        self.assertLess(r.max(), 0.45)


if __name__ == "__main__":
    unittest.main()
