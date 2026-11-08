"""Inference for the seed-conditioned SO-101 IK model.

Port of ``ik_v3/infer.py``: load ``model_best.pt`` + the two StandardScalers,
build the 14D input ``[pose_9d (scaled) | seed_5 (scaled)]``, run the network,
L2-normalise each sin/cos pair and atan2-decode to 5 joint angles.  ``solve_refined``
then polishes the DNN estimate with the damped Newton-Raphson solver.

Usage::

    from so101_ik.infer import IKSolver
    solver = IKSolver("so101_ik/results")
    q_dnn = solver.solve(T_target, q_current)
    q, info = solver.solve_refined(T_target, q_current, task="pen")

CLI demo::

    python -m so101_ik.infer --model_dir so101_ik/results --num_samples 20
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import torch

# NumPy pickle compatibility shim (scalers pickled under numpy>=2 reference numpy._core).
try:  # pragma: no cover - depends on numpy version
    import numpy._core as _np_core
except Exception:  # pragma: no cover
    import numpy.core as _np_core
sys.modules.setdefault("numpy._core", _np_core)
sys.modules.setdefault("numpy._core.multiarray", _np_core.multiarray)

from so101_ik.fk import JOINT_LIMITS, N_JOINTS, build_so101_model, forward_kinematics
from so101_ik.model import SeedConditionedIKModel
from so101_ik.newton import ik_newton_refine, rotation_log
from so101_ik.representations import decode_sincos_torch, extract_pose_9d, normalize_sincos

_LIMITS_LO = np.array([lo for lo, _ in JOINT_LIMITS], dtype=np.float64)
_LIMITS_HI = np.array([hi for _, hi in JOINT_LIMITS], dtype=np.float64)


def _load_state_dict(path: str, device: torch.device) -> dict:
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


class IKSolver:
    """Seed-conditioned neural IK solver for the SO-101 (5 kinematic joints)."""

    def __init__(self, model_dir: str, device: str = "cpu", tool_offset_m=(0.0, 0.0, 0.0)):
        self.model_dir = model_dir
        self.device = torch.device(device)
        self.tool_offset_m = tuple(float(x) for x in tool_offset_m)
        self.robot = build_so101_model(tool_offset_m=self.tool_offset_m)

        self.model = SeedConditionedIKModel()
        self.model.load_state_dict(_load_state_dict(os.path.join(model_dir, "model_best.pt"), self.device))
        self.model.to(self.device)
        self.model.eval()

        with open(os.path.join(model_dir, "pose_scaler.pkl"), "rb") as f:
            self.pose_scaler = pickle.load(f)
        with open(os.path.join(model_dir, "seed_scaler.pkl"), "rb") as f:
            self.seed_scaler = pickle.load(f)

    # ------------------------------------------------------------------ DNN
    def _prepare_input(self, pose_9d: np.ndarray, seed: np.ndarray) -> torch.Tensor:
        pose_9d = np.asarray(pose_9d, dtype=np.float32).reshape(-1, 9)
        seed = np.asarray(seed, dtype=np.float32).reshape(-1, N_JOINTS)
        pose_s = self.pose_scaler.transform(pose_9d).astype(np.float32)
        seed_s = self.seed_scaler.transform(seed).astype(np.float32)
        x = np.hstack([pose_s, seed_s])  # (N, 14)
        return torch.from_numpy(x).to(self.device)

    def _predict(self, x_t: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            pred = self.model(x_t)
            pred = normalize_sincos(pred)
            joints = decode_sincos_torch(pred)
        return joints.cpu().numpy()

    def solve(self, T_target: np.ndarray, q_seed: np.ndarray) -> np.ndarray:
        """Single DNN IK solve. ``T_target`` (4,4), ``q_seed`` (5,) -> (5,) radians."""
        pose_9d = extract_pose_9d(np.asarray(T_target, dtype=np.float64))
        x_t = self._prepare_input(pose_9d, q_seed)
        return self._predict(x_t).reshape(N_JOINTS).astype(np.float64)

    def solve_batch(self, T_targets: np.ndarray, q_seeds: np.ndarray) -> np.ndarray:
        """Batch DNN IK solve. ``T_targets`` (N,4,4), ``q_seeds`` (N,5) -> (N,5)."""
        T_targets = np.asarray(T_targets, dtype=np.float64)
        pose_9d = extract_pose_9d(T_targets).reshape(-1, 9)
        x_t = self._prepare_input(pose_9d, q_seeds)
        return self._predict(x_t).astype(np.float64)

    # ------------------------------------------------------------- checking
    def fk(self, q: np.ndarray) -> np.ndarray:
        return forward_kinematics(np.asarray(q, dtype=np.float64), self.robot)

    def verify(self, T_target: np.ndarray, q: np.ndarray) -> dict:
        """FK the joints and compare with the target."""
        T_pred = self.fk(q)
        pos_err_m = float(np.linalg.norm(np.asarray(T_target)[:3, 3] - T_pred[:3, 3]))
        _, angle = rotation_log(np.asarray(T_target)[:3, :3].T @ T_pred[:3, :3])
        return {
            "pos_error_mm": pos_err_m * 1e3,
            "rot_error_deg": float(np.degrees(angle)),
            "T_predicted": T_pred,
        }

    # ----------------------------------------------------- DNN + Newton polish
    def solve_refined(self, T_target: np.ndarray, q_seed: np.ndarray, task: str = "se3", **newton_kwargs):
        """DNN initial guess followed by damped Newton-Raphson refinement.

        Returns ``(q, info)`` where ``info`` has keys ``converged``, ``iters``,
        ``pos_err_mm``, ``rot_err_deg``, ``q_dnn`` and ``task``.
        """
        q_dnn = np.clip(self.solve(T_target, q_seed), _LIMITS_LO, _LIMITS_HI)
        q, iters, pos_err_mm, rot_err_deg, converged = ik_newton_refine(
            np.asarray(T_target, dtype=np.float64), q_dnn, self.robot, task=task, **newton_kwargs
        )
        info = {
            "converged": bool(converged),
            "iters": int(iters),
            "pos_err_mm": float(pos_err_mm),
            "rot_err_deg": float(rot_err_deg),
            "q_dnn": q_dnn,
            "task": task,
        }
        return q, info


def demo(args) -> None:
    """Random FK -> DNN IK -> verify, as in the reference demo."""
    solver = IKSolver(args.model_dir, device="cpu")
    rng = np.random.default_rng(args.seed)
    lo, hi = _LIMITS_LO, _LIMITS_HI

    pos_errors, rot_errors, joint_errors, n_conv = [], [], [], 0
    print(f"\nDemo: {args.num_samples} random FK -> IK -> verify (SO-101)")
    print("=" * 70)
    for i in range(args.num_samples):
        q_true = rng.uniform(lo, hi)
        T_target = solver.fk(q_true)
        q_seed = np.clip(q_true + rng.normal(0.0, args.seed_noise, N_JOINTS), lo, hi)

        q_pred = solver.solve(T_target, q_seed)
        res = solver.verify(T_target, q_pred)
        j_err = np.arctan2(np.sin(q_pred - q_true), np.cos(q_pred - q_true))
        j_mae = float(np.mean(np.abs(j_err)))
        _, info = solver.solve_refined(T_target, q_seed, task="se3")
        n_conv += int(info["converged"])

        pos_errors.append(res["pos_error_mm"])
        rot_errors.append(res["rot_error_deg"])
        joint_errors.append(j_mae)
        if i < 10 or (i + 1) == args.num_samples:
            print(
                f"  [{i + 1}] DNN pos_err={res['pos_error_mm']:.2f} mm  rot_err={res['rot_error_deg']:.2f} deg  "
                f"joint_mae={np.degrees(j_mae):.2f} deg | Newton: {'conv' if info['converged'] else 'FAIL'} "
                f"in {info['iters']} it"
            )

    print(f"\nSummary ({args.num_samples} samples):")
    print(f"  DNN position error: mean={np.mean(pos_errors):.2f} mm  median={np.median(pos_errors):.2f} mm")
    print(f"  DNN rotation error: mean={np.mean(rot_errors):.2f} deg  median={np.median(rot_errors):.2f} deg")
    print(
        f"  DNN joint MAE:      mean={np.degrees(np.mean(joint_errors)):.2f} deg  "
        f"median={np.degrees(np.median(joint_errors)):.2f} deg"
    )
    print(f"  DNN+Newton converged: {n_conv}/{args.num_samples}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 IK inference & demo")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--seed_noise", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    demo(parser.parse_args())


if __name__ == "__main__":
    main()
