"""Evaluate a trained SO-101 IK model on the held-out test split.

Port of the essentials of ``evaluate_model.py`` from the ES 259 project:

* loads ``model_best.pt`` + scalers from ``--model_dir``
* loads the dataset ``.npz`` and ``test_indices.npy`` (written by ``so101_ik.train``)
* per-joint MAE / RMSE (wrapped angular error) on the test split
* FK reconstruction error (position mm, rotation deg) on a subset
* pred-vs-true scatter PNG per joint and a position-error histogram into ``model_dir``
* ``evaluation_summary.txt`` and ``evaluation.json`` into ``model_dir``

Usage::

    python -m so101_ik.evaluate --model_dir DIR --data PATH.npz [--fk_samples 5000 --max_samples K]
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from so101_ik.fk import JOINT_LIMITS, JOINTS, N_JOINTS, fk_batch
from so101_ik.infer import IKSolver
from so101_ik.representations import JOINT_COLS


def angular_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Wrapped angular difference pred - true in (-pi, pi]."""
    d = pred - true
    return np.arctan2(np.sin(d), np.cos(d))


def fk_error(q_pred: np.ndarray, q_true: np.ndarray, robot: dict) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised FK reconstruction error: (pos mm (N,), rot deg (N,))."""
    T_pred = fk_batch(q_pred.astype(np.float64), robot)
    T_true = fk_batch(q_true.astype(np.float64), robot)
    pos = np.linalg.norm(T_pred[:, :3, 3] - T_true[:, :3, 3], axis=-1) * 1e3
    dR = np.einsum("nij,nkj->nik", T_pred[:, :3, :3], T_true[:, :3, :3])  # R_pred R_true^T
    tr = np.trace(dR, axis1=1, axis2=2)
    rot = np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))
    return pos, rot


def evaluate(model_dir: str, data_path: str, fk_samples: int = 5000, max_samples: int | None = None,
             seed: int = 42, device: str = "cpu", batch_size: int = 65536) -> dict:
    solver = IKSolver(model_dir, device=device)
    data = np.load(data_path)
    pose9d = data["pose9d"]
    seed_q = data["seed"]
    q_true_all = data["q"]

    idx_path = os.path.join(model_dir, "test_indices.npy")
    if os.path.exists(idx_path):
        test_idx = np.load(idx_path)
        print(f"Loaded test split from {idx_path}: {len(test_idx):,} samples")
    else:
        rng = np.random.default_rng(seed)
        n = len(q_true_all)
        test_idx = rng.permutation(n)[: max(1, n // 10)]
        print(f"WARNING: {idx_path} not found; using a random 10% split ({len(test_idx):,} samples)")
    test_idx = np.asarray(test_idx, dtype=np.int64)
    if max_samples is not None and len(test_idx) > max_samples:
        test_idx = test_idx[:max_samples]

    pose_te = pose9d[test_idx].astype(np.float32)
    seed_te = seed_q[test_idx].astype(np.float32)
    q_te = q_true_all[test_idx].astype(np.float64)

    preds = np.empty_like(q_te)
    for s in range(0, len(test_idx), batch_size):
        x_t = solver._prepare_input(pose_te[s : s + batch_size], seed_te[s : s + batch_size])
        preds[s : s + batch_size] = solver._predict(x_t)

    ang = angular_error(preds, q_te)
    abs_err = np.abs(ang)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(ang ** 2)))
    per_joint_mae = np.mean(abs_err, axis=0)
    per_joint_rmse = np.sqrt(np.mean(ang ** 2, axis=0))

    print("\n" + "=" * 70)
    print("TEST SET - ANGULAR METRICS")
    print("=" * 70)
    print(f"  Overall MAE  : {mae:.6f} rad  ({np.degrees(mae):.2f} deg)")
    print(f"  Overall RMSE : {rmse:.6f} rad  ({np.degrees(rmse):.2f} deg)")
    print("  Per-joint breakdown:")
    for i, name in enumerate(JOINTS):
        print(
            f"    {JOINT_COLS[i]} {name:14s}: MAE={per_joint_mae[i]:.6f} ({np.degrees(per_joint_mae[i]):.2f} deg)  "
            f"RMSE={per_joint_rmse[i]:.6f} ({np.degrees(per_joint_rmse[i]):.2f} deg)"
        )

    n_fk = min(fk_samples, len(preds))
    sub = np.random.RandomState(seed).choice(len(preds), n_fk, replace=False)
    pos_err, rot_err = fk_error(preds[sub], q_te[sub], solver.robot)
    print(f"\n{'=' * 70}")
    print(f"TEST SET - FK RECONSTRUCTION ERROR ({n_fk} samples)")
    print("=" * 70)
    print(f"  Position (mm): mean={np.mean(pos_err):.2f} median={np.median(pos_err):.2f} "
          f"std={np.std(pos_err):.2f} max={np.max(pos_err):.2f}")
    print(f"  Rotation (deg): mean={np.mean(rot_err):.2f} median={np.median(rot_err):.2f} "
          f"std={np.std(rot_err):.2f} max={np.max(rot_err):.2f}")
    print("=" * 70)

    os.makedirs(model_dir, exist_ok=True)
    summary = {
        "model_dir": model_dir,
        "data": data_path,
        "n_test": int(len(test_idx)),
        "mae_rad": mae,
        "mae_deg": float(np.degrees(mae)),
        "rmse_rad": rmse,
        "rmse_deg": float(np.degrees(rmse)),
        "per_joint": {
            JOINTS[i]: {
                "mae_rad": float(per_joint_mae[i]),
                "mae_deg": float(np.degrees(per_joint_mae[i])),
                "rmse_rad": float(per_joint_rmse[i]),
                "rmse_deg": float(np.degrees(per_joint_rmse[i])),
            }
            for i in range(N_JOINTS)
        },
        "fk": {
            "n_samples": int(n_fk),
            "pos_err_mm_mean": float(np.mean(pos_err)),
            "pos_err_mm_median": float(np.median(pos_err)),
            "pos_err_mm_std": float(np.std(pos_err)),
            "pos_err_mm_max": float(np.max(pos_err)),
            "rot_err_deg_mean": float(np.mean(rot_err)),
            "rot_err_deg_median": float(np.median(rot_err)),
            "rot_err_deg_std": float(np.std(rot_err)),
            "rot_err_deg_max": float(np.max(rot_err)),
        },
    }
    with open(os.path.join(model_dir, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(model_dir, "evaluation_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== Angular Metrics (test set) ===\n")
        f.write(f"Overall MAE  : {mae:.6f} rad ({np.degrees(mae):.2f} deg)\n")
        f.write(f"Overall RMSE : {rmse:.6f} rad ({np.degrees(rmse):.2f} deg)\n\nPer-joint MAE:\n")
        for i, name in enumerate(JOINTS):
            f.write(f"  {JOINT_COLS[i]} {name}: {per_joint_mae[i]:.6f} rad ({np.degrees(per_joint_mae[i]):.2f} deg)\n")
        f.write(f"\n=== FK Reconstruction Error ({n_fk} samples) ===\n")
        f.write(f"Position (mm)  : mean={np.mean(pos_err):.2f}, median={np.median(pos_err):.2f}\n")
        f.write(f"Rotation (deg) : mean={np.mean(rot_err):.2f}, median={np.median(rot_err):.2f}\n")

    # scatter plots (subsample for drawing speed)
    draw = sub if len(sub) <= 20000 else sub[:20000]
    for i in range(N_JOINTS):
        lo, hi = JOINT_LIMITS[i]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(q_te[draw, i], preds[draw, i], s=2, alpha=0.3, edgecolors="none")
        lims = [lo - 0.2, hi + 0.2]
        ax.plot(lims, lims, "r--", linewidth=1, label="perfect")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"True {JOINT_COLS[i]} ({JOINTS[i]}) [rad]")
        ax.set_ylabel(f"Predicted {JOINT_COLS[i]} [rad]")
        ax.set_title(f"{JOINT_COLS[i]} {JOINTS[i]}: MAE={np.degrees(per_joint_mae[i]):.2f} deg")
        ax.legend()
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(model_dir, f"eval_q{i + 1}.png"), dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pos_err, bins=80, edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(pos_err), color="r", linestyle="--", label=f"Mean = {np.mean(pos_err):.1f} mm")
    ax.set_xlabel("Position error (mm)")
    ax.set_ylabel("Count")
    ax.set_title("DNN-only FK reconstruction - position error (SO-101 test split)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(model_dir, "fk_position_error_hist.png"), dpi=150)
    plt.close(fig)
    print(f"Saved evaluation.json, evaluation_summary.txt, eval_q1..q{N_JOINTS}.png, "
          f"fk_position_error_hist.png -> {model_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the SO-101 IK model on the test split")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data", type=str, required=True, help="dataset .npz used for training")
    parser.add_argument("--fk_samples", type=int, default=5000)
    parser.add_argument("--max_samples", type=int, default=None, help="cap on test samples (for quick runs)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    evaluate(args.model_dir, args.data, fk_samples=args.fk_samples, max_samples=args.max_samples,
             seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
