#!/usr/bin/env python3
"""
train.py

Training script for the SO-101 seed-conditioned IK model.

Input:  14D = 3D position + 6D rotation (continuous) + 5D seed joints
Output: 10D = sin/cos pairs for 5 joints

Key features (unchanged from ES_259_final_project/ik_v3/train.py):
  - L2 normalization of predicted sin/cos pairs before loss
  - Standard MSE on normalized 10D output vs ground truth
  - Adam + ReduceLROnPlateau scheduling (factor 0.5, patience 5, min_lr 1e-6)
  - Early stopping with patience
  - Gradient clipping
  - Independent StandardScalers for pose and seed, fit on the train split
  - 80/10/10 split, atan2 decode, angular MAE metrics

Changes for the SO-101 port:
  - Reads the .npz dataset format written by so101_ik.generate_dataset
    (memory-mapped) instead of a CSV.
  - --max_samples K draws a random subset (seeded) so a single 20M file
    serves the whole data-scaling sweep.
  - --num_threads for CPU runs; CUDA is used automatically when present.
  - Writes results.json with the final metrics next to the other artifacts.

Usage:
    python -m so101_ik.train --data data/so101_seed_20m.npz --out_dir results/so101_20m
    python -m so101_ik.train --data data/so101_seed_20m.npz --out_dir results/so101_1m --max_samples 1000000
"""

import argparse
import json
import os
import time
import pickle
import signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from so101_ik.model import build_model
from so101_ik.generate_dataset import load_npz
from so101_ik.representations import (
    encode_sincos,
    decode_sincos,
    normalize_sincos,
    N_JOINTS,
)


# ============================================================================
# Data loading
# ============================================================================

def load_dataset(npz_path: str, max_samples: int | None = None,
                 seed: int = 42):
    """Load the seed-conditioned dataset (.npz, memory-mapped).

    Parameters
    ----------
    npz_path : path to the .npz written by so101_ik.generate_dataset
    max_samples : if given and smaller than the file, a random subset of
        this size is drawn with ``seed`` (sorted indices keep the mmap
        reads sequential)
    seed : RNG seed for the subset

    Returns
    -------
    X_pose : (N, 9) -- position + 6D rotation
    X_seed : (N, 5) -- seed joint angles
    Y_raw  : (N, 5) -- ground-truth joint angles (radians)
    meta   : dict from the file
    """
    pose_mm, seed_mm, q_mm, meta = load_npz(npz_path, mmap=True)
    n_file = pose_mm.shape[0]
    if pose_mm.shape[1] != 9 or seed_mm.shape[1] != N_JOINTS \
            or q_mm.shape[1] != N_JOINTS:
        raise ValueError(
            f"Unexpected array shapes in {npz_path}: pose9d{pose_mm.shape} "
            f"seed{seed_mm.shape} q{q_mm.shape}")

    if max_samples is not None and 0 < max_samples < n_file:
        rng = np.random.default_rng(seed)
        sub = np.sort(rng.choice(n_file, size=max_samples, replace=False))
        X_pose = np.ascontiguousarray(pose_mm[sub], dtype=np.float32)
        X_seed = np.ascontiguousarray(seed_mm[sub], dtype=np.float32)
        Y_raw = np.ascontiguousarray(q_mm[sub], dtype=np.float32)
        print(f"Loaded {max_samples:,} / {n_file:,} samples (random subset, "
              f"seed={seed}) from {npz_path}")
    else:
        X_pose = np.ascontiguousarray(pose_mm, dtype=np.float32)
        X_seed = np.ascontiguousarray(seed_mm, dtype=np.float32)
        Y_raw = np.ascontiguousarray(q_mm, dtype=np.float32)
        print(f"Loaded {n_file:,} samples from {npz_path}")

    print(f"  Pose:  {X_pose.shape} (3D pos + 6D rot)")
    print(f"  Seed:  {X_seed.shape} ({N_JOINTS} seed joints)")
    print(f"  Joints: {Y_raw.shape} ({N_JOINTS} ground-truth joints)")
    return X_pose, X_seed, Y_raw, meta


# ============================================================================
# Loss with L2 normalization
# ============================================================================

class NormalizedSinCosLoss(nn.Module):
    """MSE loss with L2 normalization of predicted sin/cos pairs.

    Before computing loss, each (sin, cos) pair is L2-normalized
    so that sin^2+cos^2=1. This prevents the network from predicting
    mathematically impossible trigonometric values.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : (N, 10) raw model output
        target : (N, 10) ground-truth sin/cos

        Returns
        -------
        loss : scalar MSE between L2-normalized pred and target
        """
        pred_norm = normalize_sincos(pred)
        return self.mse(pred_norm, target)


# ============================================================================
# Angular error metric (handles wraparound)
# ============================================================================

def angular_error(pred_rad: np.ndarray, true_rad: np.ndarray) -> np.ndarray:
    """Shortest angular distance, handling wraparound."""
    diff = pred_rad - true_rad
    return np.arctan2(np.sin(diff), np.cos(diff))


# ============================================================================
# Batching
# ============================================================================

def iterate_batches(X: torch.Tensor, Y: torch.Tensor, batch_size: int,
                    shuffle: bool, generator: torch.Generator | None = None):
    """Yield (xb, yb) mini-batches by slicing whole tensors.

    Equivalent to DataLoader(TensorDataset, shuffle=...) but avoids the
    per-sample collate overhead, which matters for 16M-row train splits.
    """
    n = X.shape[0]
    if shuffle:
        perm = torch.randperm(n, generator=generator)
    else:
        perm = None
    for start in range(0, n, batch_size):
        if perm is None:
            yield X[start:start + batch_size], Y[start:start + batch_size]
        else:
            idx = perm[start:start + batch_size]
            yield X[idx], Y[idx]


# ============================================================================
# Training
# ============================================================================

def save_checkpoint(out_dir, state_dict, pose_scaler, seed_scaler, history):
    """Write the best weights, both scalers and the history so far to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(state_dict, os.path.join(out_dir, "model_best.pt"))
    with open(os.path.join(out_dir, "pose_scaler.pkl"), "wb") as f:
        pickle.dump(pose_scaler, f)
    with open(os.path.join(out_dir, "seed_scaler.pkl"), "wb") as f:
        pickle.dump(seed_scaler, f)
    pd.DataFrame(history).to_csv(
        os.path.join(out_dir, "training_history.csv"), index=False)


def train(args):
    t_wall0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.num_threads and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    print(f"Device: {device}"
          + (f"  (threads={torch.get_num_threads()})"
             if device.type == "cpu" else
             f"  ({torch.cuda.get_device_name(0)})"))

    # Load data
    X_pose, X_seed, Y_raw, data_meta = load_dataset(
        args.data, max_samples=args.max_samples, seed=args.seed)
    n_samples = len(X_pose)

    # 80/10/10 split
    idx = np.arange(n_samples)
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.2,
                                        random_state=args.seed)
    idx_val, idx_te = train_test_split(idx_tmp, test_size=0.5,
                                       random_state=args.seed)
    print(f"Split: train={len(idx_tr):,}, val={len(idx_val):,}, "
          f"test={len(idx_te):,}")

    # Encode target joints as sin/cos
    Y_sc = encode_sincos(Y_raw).astype(np.float32)

    # Scale inputs: standardize pose, standardize seed separately
    pose_scaler = StandardScaler().fit(X_pose[idx_tr])
    seed_scaler = StandardScaler().fit(X_seed[idx_tr])

    X_pose_s = pose_scaler.transform(X_pose).astype(np.float32)
    X_seed_s = seed_scaler.transform(X_seed).astype(np.float32)
    del X_pose, X_seed

    # Concatenate to form 14D input: [pose_9d_scaled, seed_5d_scaled]
    X_all = np.hstack([X_pose_s, X_seed_s])  # (N, 14)
    del X_pose_s, X_seed_s
    print(f"Combined input: {X_all.shape[1]}D")

    # Tensors per split (train kept on CPU; batches are moved per step)
    def make_tensors(idx):
        return (torch.from_numpy(np.ascontiguousarray(X_all[idx])),
                torch.from_numpy(np.ascontiguousarray(Y_sc[idx])))

    X_tr, Y_tr = make_tensors(idx_tr)
    X_val, Y_val = make_tensors(idx_val)
    gen = torch.Generator().manual_seed(args.seed)

    # Build model
    model = build_model()
    model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    # Loss
    criterion = NormalizedSinCosLoss()

    # Training loop
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_mae_rad": [], "val_mae_deg": [], "lr": []}

    t0 = time.time()
    print(f"\nTraining for up to {args.epochs} epochs")
    print(f"  Batch size: {args.batch_size}  |  LR: {args.lr}")
    print(f"  Patience: {args.patience}  |  Grad clip: {args.grad_clip}")
    print("-" * 80)

    # First Ctrl+C finishes the current epoch and then saves the best model;
    # a second Ctrl+C interrupts immediately as usual.
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        print("\nStop requested; finishing this epoch and saving the best model...",
              flush=True)

    previous_handler = signal.signal(signal.SIGINT, request_stop)

    epochs_run = 0
    for epoch in range(1, args.epochs + 1):
        epochs_run = epoch
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in iterate_batches(X_tr, Y_tr, args.batch_size,
                                      shuffle=True, generator=gen):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss_sum += loss.item() * xb.size(0)
            train_count += xb.size(0)
        train_loss = train_loss_sum / train_count

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_preds = []
        with torch.no_grad():
            for xb, yb in iterate_batches(X_val, Y_val, args.batch_size,
                                          shuffle=False):
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_count += xb.size(0)

                # Normalize before decoding
                pred_norm = normalize_sincos(pred)
                val_preds.append(pred_norm.cpu().numpy())
        val_loss = val_loss_sum / val_count

        # Decode and compute angular MAE
        preds_sc = np.vstack(val_preds)
        preds_rad = decode_sincos(preds_sc)
        val_joints = Y_raw[idx_val]
        mae_rad = float(np.mean(np.abs(angular_error(preds_rad, val_joints))))
        mae_deg = float(np.degrees(mae_rad))

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        # Log
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae_rad"].append(mae_rad)
        history["val_mae_deg"].append(mae_deg)
        history["lr"].append(lr)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_counter = 0
            improved = " *"
            # Checkpoint every improvement so a stopped run still leaves a usable model.
            save_checkpoint(args.out_dir, best_state, pose_scaler, seed_scaler, history)
        else:
            patience_counter += 1

        elapsed = time.time() - t0
        print(f"Epoch {epoch:>4d}/{args.epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"mae={mae_rad:.4f} rad ({mae_deg:.2f} deg)  "
              f"lr={lr:.1e}  ({elapsed:.1f}s){improved}", flush=True)

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
        if stop_requested:
            print(f"\nStopped by request after epoch {epoch}")
            break

    signal.signal(signal.SIGINT, previous_handler)

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest validation loss: {best_val_loss:.6f}")

    # --- Save ---
    out_dir = args.out_dir
    save_checkpoint(out_dir, model.state_dict(), pose_scaler, seed_scaler, history)

    # Save test indices for evaluation (indices into the loaded/subset arrays)
    np.save(os.path.join(out_dir, "test_indices.npy"), idx_te)

    # Plots
    plot_history(history, out_dir)

    print(f"Saved to {out_dir}/")

    # --- Quick test-set evaluation ---
    print("\n" + "=" * 60)
    print("Quick test-set evaluation:")
    print("=" * 60)
    X_te, Y_te = make_tensors(idx_te)
    model.eval()
    test_preds = []
    with torch.no_grad():
        for xb, _ in iterate_batches(X_te, Y_te, args.batch_size,
                                     shuffle=False):
            xb = xb.to(device)
            pred = model(xb)
            pred_norm = normalize_sincos(pred)
            test_preds.append(pred_norm.cpu().numpy())
    test_preds_sc = np.vstack(test_preds)
    test_preds_rad = decode_sincos(test_preds_sc)
    test_joints = Y_raw[idx_te]

    errs = np.abs(angular_error(test_preds_rad, test_joints))
    per_joint_mae = np.mean(errs, axis=0)
    overall_mae = float(np.mean(errs))
    overall_mae_deg = float(np.degrees(overall_mae))

    print(f"  Overall MAE: {overall_mae:.4f} rad ({overall_mae_deg:.2f} deg)")
    for j in range(N_JOINTS):
        print(f"  Joint {j+1}: {per_joint_mae[j]:.4f} rad "
              f"({np.degrees(per_joint_mae[j]):.2f} deg)")

    pct_below_5 = float(np.mean(np.degrees(errs) < 5) * 100)
    pct_below_10 = float(np.mean(np.degrees(errs) < 10) * 100)
    pct_below_1 = float(np.mean(np.degrees(errs) < 1) * 100)
    print(f"  Errors < 1 deg:  {pct_below_1:.1f}%")
    print(f"  Errors < 5 deg:  {pct_below_5:.1f}%")
    print(f"  Errors < 10 deg: {pct_below_10:.1f}%")

    wall_time = time.time() - t_wall0
    results = {
        "data": os.path.abspath(args.data),
        "samples_used": int(n_samples),
        "split": {"train": int(len(idx_tr)), "val": int(len(idx_val)),
                  "test": int(len(idx_te))},
        "epochs_run": int(epochs_run),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_mae_rad": float(history["val_mae_rad"][best_epoch - 1])
        if best_epoch else None,
        "best_val_mae_deg": float(history["val_mae_deg"][best_epoch - 1])
        if best_epoch else None,
        "test_mae_rad": overall_mae,
        "test_mae_deg": overall_mae_deg,
        "test_per_joint_mae_rad": [float(x) for x in per_joint_mae],
        "test_per_joint_mae_deg": [float(np.degrees(x)) for x in per_joint_mae],
        "test_pct_below_1deg": pct_below_1,
        "test_pct_below_5deg": pct_below_5,
        "test_pct_below_10deg": pct_below_10,
        "train_time_s": float(time.time() - t0),
        "wall_time_s": float(wall_time),
        "device": str(device),
        "hyperparams": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "patience": args.patience,
            "grad_clip": args.grad_clip, "seed": args.seed,
            "max_samples": args.max_samples,
        },
        "dataset_meta": data_meta,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {os.path.join(out_dir, 'results.json')}")
    return results


def plot_history(history, out_dir):
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_mae_deg"], color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (degrees)")
    axes[1].set_title("Validation MAE")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["lr"], color="orange")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("LR Schedule")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train SO-101 seed-conditioned IK model")
    parser.add_argument("--data", type=str, required=True,
                        help="Dataset .npz from so101_ik.generate_dataset")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Use a random subset of this many samples")
    parser.add_argument("--num_threads", type=int, default=None,
                        help="torch CPU threads (ignored on CUDA)")
    args = parser.parse_args(argv)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return train(args)


if __name__ == "__main__":
    main()
