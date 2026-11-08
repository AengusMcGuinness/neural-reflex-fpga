#!/usr/bin/env python3
"""
generate_dataset.py

Generate a seed-conditioned IK training dataset for the SO-101 arm.

For each sample:
  1. Sample ground-truth joint angles q within JOINT_LIMITS
  2. Compute FK(q) -> 4x4 transform T  (PoE, vectorised over the batch)
  3. Extract 9D pose from T: [x, y, z, r6d_0..r6d_5]
  4. Create seed configuration: q_seed = clip(q + N(0, sigma), limits)
  5. Store: [pose_9d, seed_5d] -> [q1..q5]

The 14D input is: 3D position + 6D rotation + 5D seed joints.
The 10D output is: sin/cos encoded ground-truth joints (encoded at train time).

This trains the network to find the IK solution closest to the seed.

Output format (.npz, float32):
  pose9d : (N, 9)
  seed   : (N, 5)
  q      : (N, 5)
  meta   : 0-d array holding a JSON string (generator settings, joint names,
           column names, limits)

Samples are generated in fixed-size batches straight into preallocated
arrays, so 20M samples (20M x 19 x 4 B = 1.5 GB) fit comfortably in RAM.

Usage:
    python -m so101_ik.generate_dataset --samples 20000000 --output data/so101_seed_20m.npz
    python -m so101_ik.generate_dataset --samples 500000 --seed_noise 0.3 --output x.npz --csv
"""

import argparse
import json
import os
import time

import numpy as np

from so101_ik.fk import JOINTS, N_JOINTS, JOINT_LIMITS, build_so101_model, fk_batch
from so101_ik.representations import (
    extract_pose_9d,
    POSE9D_COLS,
    JOINT_COLS,
    SEED_COLS,
)


# ============================================================================
# Seed generation strategies
# ============================================================================

def generate_seeds(joints: np.ndarray, noise_std: float,
                   limits: list) -> np.ndarray:
    """Generate seed configurations by adding Gaussian noise to ground truth.

    The seeds are clipped to stay within joint limits.

    Parameters
    ----------
    joints : (N, 5) ground-truth joint angles
    noise_std : standard deviation of Gaussian noise (radians)
    limits : list of (lo, hi) for each joint

    Returns
    -------
    seeds : (N, 5) perturbed joint angles
    """
    noise = np.random.randn(*joints.shape).astype(np.float32) * noise_std
    seeds = joints + noise
    for i, (lo, hi) in enumerate(limits):
        seeds[:, i] = np.clip(seeds[:, i], lo, hi)
    return seeds


# ============================================================================
# Main generation
# ============================================================================

def generate_dataset(num_samples: int,
                     seed_noise: float = 0.5,
                     batch_size: int = 100_000,
                     joint_limits: list | None = None,
                     rng_seed: int = 42,
                     tool_offset_m=(0.0, 0.0, 0.0)):
    """Generate the full seed-conditioned dataset.

    Parameters
    ----------
    num_samples : total number of samples to generate
    seed_noise : std dev of Gaussian noise for seed generation (rad)
    batch_size : FK batch size
    joint_limits : per-joint (lo, hi) limits (default: JOINT_LIMITS)
    rng_seed : random seed for reproducibility
    tool_offset_m : extra tool translation in the gripper_frame_link frame

    Returns
    -------
    pose9d : (N, 9) float32
    seed   : (N, 5) float32
    q      : (N, 5) float32
    meta   : dict
    """
    np.random.seed(rng_seed)
    model = build_so101_model(tool_offset_m=tool_offset_m)

    if joint_limits is None:
        joint_limits = [tuple(map(float, lim)) for lim in JOINT_LIMITS]

    pose9d = np.empty((num_samples, 9), dtype=np.float32)
    seed = np.empty((num_samples, N_JOINTS), dtype=np.float32)
    q = np.empty((num_samples, N_JOINTS), dtype=np.float32)

    generated = 0
    t0 = time.time()

    while generated < num_samples:
        n = min(batch_size, num_samples - generated)
        sl = slice(generated, generated + n)

        # 1. Sample ground-truth joints
        joints = np.empty((n, N_JOINTS), dtype=np.float32)
        for i, (lo, hi) in enumerate(joint_limits):
            joints[:, i] = np.random.uniform(lo, hi, size=n)

        # 2. Batch FK
        T_all = fk_batch(joints.astype(np.float64), model)  # (n, 4, 4)

        # 3. Extract 9D pose
        pose9d[sl] = extract_pose_9d(T_all).astype(np.float32)

        # 4. Generate seed joints
        seed[sl] = generate_seeds(joints, seed_noise, joint_limits)

        # 5. Ground-truth joints
        q[sl] = joints

        generated += n
        elapsed = time.time() - t0
        rate = generated / max(elapsed, 1e-9)
        eta = (num_samples - generated) / max(rate, 1e-9)
        print(f"  Generated {generated:>12,} / {num_samples:,}  "
              f"({elapsed:.1f}s, {rate/1e6:.2f} M/s, ETA {eta:.0f}s)",
              flush=True)

    meta = {
        "robot": "SO-101 (so101_new_calib.urdf)",
        "num_samples": int(num_samples),
        "seed_noise_rad": float(seed_noise),
        "rng_seed": int(rng_seed),
        "batch_size": int(batch_size),
        "tool_offset_m": [float(x) for x in tool_offset_m],
        "joints": list(JOINTS),
        "joint_limits": [[float(lo), float(hi)] for lo, hi in joint_limits],
        "pose9d_cols": POSE9D_COLS,
        "seed_cols": SEED_COLS,
        "q_cols": JOINT_COLS,
        "generated_s": float(time.time() - t0),
    }
    return pose9d, seed, q, meta


def save_npz(path: str, pose9d, seed, q, meta: dict) -> None:
    """Write the dataset .npz (float32 arrays + JSON meta string)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path,
             pose9d=pose9d.astype(np.float32, copy=False),
             seed=seed.astype(np.float32, copy=False),
             q=q.astype(np.float32, copy=False),
             meta=np.array(json.dumps(meta)))


def load_npz(path: str, mmap: bool = True):
    """Load a dataset .npz -> (pose9d, seed, q, meta dict).

    Arrays inside an uncompressed .npz are memory-mapped when possible.
    """
    d = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
    meta_raw = d["meta"]
    meta = json.loads(str(meta_raw.item() if meta_raw.ndim == 0 else meta_raw[0]))
    return d["pose9d"], d["seed"], d["q"], meta


def main():
    parser = argparse.ArgumentParser(
        description="Generate seed-conditioned SO-101 IK dataset")
    parser.add_argument("--samples", type=int, default=1_000_000,
                        help="Total number of samples")
    parser.add_argument("--seed_noise", type=float, default=0.5,
                        help="Std dev of seed noise (radians, default 0.5)")
    parser.add_argument("--batch_size", type=int, default=100_000,
                        help="FK batch size")
    parser.add_argument("--rng_seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .npz path")
    parser.add_argument("--csv", action="store_true",
                        help="Also write a CSV next to the .npz "
                             "(columns: pose9d, seed_q1..5, q1..5)")
    args = parser.parse_args()

    print(f"Generating {args.samples:,} samples "
          f"(seed_noise={args.seed_noise} rad)")
    print("=" * 60)

    pose9d, seed, q, meta = generate_dataset(
        num_samples=args.samples,
        seed_noise=args.seed_noise,
        batch_size=args.batch_size,
        rng_seed=args.rng_seed,
    )

    out_path = args.output
    if not out_path.endswith(".npz"):
        out_path += ".npz"
    save_npz(out_path, pose9d, seed, q, meta)
    print(f"\nDataset: {len(q):,} samples")
    print(f"Arrays: pose9d{pose9d.shape} seed{seed.shape} q{q.shape}")
    print(f"Saved to {out_path}")

    if args.csv:
        import pandas as pd
        csv_path = out_path[:-4] + ".csv"
        columns = POSE9D_COLS + SEED_COLS + JOINT_COLS
        df = pd.DataFrame(np.hstack([pose9d, seed, q]), columns=columns)
        df.to_csv(csv_path, index=False)
        print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
