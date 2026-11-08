"""Make a stand-in model.npz + calibration features until the real EMG data exists.

model.npz format (the real training script must write the same keys):
    mean  (N_FEAT,)        feature mean used for standardization
    scale (N_FEAT,)        feature std used for standardization
    W     (N_FEAT, N_OUT)  ridge weights on standardized features
    b     (N_OUT,)         ridge intercept
    target_names (N_OUT,)  optional, e.g. ["grip", ...]

calib.npz format: X (N_SAMPLES, N_FEAT) raw features, optional Y (N_SAMPLES, N_OUT) targets.
Use held-out session features here; they set the fixed-point ranges and measure the error.
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-feat", type=int, default=40)
    ap.add_argument("--n-out", type=int, default=6)
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--alpha", type=float, default=1.0, help="ridge penalty")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="build/model.npz")
    ap.add_argument("--calib", default="build/calib.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # EMG-like features: 8 channels x 5 features, positive, with very different scales per kind
    # (RMS ~ tens of uV, MAV similar, waveform length ~ thousands, zero crossings ~ tens, ...).
    kind_scale = np.array([50.0, 40.0, 3000.0, 20.0, 0.5])
    scales = np.tile(kind_scale, int(np.ceil(args.n_feat / len(kind_scale))))[: args.n_feat]
    latent = rng.standard_normal((args.n_samples, 4))
    mix = rng.standard_normal((4, args.n_feat))
    X = scales * np.exp(0.4 * (latent @ mix) / 2.0 + 0.1 * rng.standard_normal((args.n_samples, args.n_feat)))
    true_W = rng.standard_normal((4, args.n_out))
    Y = 1.0 / (1.0 + np.exp(-(latent @ true_W)))  # targets in [0, 1], like grip closure
    Y += 0.02 * rng.standard_normal(Y.shape)

    mean, scale = X.mean(0), X.std(0)
    Z = (X - mean) / scale
    b = Y.mean(0)
    W = np.linalg.solve(Z.T @ Z + args.alpha * np.eye(args.n_feat), Z.T @ (Y - b))

    names = np.array(["grip"] + [f"y{j}" for j in range(1, args.n_out)])
    np.savez(args.model, mean=mean, scale=scale, W=W, b=b, target_names=names)
    split = args.n_samples // 2
    np.savez(args.calib, X=X[split:], Y=Y[split:])
    r2 = 1 - ((Z[split:] @ W + b - Y[split:]) ** 2).sum(0) / ((Y[split:] - Y[split:].mean(0)) ** 2).sum(0)
    print(f"wrote {args.model} ({args.n_feat} feats x {args.n_out} outs), {args.calib}; dummy R2 per output: {np.round(r2, 3)}")


if __name__ == "__main__":
    main()
