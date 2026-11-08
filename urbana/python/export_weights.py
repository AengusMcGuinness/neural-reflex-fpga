"""Fold, quantize and export a trained ridge model for the FPGA.

    python python/export_weights.py --model build/model.npz --calib build/calib.npz \
        [--map grip:0:1:2048:3000] [--out-frac N]

Writes (paths relative to the emg_decoder folder):
    data/weights.mem       int16 w_q, one per line, output-major: w_q[0][0..N_FEAT-1], w_q[1][...], ...
    data/bias.mem          48-bit b_q, one per output
    data/cfg.mem           16-bit: shift[j], lo[j], hi[j] for each output (3 lines per output)
    build/quant_params.json  everything above plus feat_exp (the Mac needs it to quantize features)

See fixed_point.py for the exact integer math.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

import fixed_point as fp

FEAT_HEADROOM = 1.5  # live features may exceed what calibration saw
OUT_HEADROOM = 1.5


def hexw(v, bits):
    return format(int(v) & ((1 << bits) - 1), f"0{bits // 4}x")


def parse_maps(specs, names):
    """--map name:in_lo:in_hi:out_lo:out_hi -> {j: (gain, offset, out_lo, out_hi)}"""
    maps = {}
    for spec in specs:
        name, in_lo, in_hi, out_lo, out_hi = spec.split(":")
        j = names.index(name) if name in names else int(name)
        in_lo, in_hi, out_lo, out_hi = map(float, (in_lo, in_hi, out_lo, out_hi))
        gain = (out_hi - out_lo) / (in_hi - in_lo)
        maps[j] = (gain, out_lo - in_lo * gain, min(out_lo, out_hi), max(out_lo, out_hi))
    return maps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="build/model.npz")
    ap.add_argument("--calib", default="build/calib.npz", help="held-out raw features (X) and optional targets (Y)")
    ap.add_argument("--map", action="append", default=[], help="name:in_lo:in_hi:out_lo:out_hi, e.g. grip:0:1:2048:3000")
    ap.add_argument("--out-frac", type=int, default=None, help="output fractional bits for unmapped outputs (default: auto); mapped outputs use 0")
    ap.add_argument("--root", default=".", help="emg_decoder folder")
    args = ap.parse_args()

    m = np.load(args.model)
    mean, scale, W, b = m["mean"], m["scale"], m["W"], m["b"]
    n_feat, n_out = W.shape
    names = [str(s) for s in m["target_names"]] if "target_names" in m else [f"y{j}" for j in range(n_out)]
    cal = np.load(args.calib)
    X = cal["X"]
    Y = cal["Y"] if "Y" in cal else None

    # 1) fold standardization into the weights
    scale = np.where(scale > 0, scale, 1.0)
    Wf = W / scale[:, None]
    bf = b - (mean / scale) @ W

    # 2) fold optional per-output map to servo units, and set clamps
    lo = np.full(n_out, fp.INT16_MIN, dtype=np.int64)
    hi = np.full(n_out, fp.INT16_MAX, dtype=np.int64)
    maps = parse_maps(args.map, names)
    # a model already in robot ticks (neural_reflex.decode_arm) carries its own clamp range
    if "clamp_lo" in m:
        for j in range(n_out):
            maps.setdefault(j, (1.0, 0.0, float(m["clamp_lo"][j]), float(m["clamp_hi"][j])))
    for j, (g, o, _, _) in maps.items():
        Wf[:, j] *= g
        bf[j] = bf[j] * g + o
    y_float = X @ Wf + bf

    # 3) output fractional bits per output: integer servo ticks for mapped outputs,
    #    otherwise as many as fit the calibrated range in int16
    out_frac = np.zeros(n_out, dtype=np.int64)
    for j in range(n_out):
        if j in maps:
            out_frac[j] = 0
            lo[j], hi[j] = max(fp.INT16_MIN, round(maps[j][2])), min(fp.INT16_MAX, round(maps[j][3]))
        elif args.out_frac is not None:
            out_frac[j] = args.out_frac
        else:
            peak = OUT_HEADROOM * max(np.abs(y_float[:, j]).max(), 1e-12)
            out_frac[j] = max(0, min(15, math.floor(math.log2(fp.INT16_MAX / peak))))

    # 4) per-feature exponents so each feature uses the int16 range
    peak_f = FEAT_HEADROOM * np.maximum(np.abs(X).max(0), 1e-12)
    feat_exp = np.clip(np.floor(np.log2(fp.INT16_MAX / peak_f)), -24, 30).astype(np.int64)

    # 5) weights per integer feature unit, w_frac per output (row)
    w_eff = Wf.T * np.exp2(-feat_exp.astype(np.float64))  # (N_OUT, N_FEAT)
    w_frac = np.floor(np.log2(fp.INT16_MAX / np.maximum(np.abs(w_eff).max(1), 1e-30))).astype(np.int64)
    # bias must fit the 48-bit accumulator with room for the products
    while (big := np.abs(bf) * np.exp2(w_frac) >= 2 ** (fp.ACC_BITS - 3)).any():
        w_frac -= big
    if (w_frac < out_frac).any():
        raise SystemExit(f"w_frac={w_frac} < out_frac={out_frac}; lower --out-frac")
    w_q = np.round(w_eff * np.exp2(w_frac)[:, None]).astype(np.int64)
    b_q = np.round(bf * np.exp2(w_frac)).astype(np.int64)
    assert np.abs(w_q).max() <= fp.INT16_MAX

    params = dict(
        n_feat=n_feat, n_out=n_out, feat_exp=feat_exp.tolist(), w_q=w_q.tolist(), b_q=b_q.tolist(),
        w_frac=w_frac.tolist(), out_frac=out_frac.tolist(), shift=(w_frac - out_frac).tolist(),
        lo=lo.tolist(), hi=hi.tolist(),
        target_names=names, maps={names[j]: list(v) for j, v in maps.items()},
    )
    qp = fp.QuantParams(params)

    # 6) worst-case accumulator check, then error report on calibration data
    worst = (np.abs(w_q) * fp.INT16_MAX).sum(1) + np.abs(b_q)
    assert worst.max() < 2 ** (fp.ACC_BITS - 1), "accumulator could overflow"
    f_q = fp.quantize_features(X, qp)
    sat = np.mean((f_q == fp.INT16_MIN) | (f_q == fp.INT16_MAX))
    y_q = fp.dequantize_outputs(fp.predict_int(f_q, qp), qp)
    y_ref = np.clip(y_float, lo / np.exp2(out_frac), hi / np.exp2(out_frac))
    err = np.abs(y_q - y_ref)
    span = np.maximum(y_ref.max(0) - y_ref.min(0), 1e-12)

    root = Path(args.root)
    (root / "data").mkdir(exist_ok=True)
    (root / "build").mkdir(exist_ok=True)
    (root / "data/weights.mem").write_text("".join(hexw(v, 16) + "\n" for v in w_q.ravel()))
    (root / "data/bias.mem").write_text("".join(hexw(v, fp.ACC_BITS) + "\n" for v in b_q))
    cfg = [v for j in range(n_out) for v in (qp.shift[j], lo[j], hi[j])]
    (root / "data/cfg.mem").write_text("".join(hexw(v, 16) + "\n" for v in cfg))
    np.save(root / "build/calib_used.npy", X)  # testbenches replay real windows
    with open(root / "build/quant_params.json", "w") as fh:
        json.dump(params, fh, indent=1)

    print(f"{n_feat} features x {n_out} outputs")
    print(f"feat_exp range [{feat_exp.min()}, {feat_exp.max()}], feature saturation {100 * sat:.3f}% of calib samples")
    print(f"worst-case |acc| = 2^{math.log2(worst.max()):.1f} (limit 2^{fp.ACC_BITS - 1})")
    for j in range(n_out):
        line = f"  {names[j]:>8} [w_frac {w_frac[j]:2d} out_frac {out_frac[j]:2d} shift {qp.shift[j]:2d} clamp {lo[j]}..{hi[j]}]"
        line += f" max err {err[:, j].max():.4g} ({100 * err[:, j].max() / span[j]:.3f}% of range)"
        line += f", rms {np.sqrt((err[:, j] ** 2).mean()):.3g}"
        if Y is not None and j not in maps:
            r2 = lambda p: 1 - ((p - Y[:, j]) ** 2).sum() / ((Y[:, j] - Y[:, j].mean()) ** 2).sum()
            line += f", R2 float {r2(y_float[:, j]):.4f} -> fixed {r2(y_q[:, j]):.4f}"
        print(line)
    print("wrote data/weights.mem data/bias.mem data/cfg.mem build/quant_params.json")


if __name__ == "__main__":
    main()
