"""Bit-exact golden model of the FPGA decoder.

This file is the single source of truth for the fixed-point math. The Mac uses
quantize_features() before sending over UART, the cocotb testbench uses
predict_int() as the reference, and the RTL must match it bit for bit.

Float model (from training, stored in model.npz):
    y = ((f - mean) / scale) @ W + b              W: (N_FEAT, N_OUT)

Folded offline (export_weights.py) so the FPGA does no division:
    W' = W / scale[:, None]
    b' = b - (mean / scale) @ W
    optional per-output linear map to servo units: W'' = W' * g, b'' = b' * g + o

Integer pipeline (what the FPGA computes):
    f_q[i]  = sat16(round(f[i] * 2**feat_exp[i]))          (Mac side)
    acc[j]  = sum_i f_q[i] * w_q[j][i]  +  b_q[j]            (48-bit signed)
    y_q[j]  = clamp((acc[j] + round_bias[j]) >>> shift[j], lo[j], hi[j])
    y[j]   ~= y_q[j] / 2**out_frac[j]
where w_q[j][i] = round(W''[i,j] * 2**(w_frac[j] - feat_exp[i])) as int16,
b_q[j] = round(b''[j] * 2**w_frac[j]), shift[j] = w_frac[j] - out_frac[j],
and round_bias[j] = 1 << (shift[j] - 1) when shift[j] > 0 (round half up), else 0.
Every scale is per output, so a servo-mapped output (integer ticks) does not cost
the other outputs precision. Only shift/lo/hi reach the hardware (cfg.mem).
"""
import json

import numpy as np

INT16_MIN, INT16_MAX = -(1 << 15), (1 << 15) - 1
ACC_BITS = 48


class QuantParams:
    def __init__(self, d):
        self.n_feat = int(d["n_feat"])
        self.n_out = int(d["n_out"])
        self.feat_exp = np.asarray(d["feat_exp"], dtype=np.int64)  # (N_FEAT,)
        self.w_q = np.asarray(d["w_q"], dtype=np.int64)            # (N_OUT, N_FEAT)
        self.b_q = np.asarray(d["b_q"], dtype=np.int64)            # (N_OUT,)
        self.w_frac = np.asarray(d["w_frac"], dtype=np.int64)      # (N_OUT,)
        self.out_frac = np.asarray(d["out_frac"], dtype=np.int64)
        self.shift = np.asarray(d["shift"], dtype=np.int64)
        self.lo = np.asarray(d["lo"], dtype=np.int64)              # (N_OUT,)
        self.hi = np.asarray(d["hi"], dtype=np.int64)
        self.target_names = list(d.get("target_names", [f"y{j}" for j in range(self.n_out)]))
        assert (self.shift == self.w_frac - self.out_frac).all() and (self.shift >= 0).all()

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(json.load(fh))


def quantize_features(f, qp):
    """Float features (..., N_FEAT) -> int16 values the Mac sends to the FPGA."""
    f = np.asarray(f, dtype=np.float64)
    scaled = np.round(f * np.exp2(qp.feat_exp.astype(np.float64)))
    return np.clip(scaled, INT16_MIN, INT16_MAX).astype(np.int64)


def predict_int(f_q, qp):
    """Integer features (..., N_FEAT) -> clamped int16 outputs (..., N_OUT). Bit-exact with RTL."""
    f_q = np.asarray(f_q, dtype=np.int64)
    acc = f_q @ qp.w_q.T + qp.b_q  # exact: |acc| < 2**47, well inside int64
    round_bias = np.where(qp.shift > 0, np.left_shift(1, np.maximum(qp.shift - 1, 0)), 0)
    y = (acc + round_bias) >> qp.shift  # numpy >> on int64 is arithmetic, same as SV >>>
    return np.clip(y, qp.lo, qp.hi)


def accumulators(f_q, qp):
    """Raw 48-bit accumulator values, handy for debugging RTL mismatches."""
    return np.asarray(f_q, dtype=np.int64) @ qp.w_q.T + qp.b_q


def dequantize_outputs(y_q, qp):
    return np.asarray(y_q, dtype=np.float64) / np.exp2(qp.out_frac.astype(np.float64))


def predict_float(f, model):
    """Reference float prediction straight from model.npz contents."""
    return ((np.asarray(f) - model["mean"]) / model["scale"]) @ model["W"] + model["b"]
