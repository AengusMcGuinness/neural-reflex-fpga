"""Check the exported .mem files and the golden model against a plain-int reference.

Run from emg_decoder/ after export_weights.py:  python3 -m pytest python/test_golden.py -q
"""
import random
from pathlib import Path

import numpy as np

import fixed_point as fp

ROOT = Path(__file__).resolve().parent.parent


def signed(v, bits):
    return v - (1 << bits) if v >= 1 << (bits - 1) else v


def read_mem(name, bits):
    return [signed(int(line, 16), bits) for line in (ROOT / "data" / name).read_text().split()]


def load():
    qp = fp.QuantParams.load(ROOT / "build/quant_params.json")
    w = read_mem("weights.mem", 16)
    b = read_mem("bias.mem", fp.ACC_BITS)
    cfg = read_mem("cfg.mem", 16)
    return qp, w, b, cfg


def rtl_style(f_q, w, b, cfg, n_feat, n_out):
    """What the hardware does, one output at a time, using only the .mem contents."""
    out = []
    for j in range(n_out):
        acc = b[j]
        for i in range(n_feat):
            acc += f_q[i] * w[j * n_feat + i]
        shift, lo, hi = cfg[3 * j], cfg[3 * j + 1], cfg[3 * j + 2]
        y = (acc + ((1 << (shift - 1)) if shift else 0)) >> shift
        out.append(min(max(y, lo), hi))
    return out


def test_mem_matches_json():
    qp, w, b, cfg = load()
    assert w == qp.w_q.ravel().tolist()
    assert b == qp.b_q.tolist()
    assert cfg == [v for j in range(qp.n_out) for v in (qp.shift[j], qp.lo[j], qp.hi[j])]


def test_golden_matches_rtl_style():
    qp, w, b, cfg = load()
    rng = random.Random(1)
    vecs = [[fp.INT16_MIN] * qp.n_feat, [fp.INT16_MAX] * qp.n_feat, [0] * qp.n_feat, [-1] * qp.n_feat]
    vecs += [[rng.randint(fp.INT16_MIN, fp.INT16_MAX) for _ in range(qp.n_feat)] for _ in range(500)]
    got = fp.predict_int(np.array(vecs), qp).tolist()
    for v, g in zip(vecs, got):
        assert g == rtl_style(v, w, b, cfg, qp.n_feat, qp.n_out)


def test_round_half_up_on_negatives():
    # SV: (acc + 2**(s-1)) >>> s ; -3/2 = -1.5 rounds up to -1, -5/2 = -2.5 -> -2
    qp = fp.QuantParams(dict(n_feat=1, n_out=1, feat_exp=[0], w_q=[[1]], b_q=[0], w_frac=[1], out_frac=[0],
                             shift=[1], lo=[-100], hi=[100]))
    assert fp.predict_int(np.array([[-3], [-5], [3], [-1]]), qp).ravel().tolist() == [-1, -2, 2, 0]
