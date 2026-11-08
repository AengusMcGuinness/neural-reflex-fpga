"""ridge_mac vs the Python golden model (python/fixed_point.py), bit for bit.

Uses the weights currently exported to data/*.mem and build/quant_params.json.
Run from emg_decoder/sim:  python test_ridge_mac.py
"""
import json
import os
import random
import sys
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, ReadOnly
from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
import fixed_point as fp  # noqa: E402

QP = fp.QuantParams.load(ROOT / "build/quant_params.json")


async def reset(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    for name in ("feat_we", "w_we", "b_we", "cfg_we", "start"):
        getattr(dut, name).value = 0
    dut.rst.value = 1
    await ClockCycles(dut.clk, 3)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)


async def write_features(dut, f_q):
    for i, v in enumerate(f_q):
        dut.feat_we.value = 1
        dut.feat_addr.value = i
        dut.feat_data.value = int(v) & 0xFFFF
        await RisingEdge(dut.clk)
    dut.feat_we.value = 0


async def run_frame(dut):
    """Pulse start, collect y in order, return (outputs, cycles from start to done)."""
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    ys, cycles = [], 0
    while True:
        await ReadOnly()
        if dut.y_valid.value:
            assert int(dut.y_idx.value) == len(ys)
            ys.append(dut.y_data.value.to_signed())
        if dut.done.value:
            break
        await RisingEdge(dut.clk)
        cycles += 1
    await RisingEdge(dut.clk)
    return ys, cycles


@cocotb.test()
async def test_matches_golden(dut):
    await reset(dut)
    rng = np.random.default_rng(0)
    calib = ROOT / "build/calib_used.npy"
    vectors = [np.full(QP.n_feat, fp.INT16_MAX), np.full(QP.n_feat, fp.INT16_MIN), np.zeros(QP.n_feat, int)]
    if calib.exists():  # real feature windows, quantized the way the Mac will
        X = np.load(calib)
        vectors += list(fp.quantize_features(X[rng.choice(len(X), 40, replace=False)], QP))
    vectors += list(rng.integers(fp.INT16_MIN, fp.INT16_MAX + 1, size=(20, QP.n_feat)))
    worst = 0
    for f_q in vectors:
        await write_features(dut, f_q)
        ys, cycles = await run_frame(dut)
        expect = fp.predict_int(f_q, QP).tolist()
        assert ys == expect, f"RTL {ys} != golden {expect}"
        worst = max(worst, cycles)
    dut._log.info(f"{len(vectors)} frames bit-exact; {worst} cycles per frame = {worst * 10 / 1000:.2f} us at 100 MHz")


@cocotb.test()
async def test_runtime_weight_load(dut):
    """Overwrite every weight, bias and cfg word through the write ports, then compare."""
    await reset(dut)
    rng = random.Random(3)
    n, m = QP.n_feat, QP.n_out
    w = [[rng.randint(-3000, 3000) for _ in range(n)] for _ in range(m)]
    b = [rng.randint(-(1 << 40), 1 << 40) for _ in range(m)]
    shift = [rng.randint(0, 24) for _ in range(m)]
    lo = [rng.randint(-20000, 0) for _ in range(m)]
    hi = [rng.randint(1, 20000) for _ in range(m)]
    for j in range(m):
        for i in range(n):
            dut.w_we.value, dut.w_addr.value, dut.w_data.value = 1, j * n + i, w[j][i] & 0xFFFF
            await RisingEdge(dut.clk)
        dut.w_we.value = 0
        dut.b_we.value, dut.b_addr.value, dut.b_data.value = 1, j, b[j] & ((1 << 48) - 1)
        await RisingEdge(dut.clk)
        dut.b_we.value = 0
        for k, v in enumerate((shift[j], lo[j], hi[j])):
            dut.cfg_we.value, dut.cfg_addr.value, dut.cfg_data.value = 1, 3 * j + k, v & 0xFFFF
            await RisingEdge(dut.clk)
        dut.cfg_we.value = 0
    qp = fp.QuantParams(dict(n_feat=n, n_out=m, feat_exp=[0] * n, w_q=w, b_q=b, w_frac=shift,
                             out_frac=[0] * m, shift=shift, lo=lo, hi=hi))
    nrng = np.random.default_rng(5)
    for _ in range(15):
        f_q = nrng.integers(-4000, 4000, size=n)
        await write_features(dut, f_q)
        ys, _ = await run_frame(dut)
        assert ys == fp.predict_int(f_q, qp).tolist()


def runner():
    proj = Path(__file__).resolve().parent.parent
    sim = get_runner(os.getenv("SIM", "icarus"))
    params = dict(N_FEAT=QP.n_feat, N_OUT=QP.n_out, WEIGHTS_FILE=f'"{proj}/data/weights.mem"',
                  BIAS_FILE=f'"{proj}/data/bias.mem"', CFG_FILE=f'"{proj}/data/cfg.mem"')
    sim.build(sources=[proj / "hdl" / "ridge_mac.sv"], hdl_toplevel="ridge_mac", always=True,
              parameters=params, timescale=("1ns", "1ps"), build_args=["-g2012"])
    sim.test(hdl_toplevel="ridge_mac", test_module=Path(__file__).stem, parameters=params)


if __name__ == "__main__":
    runner()
