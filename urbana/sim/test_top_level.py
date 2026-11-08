"""End to end through real UART bits: feature frames in, decoded frames out, checked against the golden model.

The UART runs at 10 Mbaud here (10 clock cycles per bit) so the sim is quick; the board uses 921600.
Run from emg_decoder/sim:  python test_top_level.py
"""
import os
import sys
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge, Timer
from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
import fixed_point as fp  # noqa: E402
import uart_protocol as up  # noqa: E402

QP = fp.QuantParams.load(ROOT / "build/quant_params.json")
CLK_NS = 10
BAUD = 10_000_000
BIT_NS = CLK_NS * (100_000_000 // BAUD)


async def uart_send(dut, data: bytes):
    for byte in data:
        for bit in [0] + [(byte >> i) & 1 for i in range(8)] + [1]:
            dut.uart_rxd.value = bit
            await Timer(BIT_NS, unit="ns")


async def uart_listen(dut, out: bytearray):
    while True:
        await FallingEdge(dut.uart_txd)
        await Timer(BIT_NS // 2, unit="ns")
        if dut.uart_txd.value != 0:
            continue
        byte = 0
        for i in range(8):
            await Timer(BIT_NS, unit="ns")
            byte |= int(dut.uart_txd.value) << i
        await Timer(BIT_NS, unit="ns")
        assert dut.uart_txd.value == 1, "missing stop bit"
        out.append(byte)


async def setup(dut):
    cocotb.start_soon(Clock(dut.clk_100mhz, CLK_NS, unit="ns").start())
    dut.uart_rxd.value = 1
    dut.btn.value = 1
    await ClockCycles(dut.clk_100mhz, 5)
    dut.btn.value = 0
    await ClockCycles(dut.clk_100mhz, 5)
    rx = bytearray()
    cocotb.start_soon(uart_listen(dut, rx))
    return rx, up.Parser(QP.n_out)


async def expect(dut, rx, parser, n_msgs, timeout_us=200):
    got = []
    for _ in range(timeout_us * 10):
        await Timer(100, unit="ns")
        if rx:
            got += list(parser.feed(bytes(rx)))
            rx.clear()
        if len(got) >= n_msgs:
            return got
    raise AssertionError(f"timed out; got {got}")


def led_bar(y0, lo, hi):
    return sum(1 << k for k in range(16) if y0 >= lo + (((hi - lo) * (k + 1)) >> 4))


@cocotb.test()
async def test_feature_frames(dut):
    rx, parser = await setup(dut)
    X = np.load(ROOT / "build/calib_used.npy")
    idx = np.random.default_rng(1).choice(len(X), 8, replace=False)
    for f_q in fp.quantize_features(X[idx], QP):
        await uart_send(dut, up.features_frame(f_q))
        (kind, ys), = await expect(dut, rx, parser, 1)
        golden = fp.predict_int(f_q, QP).tolist()
        assert kind == "result" and ys == golden, f"{kind} {ys} != {golden}"
        await ClockCycles(dut.clk_100mhz, 3)
        assert int(dut.led.value) == led_bar(ys[0], int(QP.lo[0]), int(QP.hi[0]))
        dut._log.info(f"decoded {dict(zip(QP.target_names, ys))} ticks  (golden {golden})  led={int(dut.led.value):016b}")


@cocotb.test()
async def test_bad_checksum_then_resync(dut):
    rx, parser = await setup(dut)
    good = up.features_frame(np.zeros(QP.n_feat, int))
    bad = good[:-1] + bytes([good[-1] ^ 0xFF])
    await uart_send(dut, bad)
    assert await expect(dut, rx, parser, 1) == [("error", 0x01)]
    assert int(dut.rgb1.value) == 0b111
    await uart_send(dut, b"\x00\xA5\x13\x37" + good)       # junk before a good frame
    (kind, ys), = await expect(dut, rx, parser, 1)
    assert kind == "result" and ys == fp.predict_int(np.zeros(QP.n_feat, int), QP).tolist()
    assert int(dut.rgb1.value) == 0


@cocotb.test()
async def test_load_weights_over_uart(dut):
    rx, parser = await setup(dut)
    rng = np.random.default_rng(7)
    d = dict(n_feat=QP.n_feat, n_out=QP.n_out, feat_exp=[0] * QP.n_feat,
             w_q=rng.integers(-2000, 2000, size=(QP.n_out, QP.n_feat)).tolist(),
             b_q=rng.integers(-(1 << 30), 1 << 30, size=QP.n_out).tolist(),
             w_frac=[12] * QP.n_out, out_frac=[0] * QP.n_out, shift=[12] * QP.n_out,
             lo=[-30000] * QP.n_out, hi=[30000] * QP.n_out)
    new = fp.QuantParams(d)
    await uart_send(dut, up.load_frame(new))
    assert await expect(dut, rx, parser, 1, timeout_us=2000) == [("load_ok", None)]
    f_q = rng.integers(-3000, 3000, size=QP.n_feat)
    await uart_send(dut, up.features_frame(f_q))
    (kind, ys), = await expect(dut, rx, parser, 1)
    assert ys == fp.predict_int(f_q, new).tolist()


def runner():
    proj = Path(__file__).resolve().parent.parent
    sim = get_runner(os.getenv("SIM", "icarus"))
    params = dict(N_FEAT=QP.n_feat, N_OUT=QP.n_out, BAUD_RATE=BAUD,
                  WEIGHTS_FILE=f'"{proj}/data/weights.mem"', BIAS_FILE=f'"{proj}/data/bias.mem"',
                  CFG_FILE=f'"{proj}/data/cfg.mem"')
    sources = [proj / "hdl" / f for f in ("uart_receive.sv", "uart_transmit.sv", "frame_rx.sv",
                                          "frame_tx.sv", "ridge_mac.sv", "top_level.sv")]
    sim.build(sources=sources, hdl_toplevel="top_level", always=True, parameters=params,
              timescale=("1ns", "1ps"), build_args=["-g2012"])
    sim.test(hdl_toplevel="top_level", test_module=Path(__file__).stem, parameters=params)


if __name__ == "__main__":
    runner()
