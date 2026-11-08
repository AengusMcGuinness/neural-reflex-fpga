#!/usr/bin/env python3
"""Run the FP4 stories260K accelerator on the Urbana and check it.

The whole generate loop runs on the FPGA: the host writes the prompt tokens, starts the
kernel once, waits, and reads back every position's argmax plus the FPGA's own busy-cycle
counter, so tokens/second is measured in hardware (the UART is not in the timed path).

Checks: the board's tokens must equal the C++ golden model's (fp4/golden/*.tokens) exactly.

usage (the Urbana's UART is the FT2232H's second channel, e.g. /dev/cu.usbserial-<serial>1):
  fp4_host.py --port /dev/cu.usbserial-XXXX1 check            # all golden cases, exact match
  fp4_host.py --port /dev/cu.usbserial-XXXX1 run -i "Once upon a time" -n 256
"""
import argparse
import os
import re
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from llama_host import Board, Tokenizer  # noqa: E402

STATUS = 0x4000_0000
MAGIC = 0xF4F4_0001
MAC_TOKS = {"clang -O3": 8225, "clang -O3 -ffast-math -mcpu=native": 15937}  # measured, M3 Pro


def ctrl_map():
    """Register offsets straight from the Vitis HLS generated driver header."""
    h = open(os.path.join(ROOT, "fp4", "xllama_fp4_hw.h")).read()
    d = dict((k, int(v, 16)) for k, v in re.findall(r"#define XLLAMA_FP4_CONTROL_(\w+)\s+0x([0-9a-fA-F]+)", h))
    need = ["ADDR_AP_CTRL", "ADDR_GIE", "ADDR_IER", "ADDR_ISR", "ADDR_PROMPT_BASE",
            "ADDR_N_PROMPT_DATA", "ADDR_STEPS_DATA", "ADDR_TOKENS_BASE", "ADDR_N_OUT_DATA"]
    missing = [k for k in need if k not in d]
    if missing:
        raise RuntimeError(f"xllama_fp4_hw.h lacks {missing}")
    return d


def run_board(board, R, prompt, steps, timeout=10.0):
    if not 1 <= len(prompt) <= 64 or not 1 <= steps <= 256:
        raise ValueError("1..64 prompt tokens, 1..256 steps")
    board.write_words(R["ADDR_PROMPT_BASE"], struct.pack(f"<{len(prompt)}i", *prompt))
    board.wr32(R["ADDR_N_PROMPT_DATA"], len(prompt))
    board.wr32(R["ADDR_STEPS_DATA"], steps)
    board.wr32(R["ADDR_GIE"], 1)
    board.wr32(R["ADDR_IER"], 1)                      # ap_done -> interrupt (stops the counter)
    isr = board.rd32(R["ADDR_ISR"])
    if isr & 1:
        board.wr32(R["ADDR_ISR"], 1)                  # toggle-on-write: clear a stale ap_done
    board.wr32(R["ADDR_AP_CTRL"], 1)                  # ap_start
    t0 = time.time()
    while not board.rd32(R["ADDR_AP_CTRL"]) & 0x2:    # ap_done (clear on read)
        if time.time() - t0 > timeout:
            raise IOError("kernel did not finish")
    n = struct.unpack("<i", board.read_words(R["ADDR_N_OUT_DATA"], 1))[0]
    if n < 1:
        raise IOError(f"kernel rejected the arguments (n_out={n})")
    toks = list(struct.unpack(f"<{n}i", board.read_words(R["ADDR_TOKENS_BASE"], n)))
    lo, hi, hz = struct.unpack("<3I", board.read_words(STATUS, 3))
    return toks, (hi << 32) | lo, hz


def text(tok, prompt, amax):
    out, t = b"", prompt[0]
    for pos in range(len(amax)):
        nxt = prompt[pos + 1] if pos < len(prompt) - 1 else amax[pos]
        if nxt == 1:
            break
        out += Tokenizer.safe(tok.decode(t, nxt))
        t = nxt
    return out


def report(n, cycles, hz):
    secs = cycles / hz
    tps = n / secs
    print(f"FPGA: {n} positions in {cycles} cycles at {hz / 1e6:.0f} MHz = {secs * 1e3:.3f} ms "
          f"-> {tps:,.0f} tok/s (measured on chip)")
    for k, v in MAC_TOKS.items():
        print(f"      vs MacBook llama2.c ({k}): {v:,} tok/s -> {tps / v:.2f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=2_000_000)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="run every golden case and require exact token equality")
    r = sub.add_parser("run")
    r.add_argument("-i", "--prompt", default="Once upon a time")
    r.add_argument("-n", "--steps", type=int, default=256)
    args = ap.parse_args()

    board = Board(args.port, args.baud)
    magic = board.rd32(STATUS + 0xC)
    if magic != MAGIC:
        raise SystemExit(f"not the FP4 bitstream (status magic 0x{magic:08x})")
    R = ctrl_map()
    tok = Tokenizer(os.path.join(ROOT, "fp4", "tok512.bin"), 512)  # llama2.c's stories260K tokenizer

    if args.cmd == "run":
        prompt = tok.encode(args.prompt.encode("utf-8"))
        toks, cycles, hz = run_board(board, R, prompt, args.steps)
        sys.stdout.buffer.write(text(tok, prompt, toks) + b"\n")
        report(len(toks), cycles, hz)
        return 0

    gdir = os.path.join(ROOT, "fp4", "golden")
    fails = 0
    best = None
    for f in sorted(os.listdir(gdir)):
        if not f.endswith(".tokens"):
            continue
        b = open(os.path.join(gdir, f), "rb").read()
        v = struct.unpack(f"<{len(b) // 4}i", b)
        np_, n = v[0], v[1]
        prompt, want = list(v[2:2 + np_]), list(v[2 + np_:2 + np_ + n])
        assert tok.encode(open(os.path.join(gdir, f[:-7] + ".prompt.txt"), "rb").read()) == prompt
        toks, cycles, hz = run_board(board, R, prompt, 256)
        ok = toks == want
        fails += not ok
        first = next((i for i, (a, w) in enumerate(zip(toks, want)) if a != w), None)
        print(f"{f[:-7]:8s} {'MATCH' if ok else 'MISMATCH'}  {len(toks)} positions, "
              f"{len(toks) / (cycles / hz):,.0f} tok/s" + ("" if ok else f", first difference at pos {first}"))
        if best is None or f.startswith("s260K"):  # report the "Once upon a time" run
            best = (len(toks), cycles, hz, prompt, toks)
    n, cycles, hz, prompt, toks = best
    print()
    sys.stdout.buffer.write(text(tok, prompt, toks)[:400] + b"...\n\n")
    report(n, cycles, hz)
    print(f"\n{'ALL CASES MATCH the C++ golden model bit for bit' if not fails else f'{fails} CASES MISMATCH'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
