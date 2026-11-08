#!/usr/bin/env python3
"""Run the FP4 stories260K accelerator on the Opal Kelly XEM7310 over FrontPanel USB.

A run is one PipeIn write and one PipeOut read: the host queues the prompt writes, ap_start,
a WAIT (the FPGA polls ap_done itself) and every read, and the FPGA answers with n_out, the
tokens and its busy-cycle counter. Tokens/second is measured on the FPGA.

Checks: the board's tokens must equal the C++ golden model's (fp4/golden/*.tokens) exactly.

usage (needs the CS1410 FrontPanel Python binding, see README.md):
  fp4_ok_host.py check                        # all golden cases, exact match
  fp4_ok_host.py run -i "Once upon a time" -n 256
  fp4_ok_host.py stream                       # endless stories, Ctrl-C to stop
  fp4_ok_host.py --bit ../bitstreams/llama_xem7310_top.bit check   # program first
"""
import argparse
import os
import re
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # xem7310/
sys.path.insert(0, str(ROOT.parent / "llama" / "host"))
from llama_host import Tokenizer  # noqa: E402  (pyserial is only imported by its Board class)

MAGIC = 0xF4F4_0002
OP_NOP, OP_WRITE, OP_READ, OP_WAIT = 0, 1, 2, 3
PIPE_IN, PIPE_OUT = 0x80, 0xA0
MAC_TOKS = {"clang -O3": 8225, "clang -O3 -ffast-math -mcpu=native": 15937}  # measured, M3 Pro
URBANA_TOKS = 8981                                                          # measured, 256 pos


def frontpanel():
    sdk = Path(os.environ.get("NEO_FRONTPANEL_PYTHON", str(Path.home() / "cs1410_lab/lib/macOS")))
    sys.path.insert(0, str(sdk))
    try:
        import ok  # noqa: E402
    except ImportError as e:
        raise SystemExit(f"cannot load the FrontPanel binding from {sdk} ({e}); "
                         f"set DYLD_LIBRARY_PATH={sdk}") from None
    return ok


def ctrl_map():
    """Register offsets from the Vitis HLS generated driver header."""
    h = (ROOT / "fp4" / "xllama_fp4_hw.h").read_text()
    return dict((k, int(v, 16)) for k, v in re.findall(r"#define XLLAMA_FP4_CONTROL_(\w+)\s+0x([0-9a-fA-F]+)", h))


def hdr(op, addr=0, count=1, status=False):
    assert 1 <= count <= 2048 and 0 <= addr < 1 << 16
    return (op << 28) | (int(status) << 27) | ((count - 1) << 16) | addr


class Board:
    def __init__(self, bitfile=None, serial=""):
        ok = frontpanel()
        self.ok = ok
        self.dev = ok.okCFrontPanel()
        if self.dev.GetDeviceCount() < 1:
            raise SystemExit("no FrontPanel board found (is the XEM7310 plugged in?)")
        if self.dev.OpenBySerial(serial) != 0:
            raise SystemExit("could not open the FrontPanel board")
        if bitfile:
            if self.dev.ConfigureFPGA(str(bitfile)) != 0:
                raise SystemExit(f"FrontPanel rejected {bitfile}")
        if not self.dev.IsFrontPanelEnabled():
            raise SystemExit("FrontPanel is not enabled: load the bitstream with --bit")
        self.reset()
        lo, hi, hz, magic = self.transact([], [(0, 4, True)])
        if magic != MAGIC:
            raise SystemExit(f"not the XEM7310 FP4 bitstream (status magic 0x{magic:08x})")
        self.hz = hz

    def reset(self):
        self.dev.SetWireInValue(0x00, 1, 1)
        self.dev.UpdateWireIns()
        time.sleep(0.002)
        self.dev.SetWireInValue(0x00, 0, 1)
        self.dev.UpdateWireIns()
        time.sleep(0.002)

    def status(self):
        self.dev.UpdateWireOuts()
        return self.dev.GetWireOutValue(0x20), self.dev.GetWireOutValue(0x21)

    def transact(self, writes, reads, wait=False, expect_s=0.0, timeout=10.0):
        """writes: [(addr, [words])], reads: [(addr, count, status)]; returns the read words.
        All of it is one PipeIn write and (after the FPGA has answered) one PipeOut read."""
        cmd = []
        for addr, words in writes:
            for k in range(0, len(words), 2048):
                part = words[k:k + 2048]
                cmd.append(hdr(OP_WRITE, addr + 4 * k, len(part)))
                cmd += [w & 0xFFFFFFFF for w in part]
        if wait:
            cmd.append(hdr(OP_WAIT))
        nresp = 0
        for addr, count, st in reads:
            cmd.append(hdr(OP_READ, addr, count, st))
            nresp += count
        pad = -nresp % 4                       # transfers are multiples of 16 bytes
        if pad:
            cmd.append(hdr(OP_READ, 0xC, pad, True))
        cmd += [hdr(OP_NOP)] * (-len(cmd) % 4)
        if len(cmd) > 2048 or nresp + pad > 2048:
            raise ValueError("batch larger than the FPGA's FIFOs")
        buf = bytearray(struct.pack(f"<{len(cmd)}I", *cmd))
        n = self.dev.WriteToPipeIn(PIPE_IN, buf)
        if n != len(buf):
            raise IOError(f"WriteToPipeIn returned {n}")
        if expect_s > 0:
            time.sleep(expect_s)
        want = nresp + pad
        t0 = time.time()
        while True:
            have, flags = self.status()
            if flags & 0xC:
                raise IOError(f"FIFO overflow (flags 0x{flags:x}); reset the board")
            if have >= want:
                break
            if time.time() - t0 > timeout:
                raise IOError(f"FPGA answered {have} of {want} words (flags 0x{flags:x})")
            time.sleep(0.0002)
        out = bytearray(4 * want)
        n = self.dev.ReadFromPipeOut(PIPE_OUT, out)
        if n != len(out):
            raise IOError(f"ReadFromPipeOut returned {n}")
        return list(struct.unpack(f"<{want}I", out))[:nresp]


def run_board(board, R, prompt, steps, expect_s=0.0):
    """One generate() run on the FPGA: returns (tokens, busy cycles)."""
    if not 1 <= len(prompt) <= 64 or not 1 <= steps <= 256:
        raise ValueError("1..64 prompt tokens, 1..256 steps")
    v = board.transact(
        [(R["ADDR_PROMPT_BASE"], list(prompt)), (R["ADDR_N_PROMPT_DATA"], [len(prompt)]),
         (R["ADDR_STEPS_DATA"], [steps]), (R["ADDR_AP_CTRL"], [1])],
        [(R["ADDR_N_OUT_DATA"], 1, False), (R["ADDR_TOKENS_BASE"], steps, False), (0, 2, True)],
        wait=True, expect_s=expect_s)
    n = struct.unpack("<i", struct.pack("<I", v[0]))[0]
    if n < 1:
        raise IOError(f"kernel rejected the arguments (n_out={n})")
    return v[1:1 + n], v[1 + steps] | (v[2 + steps] << 32)


OPENERS = ["Once upon a time", "One day", "Lily and Ben", "The little dog", "Tom had a red ball",
           "There was a big tree", "Sue liked to sing", "A cat named Max", "It was a sunny day",
           "The bird flew"]


def stream(board, R, tok, opener, context, window, limit):
    """Endless generation (same algorithm as llama/host/fp4_host.py): each run re-prompts with
    BOS + the last context-1 tokens and computes `window` positions; a story ends at BOS, in a
    greedy loop (<= 12 distinct tokens in the last 64) or after 800 positions."""
    if not 1 <= context <= 64 or not context < window <= 256:
        raise ValueError("need 1 <= context <= 64 and context < window <= 256")
    out = sys.stdout.buffer
    k = 0
    hist = tok.encode(opener.encode("utf-8"))[1:]
    prev = 1
    for t in hist:
        out.write(Tokenizer.safe(tok.decode(prev, t)))
        prev = t
    out.flush()
    new_total = pos_total = cyc_total = 0
    t_start = time.time()
    expect = 0.0
    story_len = 0
    try:
        while limit == 0 or new_total < limit:
            prompt = [1] + hist[-(context - 1):] if context > 1 else [1]
            toks, cycles = run_board(board, R, prompt, window, expect)
            expect = 0.9 * cycles / board.hz
            cyc_total += cycles
            pos_total += len(toks)
            ended = False
            for t in toks[len(prompt) - 1:]:
                if t == 1:
                    ended = True
                    break
                out.write(Tokenizer.safe(tok.decode(prev, t)))
                prev = t
                hist.append(t)
                new_total += 1
            out.flush()
            story_len += len(toks) - (len(prompt) - 1)
            tail = hist[-64:]
            if not ended and ((len(tail) == 64 and len(set(tail)) <= 12) or story_len > 800):
                ended = True
            if ended:
                story_len = 0
                k = (k + 1) % len(OPENERS)
                hist = tok.encode(OPENERS[k].encode("utf-8"))[1:]
                out.write(b"\n\n")
                prev = 1
                for t in hist:
                    out.write(Tokenizer.safe(tok.decode(prev, t)))
                    prev = t
                out.flush()
    except KeyboardInterrupt:
        pass
    wall = time.time() - t_start
    fsec = cyc_total / board.hz
    print(f"\n\n[{new_total:,} new tokens in {wall:.2f} s wall = {new_total / max(wall, 1e-9):,.0f} tok/s end to end; "
          f"FPGA busy {fsec * 1e3:.1f} ms = {new_total / max(fsec, 1e-9):,.0f} new tok/s "
          f"({pos_total / max(fsec, 1e-9):,.0f} positions/s incl. re-read context); "
          f"context {context}, window {window}]", file=sys.stderr)
    return new_total, wall, fsec


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
    print(f"      vs Urbana FP4 (same model, 64 lanes): {URBANA_TOKS:,} tok/s -> {tps / URBANA_TOKS:.2f}x")
    for k, v in MAC_TOKS.items():
        print(f"      vs MacBook llama2.c ({k}): {v:,} tok/s -> {tps / v:.2f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bit", help="program this bitstream first")
    ap.add_argument("--serial", default="", help="board serial (default: the first one)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="run every golden case and require exact token equality")
    r = sub.add_parser("run")
    r.add_argument("-i", "--prompt", default="Once upon a time")
    r.add_argument("-n", "--steps", type=int, default=256)
    st = sub.add_parser("stream", help="generate forever (Ctrl-C to stop)")
    st.add_argument("-i", "--prompt", default="Once upon a time")
    st.add_argument("-c", "--context", type=int, default=48)
    st.add_argument("-w", "--window", type=int, default=256)
    st.add_argument("-n", "--tokens", type=int, default=0, help="stop after this many new tokens (0 = never)")
    args = ap.parse_args()

    board = Board(args.bit, args.serial)
    R = ctrl_map()
    tok = Tokenizer(str(ROOT / "fp4" / "tok512.bin"), 512)

    if args.cmd == "stream":
        stream(board, R, tok, args.prompt, args.context, args.window, args.tokens)
        return 0
    if args.cmd == "run":
        prompt = tok.encode(args.prompt.encode("utf-8"))
        t0 = time.time()
        toks, cycles = run_board(board, R, prompt, args.steps)
        wall = time.time() - t0
        sys.stdout.buffer.write(text(tok, prompt, toks) + b"\n")
        report(len(toks), cycles, board.hz)
        print(f"      end to end (USB included): {wall * 1e3:.2f} ms -> {len(toks) / wall:,.0f} tok/s")
        return 0

    gdir = ROOT / "fp4" / "golden"
    fails = 0
    best = None
    for f in sorted(gdir.glob("*.tokens")):
        b = f.read_bytes()
        v = struct.unpack(f"<{len(b) // 4}i", b)
        np_, n = v[0], v[1]
        prompt, want = list(v[2:2 + np_]), list(v[2 + np_:2 + np_ + n])
        assert tok.encode((gdir / (f.stem + ".prompt.txt")).read_bytes()) == prompt
        toks, cycles = run_board(board, R, prompt, 256)
        good = toks == want
        fails += not good
        first = next((i for i, (a, w) in enumerate(zip(toks, want)) if a != w), None)
        print(f"{f.stem:8s} {'MATCH' if good else 'MISMATCH'}  {len(toks)} positions, "
              f"{len(toks) / (cycles / board.hz):,.0f} tok/s" + ("" if good else f", first difference at pos {first}"))
        if best is None or f.stem == "s260K":
            best = (len(toks), cycles, prompt, toks)
    n, cycles, prompt, toks = best
    print()
    sys.stdout.buffer.write(text(tok, prompt, toks)[:400] + b"...\n\n")
    report(n, cycles, board.hz)
    print(f"\n{'ALL CASES MATCH the C++ golden model bit for bit' if not fails else f'{fails} CASES MISMATCH'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
