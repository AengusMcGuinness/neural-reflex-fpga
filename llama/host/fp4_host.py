#!/usr/bin/env python3
"""Run the FP4 stories260K accelerator on the Urbana and check it.

The whole generate loop runs on the FPGA: the host writes the prompt tokens, starts the
kernel once, waits, and reads back every position's argmax plus the FPGA's own busy-cycle
counter, so tokens/second is measured in hardware (the UART is not in the timed path).

Checks: the board's tokens must equal the C++ golden model's (fp4/golden/*.tokens) exactly.

usage (the Urbana's UART is the FT2232H's second channel, e.g. /dev/cu.usbserial-<serial>1):
  fp4_host.py --port /dev/cu.usbserial-XXXX1 stream          # endless stories, Ctrl-C to stop
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
from llama_host import Board, Tokenizer, sum32  # noqa: E402

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


def wait_idle(board, R, timeout=2.0):
    """Wait until the kernel is idle with no start pending, e.g. after a host that was killed
    mid-run, so that run's results can't be mistaken for ours."""
    t0 = time.time()
    while True:
        c = board.rd32(R["ADDR_AP_CTRL"])
        if c & 0x4 and not c & 0x1:  # ap_idle, ap_start clear
            return
        if time.time() - t0 > timeout:
            raise IOError(f"kernel stays busy (ctrl=0x{c:x}); press btn[0] or reload the bitstream")
        time.sleep(0.005)


def run_board(board, R, prompt, steps, timeout=10.0):
    if not 1 <= len(prompt) <= 64 or not 1 <= steps <= 256:
        raise ValueError("1..64 prompt tokens, 1..256 steps")
    wait_idle(board, R)
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


def run_chunk(board, R, prompt, steps, expect_s=0.0, timeout=10.0):
    """One FPGA run with a single USB round trip in the common case: all writes go out
    without waiting for their acks; after sleeping for the expected FPGA time, one batch
    reads ap_ctrl, n_out, the tokens and the cycle counter. Assumes GIE/IER are set and an
    ap_done is pending from the previous run (stream() guarantees both)."""
    cmds = [(R["ADDR_PROMPT_BASE"], struct.pack(f"<{len(prompt)}i", *prompt)),
            (R["ADDR_N_PROMPT_DATA"], struct.pack("<I", len(prompt))),
            (R["ADDR_STEPS_DATA"], struct.pack("<I", steps)),
            (R["ADDR_ISR"], struct.pack("<I", 1)),       # clear the previous ap_done (toggle)
            (R["ADDR_AP_CTRL"], struct.pack("<I", 1))]   # ap_start
    board.s.write(b"".join(b"W" + struct.pack("<II", a, len(d) // 4) + d for a, d in cmds))
    t0 = time.time()
    if expect_s > 0:
        time.sleep(expect_s)
    reads = [(R["ADDR_AP_CTRL"], 1), (R["ADDR_N_OUT_DATA"], 1), (R["ADDR_TOKENS_BASE"], steps), (STATUS, 2)]
    acks_pending = True
    while True:
        board.s.write(b"".join(b"R" + struct.pack("<II", a, n) for a, n in reads))
        if acks_pending:
            for a, d in cmds:
                ack = board._read(5)
                if ack[:1] != b"K" or struct.unpack("<I", ack[1:])[0] != sum32(d):
                    raise IOError(f"bad ack {ack!r} for write to 0x{a:x}")
            acks_pending = False
        ctrl = struct.unpack("<I", board._read(4))[0]
        n = struct.unpack("<i", board._read(4))[0]
        raw = board._read(4 * steps)
        lo, hi = struct.unpack("<2I", board._read(8))
        if ctrl & 0x2:  # ap_done: n_out, tokens and the counter were read after completion
            break
        if time.time() - t0 > timeout:
            raise IOError("kernel did not finish")
    if n < 1:
        raise IOError(f"kernel rejected the arguments (n_out={n})")
    return list(struct.unpack(f"<{n}i", raw[:4 * n])), (hi << 32) | lo


OPENERS = ["Once upon a time", "One day", "Lily and Ben", "The little dog", "Tom had a red ball",
           "There was a big tree", "Sue liked to sing", "A cat named Max", "It was a sunny day",
           "The bird flew"]


def stream(board, R, tok, opener, context, window, limit, quiet=False):
    """Endless generation: each chunk re-prompts the FPGA with BOS + the last `context`-1
    tokens and computes `window` positions. When a story ends (BOS), start the next one
    from the next opener. Prints live; stops after `limit` new tokens (0 = until Ctrl-C)."""
    if not 1 <= context <= 64 or not context < window <= 256:
        raise ValueError("need 1 <= context <= 64 and context < window <= 256")
    wait_idle(board, R)
    board.wr32(R["ADDR_GIE"], 1)
    board.wr32(R["ADDR_IER"], 1)
    if not board.rd32(R["ADDR_ISR"]) & 1:
        # make sure an ap_done is pending, so run_chunk's toggle always clears it
        run_board(board, R, [1], 1)
    hz = board.rd32(STATUS + 8)
    out = sys.stdout.buffer
    k = 0
    hist = tok.encode(opener.encode("utf-8"))[1:]       # without BOS
    prev = 1
    for t in hist:
        out.write(Tokenizer.safe(tok.decode(prev, t)))
        prev = t
    out.flush()
    new_total = pos_total = cyc_total = 0
    t_start = time.time()
    expect = 0.0
    story_len, max_story = 0, 800
    try:
        while limit == 0 or new_total < limit:
            prompt = [1] + hist[-(context - 1):] if context > 1 else [1]
            toks, cycles = run_chunk(board, R, prompt, window, expect)
            expect = 0.97 * cycles / hz  # next chunk: same window, about the same FPGA time
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
            # greedy decoding can fall into a loop that never emits BOS ("the broken broken ...");
            # treat a stuck window (few distinct tokens) or a very long story like an ending
            story_len += len(toks) - (len(prompt) - 1)
            tail = hist[-64:]
            if not ended and ((len(tail) == 64 and len(set(tail)) <= 12) or story_len > max_story):
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
    fsec = cyc_total / hz
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
    st = sub.add_parser("stream", help="generate forever (Ctrl-C to stop), re-prompting with recent tokens")
    st.add_argument("-i", "--prompt", default="Once upon a time")
    st.add_argument("-c", "--context", type=int, default=48, help="tokens carried into each chunk (<= 64)")
    st.add_argument("-w", "--window", type=int, default=256, help="positions per FPGA run (<= 256)")
    st.add_argument("-n", "--tokens", type=int, default=0, help="stop after this many new tokens (0 = never)")
    args = ap.parse_args()

    board = Board(args.port, args.baud)
    magic = board.rd32(STATUS + 0xC)
    if magic != MAGIC:
        raise SystemExit(f"not the FP4 bitstream (status magic 0x{magic:08x})")
    R = ctrl_map()
    tok = Tokenizer(os.path.join(ROOT, "fp4", "tok512.bin"), 512)  # llama2.c's stories260K tokenizer

    if args.cmd == "stream":
        stream(board, R, tok, args.prompt, args.context, args.window, args.tokens)
        return 0

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
