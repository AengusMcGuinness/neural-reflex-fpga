#!/usr/bin/env python3
"""Host side for llama2.c on the Urbana FPGA.

The FPGA does the whole forward pass and the greedy argmax; this script only uploads the
checkpoint, drives the generate() loop from karpathy/llama2.c run.c, and does tokenization
(a line-for-line port of run.c's encode/decode/safe_printf, checked by test_tokenizer.py).

Board protocol (uart_axi_bridge.sv), little-endian, 2 Mbaud 8N1:
  'W' addr n data[n]  -> 'K' sum32     'R' addr n -> data[n]     'S' addr n -> 'S' sum32
DDR3 lives at byte 0x0000_0000; the kernel's AXI-Lite control registers at 0x8000_0000.

Examples:
  llama_host.py --port /dev/cu.usbserial-XXXXB memtest
  llama_host.py --port ... run models/stories15M.bin models/tokenizer.bin models/stories15M.rope \
      -i "Once upon a time" -n 200 --ref ref_out/s15M     # ref = ref_dump prefix, checks every token
"""
import argparse
import os
import random
import re
import struct
import sys
import time

CTRL = 0x8000_0000
REG = {  # from the Vitis HLS generated llama_forward_control_s_axi.v address map
    "ctrl": 0x00, "ap_return": 0x10, "mem": 0x18, "w_off": 0x20, "kv_off": 0x28,
    "rope_off": 0x30, "logits_off": 0x38, "dim": 0x40, "hidden_dim": 0x48, "n_layers": 0x50,
    "n_heads": 0x58, "n_kv_heads": 0x60, "vocab_size": 0x68, "seq_len": 0x70,
    "shared_weights": 0x78, "head_size": 0x80, "kv_dim": 0x88, "kv_mul": 0x90,
    "token": 0x98, "pos": 0xA0, "write_logits": 0xA8,
}
MASK32 = 0xFFFFFFFF


# --------------------------------------------------------------------------- board link
class Board:
    def __init__(self, port, baud=2_000_000, timeout=5.0):
        import serial  # pyserial
        self.s = serial.Serial(port, baudrate=baud, timeout=timeout)
        self.s.reset_input_buffer()

    def _read(self, n):
        b = self.s.read(n)
        if len(b) != n:
            raise IOError(f"board timeout: wanted {n} bytes, got {len(b)}")
        return b

    def write_words(self, addr, words_le_bytes):
        """words_le_bytes: bytes, multiple of 4. Returns after the board confirms the checksum."""
        assert addr % 4 == 0 and len(words_le_bytes) % 4 == 0
        n = len(words_le_bytes) // 4
        self.s.write(b"W" + struct.pack("<II", addr, n) + words_le_bytes)
        ack = self._read(5)
        if ack[0:1] == b"!":
            raise IOError("board receive FIFO overflowed (press btn[0] / reprogram and retry)")
        if ack[0:1] != b"K":
            raise IOError(f"bad write ack {ack!r}")
        got = struct.unpack("<I", ack[1:])[0]
        exp = sum32(words_le_bytes)
        if got != exp:
            raise IOError(f"write checksum mismatch at 0x{addr:08x}: board {got:08x} host {exp:08x}")

    def read_words(self, addr, n):
        self.s.write(b"R" + struct.pack("<II", addr, n))
        return self._read(4 * n)

    def sum_words(self, addr, n):
        self.s.write(b"S" + struct.pack("<II", addr, n))
        ack = self._read(5)
        if ack[0:1] != b"S":
            raise IOError(f"bad sum ack {ack!r}")
        return struct.unpack("<I", ack[1:])[0]

    def wr32(self, addr, value):
        self.write_words(addr, struct.pack("<I", value & MASK32))

    def rd32(self, addr):
        return struct.unpack("<I", self.read_words(addr, 1))[0]


def sum32(b):
    return sum(struct.unpack(f"<{len(b) // 4}I", b)) & MASK32


def upload(board, addr, data, label, chunk_words=4096):
    data = data + b"\0" * (-len(data) % 4)
    total = len(data)
    t0 = time.time()
    for off in range(0, total, 4 * chunk_words):
        board.write_words(addr + off, data[off:off + 4 * chunk_words])
        done = min(off + 4 * chunk_words, total)
        rate = done / max(time.time() - t0, 1e-6)
        print(f"\r  {label}: {done / 1e6:7.2f}/{total / 1e6:.2f} MB  {rate / 1e3:6.1f} kB/s", end="", file=sys.stderr)
    print(file=sys.stderr)
    # independent end-to-end check: the board re-reads the whole region from DDR3 and sums it
    got = board.sum_words(addr, total // 4)
    if got != sum32(data):
        raise IOError(f"{label}: DDR3 read-back sum {got:08x} != host {sum32(data):08x}")
    print(f"  {label}: verified in DDR3 (sum32 {got:08x})", file=sys.stderr)


# --------------------------------------------------------------------------- tokenizer
class Tokenizer:
    """Port of run.c build_tokenizer/encode/decode/safe_printf (byte strings throughout)."""

    def __init__(self, path, vocab_size):
        with open(path, "rb") as f:
            self.max_token_length = struct.unpack("<i", f.read(4))[0]
            self.vocab, self.scores = [], []
            for _ in range(vocab_size):
                score = struct.unpack("<f", f.read(4))[0]
                n = struct.unpack("<i", f.read(4))[0]
                s = f.read(n)
                self.scores.append(score)
                self.vocab.append(s.split(b"\0", 1)[0])  # C string semantics
        self.lookup = {}
        for i, s in enumerate(self.vocab):
            self.lookup.setdefault(s, i)

    def encode(self, text: bytes, bos=True, eos=False):
        toks = []
        if bos:
            toks.append(1)
        if text:
            toks.append(self.lookup.get(b" ", -1))
        buf = bytearray()
        i = 0
        while i < len(text):
            c = text[i]
            if (c & 0xC0) != 0x80:
                buf = bytearray()
            buf.append(c)
            nxt = text[i + 1] if i + 1 < len(text) else 0
            if (nxt & 0xC0) == 0x80 and len(buf) < 4:
                i += 1
                continue
            tid = self.lookup.get(bytes(buf), -1)
            if tid != -1:
                toks.append(tid)
            else:
                toks.extend(b + 3 for b in buf)
            buf = bytearray()
            i += 1
        best_init = struct.unpack("<f", struct.pack("<f", -1e10))[0]
        while True:
            best_score, best_id, best_idx = best_init, -1, -1
            for k in range(len(toks) - 1):
                tid = self.lookup.get(self.vocab[toks[k]] + self.vocab[toks[k + 1]], -1)
                if tid != -1 and self.scores[tid] > best_score:
                    best_score, best_id, best_idx = self.scores[tid], tid, k
            if best_idx == -1:
                break
            toks[best_idx] = best_id
            del toks[best_idx + 1]
        if eos:
            toks.append(2)
        return toks

    _byte_re = re.compile(rb"<0x([0-9A-Fa-f]{1,2})")

    def decode(self, prev, token) -> bytes:
        piece = self.vocab[token]
        if prev == 1 and piece[:1] == b" ":
            piece = piece[1:]
        m = self._byte_re.match(piece)
        if m:
            piece = bytes([int(m.group(1), 16)])
        return piece

    @staticmethod
    def safe(piece: bytes) -> bytes:
        if not piece:
            return b""
        if len(piece) == 1:
            b = piece[0]
            if not (0x20 <= b <= 0x7E or b in (0x09, 0x0A, 0x0B, 0x0C, 0x0D)):
                return b""
        return piece


# --------------------------------------------------------------------------- model + layout
def load_model(path):
    raw = open(path, "rb").read()
    dim, hidden, layers, heads, kvh, vocab, seq = struct.unpack("<7i", raw[:28])
    cfg = dict(dim=dim, hidden_dim=hidden, n_layers=layers, n_heads=heads, n_kv_heads=kvh,
               vocab_size=abs(vocab), seq_len=seq, shared_weights=1 if vocab > 0 else 0)
    cfg["head_size"] = dim // heads
    cfg["kv_dim"] = dim * kvh // heads
    cfg["kv_mul"] = heads // kvh
    return raw, cfg


def layout(file_bytes, cfg):
    """Float-word offsets; identical to tb_llama.cpp so the board runs the tested layout."""
    a16 = lambda x: (x + 15) & ~15
    words = (len(file_bytes) + 3) // 4
    kv_off = a16(words)
    rope_off = kv_off + 2 * cfg["n_layers"] * cfg["seq_len"] * cfg["kv_dim"]
    logits_off = a16(rope_off + cfg["seq_len"] * cfg["head_size"])
    end = logits_off + cfg["vocab_size"]
    if end * 4 > 256 * 1024 * 1024:
        raise ValueError("model does not fit in the 256 MB DDR3")
    return dict(w_off=7, kv_off=kv_off, rope_off=rope_off, logits_off=logits_off, end=end)


def forward(board, token, pos, write_logits, timeout=60.0):
    board.wr32(CTRL + REG["token"], token)
    board.wr32(CTRL + REG["pos"], pos)
    board.wr32(CTRL + REG["write_logits"], 1 if write_logits else 0)
    board.wr32(CTRL + REG["ctrl"], 1)  # ap_start
    t0 = time.time()
    while True:
        c = board.rd32(CTRL + REG["ctrl"])
        if c & 0x2:  # ap_done (clear on read)
            break
        if time.time() - t0 > timeout:
            raise IOError(f"kernel did not finish (ctrl=0x{c:x})")
    r = board.rd32(CTRL + REG["ap_return"])
    return r - (1 << 32) if r & 0x8000_0000 else r


# --------------------------------------------------------------------------- commands
def cmd_memtest(board, args):
    rnd = random.Random(1234)
    base = args.base
    n = args.words
    data = bytes(rnd.getrandbits(8) for _ in range(4 * n))
    print(f"memtest: writing {n} words at 0x{base:08x}", file=sys.stderr)
    upload(board, base, data, "pattern", chunk_words=1024)
    back = board.read_words(base, n)
    bad = [i for i in range(n) if back[4 * i:4 * i + 4] != data[4 * i:4 * i + 4]]
    # single-word overwrites inside full lines: checks the byte-lane selects (DM) leave
    # the neighbouring words of a 128-bit DDR3 line untouched
    for i in range(0, min(n, 256), 3):
        w = rnd.getrandbits(32)
        board.wr32(base + 4 * i, w)
        data = data[:4 * i] + struct.pack("<I", w) + data[4 * i + 4:]
    back2 = board.read_words(base, min(n, 256))
    bad2 = [i for i in range(min(n, 256)) if back2[4 * i:4 * i + 4] != data[4 * i:4 * i + 4]]
    print(f"memtest: {len(bad)} bad words after bulk write, {len(bad2)} after partial writes")
    return 0 if not bad and not bad2 else 1


def cmd_run(board, args):
    raw, cfg = load_model(args.checkpoint)
    lay = layout(raw, cfg)
    rope = open(args.rope, "rb").read()
    if len(rope) != 4 * cfg["seq_len"] * cfg["head_size"]:
        raise ValueError("rope table does not match this model (make it with ref_dump on Linux)")
    tok = Tokenizer(args.tokenizer, cfg["vocab_size"])
    print(f"model {cfg}", file=sys.stderr)

    if not args.skip_upload:
        upload(board, 0, raw, "checkpoint")
        upload(board, 4 * lay["rope_off"], rope, "rope table")
    for k in ("dim", "hidden_dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "seq_len",
              "shared_weights", "head_size", "kv_dim", "kv_mul"):
        board.wr32(CTRL + REG[k], cfg[k])
    board.wr32(CTRL + REG["mem"], 0)
    for k in ("w_off", "kv_off", "rope_off", "logits_off"):
        board.wr32(CTRL + REG[k], lay[k])

    ref = None
    if args.ref:
        ref_prompt = list(struct.unpack(f"<{os.path.getsize(args.ref + '.prompt') // 4}i", open(args.ref + ".prompt", "rb").read()))
        ref_amax = list(struct.unpack(f"<{os.path.getsize(args.ref + '.argmax') // 4}i", open(args.ref + ".argmax", "rb").read()))
        ref = (ref_prompt, ref_amax, args.ref + ".logits")

    prompt = tok.encode(args.prompt.encode("utf-8"), bos=True, eos=False)
    if ref and prompt != ref[0]:
        raise ValueError(f"prompt tokens {prompt} differ from the reference {ref[0]}")
    steps = min(args.steps, cfg["seq_len"]) if args.steps > 0 else cfg["seq_len"]

    out = sys.stdout.buffer
    token, pos, mism, t_start = prompt[0], 0, 0, None
    worst = 0.0
    while pos < steps:
        hw = forward(board, token, pos, write_logits=args.check_logits)
        if hw < 0:
            raise IOError(f"kernel rejected the configuration (returned {hw})")
        if ref:
            if pos < len(ref[1]) and hw != ref[1][pos]:
                mism += 1
                print(f"\nMISMATCH pos {pos}: fpga {hw} cpu {ref[1][pos]}", file=sys.stderr)
            if args.check_logits and pos < len(ref[1]):
                v = cfg["vocab_size"]
                lg = struct.unpack(f"<{v}f", board.read_words(4 * lay["logits_off"], v))
                with open(ref[2], "rb") as f:
                    f.seek(4 * v * pos)
                    rl = struct.unpack(f"<{v}f", f.read(4 * v))
                worst = max(worst, max(abs(a - b) for a, b in zip(lg, rl)))
        nxt = prompt[pos + 1] if pos < len(prompt) - 1 else hw
        pos += 1
        if nxt == 1:
            break
        out.write(Tokenizer.safe(tok.decode(token, nxt)))
        out.flush()
        token = nxt
        if t_start is None:
            t_start = time.time()
    out.write(b"\n")
    if pos > 1 and t_start:
        print(f"achieved tok/s: {(pos - 1) / (time.time() - t_start):.3f}", file=sys.stderr)
    if ref:
        msg = f"checked {min(pos, len(ref[1]))} positions against CPU llama2.c: {mism} argmax mismatches"
        if args.check_logits:
            msg += f", max |logit diff| {worst:.3g}"
        print(msg, file=sys.stderr)
        return 0 if mism == 0 else 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="serial port of the Urbana UART, e.g. /dev/cu.usbserial-XXXXB")
    ap.add_argument("--baud", type=int, default=2_000_000)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("memtest", help="write/read a random pattern in DDR3")
    m.add_argument("--base", type=lambda s: int(s, 0), default=0x0F00_0000)
    m.add_argument("--words", type=int, default=16384)
    r = sub.add_parser("run", help="generate text with the FPGA")
    r.add_argument("checkpoint")
    r.add_argument("tokenizer")
    r.add_argument("rope", help="RoPE table from ref_dump (<prefix>.rope) for this model")
    r.add_argument("-i", "--prompt", default="")
    r.add_argument("-n", "--steps", type=int, default=256)
    r.add_argument("--ref", help="ref_dump prefix: check every argmax against CPU llama2.c")
    r.add_argument("--check-logits", action="store_true", help="also read back and compare logits (slow)")
    r.add_argument("--skip-upload", action="store_true", help="weights already in DDR3")
    args = ap.parse_args()
    board = Board(args.port, args.baud)
    sys.exit(cmd_memtest(board, args) if args.cmd == "memtest" else cmd_run(board, args))


if __name__ == "__main__":
    main()
