#!/usr/bin/env python3
"""Board stand-in for testing llama_host.py without hardware.

Speaks the uart_axi_bridge protocol on a pseudo-terminal, keeps a DDR3 image and the
kernel's control registers, and "runs" the kernel by answering with the CPU reference
argmax/logits for the requested position, after checking that the host programmed every
register to the values the hardware expects and that the uploaded checkpoint is intact.
usage: fake_board.py <ref_prefix> <checkpoint>   (prints the pty path, then serves forever)
"""
import os
import struct
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llama_host import REG, load_model, layout  # noqa: E402

CTRL = 0x8000_0000


def main():
    ref, ckpt = sys.argv[1], sys.argv[2]
    raw, cfg = load_model(ckpt)
    lay = layout(raw, cfg)
    amax = open(ref + ".argmax", "rb").read()
    amax = struct.unpack(f"<{len(amax) // 4}i", amax)
    logits_f = open(ref + ".logits", "rb")
    mem = {}  # word address -> 4 bytes
    regs = {}
    done = [False]

    master, slave = os.openpty()
    print(os.ttyname(slave), flush=True)

    def rd(n):
        b = b""
        while len(b) < n:
            b += os.read(master, n - len(b))
        return b

    def run_kernel():
        for k in ("dim", "hidden_dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "seq_len",
                  "shared_weights", "head_size", "kv_dim", "kv_mul"):
            assert regs.get(REG[k]) == cfg[k], f"register {k} = {regs.get(REG[k])} expected {cfg[k]}"
        for k in ("w_off", "kv_off", "rope_off", "logits_off"):
            assert regs.get(REG[k]) == lay[k], f"register {k} wrong"
        assert regs.get(REG["mem"]) == 0
        img = b"".join(mem.get(i, b"\0\0\0\0") for i in range((len(raw) + 3) // 4))
        assert img[:len(raw)] == raw, "checkpoint in DDR3 differs from the file"
        pos = regs[REG["pos"]]
        if regs.get(REG["write_logits"]):
            v = cfg["vocab_size"]
            logits_f.seek(4 * v * pos)
            lg = logits_f.read(4 * v)
            for i in range(v):
                mem[lay["logits_off"] + i] = lg[4 * i:4 * i + 4]
        regs[REG["ap_return"]] = amax[pos] & 0xFFFFFFFF
        done[0] = True

    def word_read(a):
        if a & 0x8000_0000:
            off = a & 0xFF
            if off == REG["ctrl"]:
                v = 0x4 | (0x2 if done[0] else 0)
                done[0] = False  # ap_done is clear-on-read
                return v
            return regs.get(off, 0)
        return struct.unpack("<I", mem.get(a // 4, b"\0\0\0\0"))[0]

    def serve():
        while True:
            c = rd(1)
            if c not in b"WRS":
                os.write(master, b"?")
                continue
            addr, n = struct.unpack("<II", rd(8))
            if c == b"W":
                data = rd(4 * n)
                s = 0
                for i in range(n):
                    w = data[4 * i:4 * i + 4]
                    s += struct.unpack("<I", w)[0]
                    a = addr + 4 * i
                    if a & 0x8000_0000:
                        regs[a & 0xFF] = struct.unpack("<I", w)[0]
                        if (a & 0xFF) == REG["ctrl"] and struct.unpack("<I", w)[0] & 1:
                            run_kernel()
                    else:
                        mem[a // 4] = w
                os.write(master, b"K" + struct.pack("<I", s & 0xFFFFFFFF))
            elif c == b"R":
                os.write(master, b"".join(struct.pack("<I", word_read(addr + 4 * i)) for i in range(n)))
            else:
                s = sum(word_read(addr + 4 * i) for i in range(n)) & 0xFFFFFFFF
                os.write(master, b"S" + struct.pack("<I", s))

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    t.join()


if __name__ == "__main__":
    main()
