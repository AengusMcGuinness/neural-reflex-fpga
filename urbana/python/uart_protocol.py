"""Host side of the UART frame protocol (see hdl/frame_rx.sv and hdl/frame_tx.sv).

Host -> FPGA:  A5 5A 01 <N_FEAT int16 LE> CHK                       features, FPGA replies with outputs
               A5 5A 02 <w int16 LE> <b int48 LE> <cfg int16 LE> CHK   load weights, FPGA replies 82
FPGA -> host:  5A A5 81 <N_OUT int16 LE> CHK | 5A A5 82 00 CHK | 5A A5 EE <cmd> CHK
CHK = sum of the command/type byte and payload bytes, mod 256.
"""
import struct

import fixed_point as fp

SYNC_TX = b"\xA5\x5A"
SYNC_RX = b"\x5A\xA5"


def _frame(cmd: int, payload: bytes) -> bytes:
    return SYNC_TX + bytes([cmd]) + payload + bytes([(cmd + sum(payload)) & 0xFF])


def features_frame(f_q) -> bytes:
    return _frame(0x01, struct.pack(f"<{len(f_q)}h", *(int(v) for v in f_q)))


def load_frame(qp: "fp.QuantParams") -> bytes:
    w = struct.pack(f"<{qp.w_q.size}h", *(int(v) for v in qp.w_q.ravel()))
    b = b"".join((int(v) & ((1 << 48) - 1)).to_bytes(6, "little") for v in qp.b_q)
    cfg = [v for j in range(qp.n_out) for v in (qp.shift[j], qp.lo[j], qp.hi[j])]
    c = struct.pack(f"<{len(cfg)}h", *(int(v) for v in cfg))
    return _frame(0x02, w + b + c)


class Parser:
    """Feed received bytes; yields ("result", [y...]) / ("load_ok", None) / ("error", cmd)."""

    def __init__(self, n_out: int):
        self.n_out = n_out
        self.buf = bytearray()

    def feed(self, data: bytes):
        self.buf += data
        while True:
            start = self.buf.find(SYNC_RX)
            if start < 0:
                del self.buf[:-1]
                return
            del self.buf[:start]
            if len(self.buf) < 3:
                return
            kind = self.buf[2]
            n = {0x81: 2 * self.n_out, 0x82: 1, 0xEE: 1}.get(kind)
            if n is None:
                del self.buf[:2]
                continue
            if len(self.buf) < 4 + n:
                return
            payload, chk = bytes(self.buf[3:3 + n]), self.buf[3 + n]
            del self.buf[:4 + n]
            if (kind + sum(payload)) & 0xFF != chk:
                yield ("bad_checksum", kind)
            elif kind == 0x81:
                yield ("result", list(struct.unpack(f"<{self.n_out}h", payload)))
            elif kind == 0x82:
                yield ("load_ok", None)
            else:
                yield ("error", payload[0])
