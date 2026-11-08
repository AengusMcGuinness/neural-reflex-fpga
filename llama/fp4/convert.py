#!/usr/bin/env python3
"""Convert karpathy/llama2.c stories260K.bin into the FP4 accelerator's on-chip ROM image.

Format (everything the kernel and its C++ golden model read; see llama_fp4.h):
  * Weights are FP4 E2M1 codes (sign<<3 | idx, idx -> |value| {0,.5,1,1.5,2,3,4,6}) in blocks
    of 16 along the input dimension. Each block has an 8-bit scale (e<<4 | m): block scale =
    m * 2^e * base, with base one float per tensor. The kernel uses integer weights
    kw = 2*value = {0,1,2,3,4,6,8,12} and a per-tensor constant BH = base/2.
  * A "chunk" is 64 weights (4 blocks) = 32 code bytes (low nibble = even element) + 4 scale
    bytes = 36 bytes (288 bits). CROM holds chunks; the kernel reads chunks 2w and 2w+1 per cycle.
  * RMSNorm gains are folded into the following matmul weights (W' = W diag(g)); the final
    norm gain into the classifier. Hidden 172 is zero-padded to 192 (3 chunks).
  * Word order per layer: qkv (wq 64 rows, wk 32, wv 32; 2 rows/word), wo (64 rows, 2/word),
    w13 (word r = [w1 row r, w3 row r], 192 words), w2 (64 rows x 3 chunks, chunk stream
    packed 2/word, 96 words). Then the classifier = the input embedding (512 rows, 2/word).
usage: convert.py stories260K.bin out_header.h
"""
import struct
import sys

import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
DIM, HID, HIDP, NL, NH, NKV, VOCAB, SEQ_HW = 64, 172, 192, 5, 8, 4, 512, 256
HS = DIM // NH
KVD = DIM * NKV // NH


def load(path):
    raw = open(path, "rb").read()
    hdr = struct.unpack("<7i", raw[:28])
    dim, hidden, layers, heads, kvh, vocab, seq = hdr
    assert (dim, hidden, layers, heads, kvh, abs(vocab)) == (DIM, HID, NL, NH, NKV, VOCAB), hdr
    assert vocab > 0, "expects shared classifier weights"
    f = np.frombuffer(raw, dtype="<f4", offset=28).astype(np.float64)
    o = 0

    def take(*shape):
        nonlocal o
        n = int(np.prod(shape))
        a = f[o:o + n].reshape(shape)
        o += n
        return a

    w = {}
    w["emb"] = take(VOCAB, DIM)
    w["rms_att"] = take(NL, DIM)
    w["wq"] = take(NL, DIM, DIM)
    w["wk"] = take(NL, KVD, DIM)
    w["wv"] = take(NL, KVD, DIM)
    w["wo"] = take(NL, DIM, DIM)
    w["rms_ffn"] = take(NL, DIM)
    w["w1"] = take(NL, HID, DIM)
    w["w2"] = take(NL, DIM, HID)
    w["w3"] = take(NL, HID, DIM)
    w["rms_final"] = take(DIM)
    return w


def quantize(W):
    """W: (rows, n), n % 16 == 0 -> codes (rows, n) uint8, scales (rows, n/16) uint8, base f32."""
    rows, n = W.shape
    B = W.reshape(rows, n // 16, 16)
    amax = np.abs(B).max(axis=2)
    s = amax / 6.0
    base = np.float32(s.max() / (15 * 2 ** 15)) if s.max() > 0 else np.float32(1.0)
    t = s / np.float64(base)
    e = np.where(t >= 8, np.floor(np.log2(np.maximum(t, 1e-300))) - 3, 0).astype(np.int64)
    m = np.rint(t / 2.0 ** e).astype(np.int64)
    bump = m > 15
    e = np.where(bump, e + 1, e)
    m = np.where(bump, np.rint(t / 2.0 ** e).astype(np.int64), m)
    assert e.min() >= 0 and e.max() <= 15 and m.min() >= 0 and m.max() <= 15
    scale = (m * 2.0 ** e * np.float64(base))[:, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(scale > 0, np.abs(B) / scale, 0.0)
    v = np.minimum(v, 6.0)
    d = np.abs(v[..., None] - E2M1)                  # distance to each E2M1 magnitude
    idx = np.argmin(d + (np.arange(8) % 2) * 1e-12, axis=-1)  # ties -> even code
    codes = (idx | ((B < 0) & (idx > 0)) * 8).astype(np.uint8).reshape(rows, n)
    scales = ((e << 4) | m).astype(np.uint8)
    return codes, scales, base


def chunks(codes, scales):
    """-> list of 36-byte chunks, row-major, 64 weights each."""
    rows, n = codes.shape
    out = []
    for r in range(rows):
        for c in range(n // 64):
            cc = codes[r, 64 * c:64 * c + 64]
            b = bytes(int(cc[2 * j]) | (int(cc[2 * j + 1]) << 4) for j in range(32))
            out.append(b + bytes(int(x) for x in scales[r, 4 * c:4 * c + 4]))
    return out


def main():
    w = load(sys.argv[1])
    words, bh = [], np.zeros((NL, 7), np.float32)  # BH: wq wk wv wo w1 w3 w2
    for l in range(NL):
        g_att, g_ffn = w["rms_att"][l], w["rms_ffn"][l]
        qc, qs, qb = quantize(w["wq"][l] * g_att)
        kc, ks, kb = quantize(w["wk"][l] * g_att)
        vc, vs, vb = quantize(w["wv"][l] * g_att)
        oc, os_, ob = quantize(w["wo"][l])
        w1 = np.zeros((HIDP, DIM)); w1[:HID] = w["w1"][l] * g_ffn
        w3 = np.zeros((HIDP, DIM)); w3[:HID] = w["w3"][l] * g_ffn
        w2 = np.zeros((DIM, HIDP)); w2[:, :HID] = w["w2"][l]
        c1, s1, b1 = quantize(w1)
        c3, s3, b3 = quantize(w3)
        c2, s2, b2 = quantize(w2)
        bh[l] = [qb / 2, kb / 2, vb / 2, ob / 2, b1 / 2, b3 / 2, b2 / 2]
        qkv = chunks(qc, qs) + chunks(kc, ks) + chunks(vc, vs)   # 128 chunks
        words += [qkv[2 * i] + qkv[2 * i + 1] for i in range(64)]
        wo = chunks(oc, os_)
        words += [wo[2 * i] + wo[2 * i + 1] for i in range(32)]
        a1, a3 = chunks(c1, s1), chunks(c3, s3)
        words += [a1[r] + a3[r] for r in range(HIDP)]
        a2 = chunks(c2, s2)                                     # 192 chunks
        words += [a2[2 * i] + a2[2 * i + 1] for i in range(96)]
    # classifier = input embedding (shared, unfolded): token t is chunk 3840 + t. The final
    # RMSNorm gain (GF) is applied to the activations instead of folded into these weights.
    cc, cs, cb = quantize(w["emb"])
    cls = chunks(cc, cs)
    words += [cls[2 * i] + cls[2 * i + 1] for i in range(256)]
    assert len(words) == 2176 and all(len(x) == 72 for x in words)

    rope = np.zeros((SEQ_HW, 8), np.float32)
    for pos in range(SEQ_HW):
        for f in range(HS // 2):
            freq = 1.0 / (10000.0 ** ((2 * f) / HS))
            rope[pos, 2 * f] = np.cos(pos * freq)
            rope[pos, 2 * f + 1] = np.sin(pos * freq)
    tf = np.array([2.0 ** (i / 256) for i in range(256)], np.float32)
    t16 = np.array([int(round(2.0 ** (i / 256) * 32768)) for i in range(256)], np.int64)
    assert t16[0] == 32768 and t16.max() < 65536  # 2^(i/256) * 2^15, i in [0, 256)

    def repr_f(x):  # 9 significant digits round-trip any float32 exactly
        v = float(np.float32(x))
        assert np.isfinite(v)
        s = "%.9g" % v
        return s + ("f" if ("." in s or "e" in s) else ".0f")

    def barr(rows):
        return ",\n".join("{" + ",".join(str(b) for b in r) + "}" for r in rows)

    with open(sys.argv[2], "w") as h:
        h.write("// generated by fp4/convert.py from stories260K.bin - do not edit\n#pragma once\n")
        h.write("#include <stdint.h>\n")
        # word w = chunks (2w, 2w+1): lane A ROM holds the even chunks, lane B ROM the odd ones
        # each lane ROM = 2048-word main part (packs exactly into 2Kx9 block RAMs) + 128-word tail
        A, B = [wd[:36] for wd in words], [wd[36:] for wd in words]
        h.write(f"static const uint8_t CROMA_M[2048][36] = {{\n{barr(A[:2048])}}};\n")
        h.write(f"static const uint8_t CROMB_M[2048][36] = {{\n{barr(B[:2048])}}};\n")
        # both 128-word tails in one memory: row = lane * 128 + (word - 2048)
        h.write(f"static const uint8_t CROMT[256][36] = {{\n{barr(A[2048:] + B[2048:])}}};\n")
        h.write("static const float BH[5][7] = {" + ",".join(
            "{" + ",".join(repr_f(x) for x in bh[l]) + "}" for l in range(NL)) + "};\n")
        h.write(f"static const float CLS_BH = {repr_f(np.float32(cb / 2))};\n")
        h.write("static const float GF[64] = {" + ",".join(repr_f(x) for x in w["rms_final"]) + "};\n")
        h.write("static const float ROPE[2048] = {" + ",".join(repr_f(x) for x in rope.ravel()) + "};\n")
        h.write("static const float TF[256] = {" + ",".join(repr_f(x) for x in tf) + "};\n")
        h.write("static const uint16_t T16[256] = {" + ",".join(str(int(x)) for x in t16) + "};\n")
        h.write(f"static const float C0 = {repr_f(np.float32(np.log2(np.e) / np.sqrt(8.0)))};\n")
        h.write(f"static const float LOG2E = {repr_f(np.float32(np.log2(np.e)))};\n")
    print(f"wrote {sys.argv[2]}: 2176 weight words (2 single-port ROMs), embedding = classifier")


if __name__ == "__main__":
    main()
