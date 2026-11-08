// FP4 llama2 accelerator (stories260K), Vitis HLS source and bit-exact golden model.
//
// Numerics (this code is the definition; the board must reproduce it bit for bit):
//  * weights   : FP4 E2M1 codes, blocks of 16, block scale m*2^e (4+4 bits), one float per
//                tensor (BH = base/2); RMSNorm gains folded into the weights (convert.py).
//  * matmul    : activations quantized to int8 with a power-of-two scale; the dot product is
//                exact integer arithmetic: sum_blocks (sum_i kw_i*x_i) * m << e. The integer
//                result becomes float by i2f48 (round-to-nearest-even) and one fmul.
//  * RMSNorm   : only its scalar 1/sqrt(mean(x^2)+1e-5) is needed (gain folded, int8
//                quantization is scale invariant); mean(x^2) uses the quantized x, exactly.
//  * attention : q, k, v quantized per head to int8 with power-of-two scales; K/V cache int8
//                on chip. Scores are integers; softmax uses a 2^(i/256) table (T16) on a
//                fixed-point exponent; the weighted sum of V is an integer accumulation.
//  * SwiGLU    : sigmoid(h) = 1 / (1 + 2^(-h*log2 e)) with the 2^(i/256) float table (TF).
//  * argmax    : on the classifier's integer dot products (the output scale is positive).
// Floats are only combined by single IEEE-754 add/mul/div/sqrt operations (-ffp-contract=off);
// the bit-level helpers flush subnormals to zero like the FPGA floating-point cores.
// The golden outputs come from running this file as C++ (Vitis HLS C simulation).
#include "llama_fp4.h"
#include "model_rom.h"
#include <ap_int.h>
#include <math.h>

typedef ap_int<8> s8;
typedef ap_int<40> acc_t;   // matmul accumulators, |.| < 2^38
typedef ap_int<48> aacc_t;  // attention value accumulators, |.| < 2^40

// ------------------------------------------------------------------ exact bit-level helpers
static inline uint32_t f2u(float f) {
    union { float f; uint32_t u; } c;
    c.f = f;
    return c.u;
}
static inline float u2f(uint32_t u) {
    union { float f; uint32_t u; } c;
    c.u = u;
    return c.f;
}

// f * 2^e (exact), flushing results that would be subnormal to signed zero
static float ldexp_b(float f, int e) {
#pragma HLS INLINE
    uint32_t u = f2u(f);
    uint32_t s = u & 0x80000000u;
    int E = (int)((u >> 23) & 255u);
    if (E == 0) return u2f(s);
    int E2 = E + e;
    if (E2 <= 0) return u2f(s);
    if (E2 >= 255) return u2f(s | 0x7f800000u);
    return u2f(s | ((uint32_t)E2 << 23) | (u & 0x7fffffu));
}

// |v| < 2^47 -> float, round to nearest, ties to even
static float i2f48(aacc_t v) {
#pragma HLS INLINE
    if (v == 0) return 0.0f;
    bool neg = v < 0;
    ap_uint<48> a = neg ? (ap_uint<48>)(-v) : (ap_uint<48>)v;
    int lz = a.countLeadingZeros();
    int msb = 47 - lz;
    ap_uint<48> n = a << lz;                 // leading one at bit 47
    ap_uint<24> mant = n(47, 24);
    bool half = n[23];
    bool sticky = n(22, 0) != 0;
    ap_uint<25> m2 = (ap_uint<25>)mant + (ap_uint<25>)((half && (sticky || mant[0])) ? 1 : 0);
    if (m2[24]) {
        m2 = m2 >> 1;
        msb++;
    }
    uint32_t bits = (neg ? 0x80000000u : 0u) | ((uint32_t)(msb + 127) << 23) |
                    (uint32_t)m2(22, 0).to_uint();
    return u2f(bits);
}

// smallest e with amax <= 127 * 2^e, for amax given as |x| bits; FP_SENT if amax is zero
static int pow2exp(uint32_t ab) {
#pragma HLS INLINE
    int E = (int)((ab >> 23) & 255u);
    if (E == 0) return FP_SENT;
    uint32_t M = (ab & 0x7fffffu) | 0x800000u;
    int e = E - 133 + (M > (127u << 17) ? 1 : 0);
    return e < -127 ? -127 : e;
}

// round-half-away-from-zero(x / 2^e), clamped to [-127, 127]
static int roundq(float x, int e) {
#pragma HLS INLINE
    uint32_t u = f2u(x);
    int E = (int)((u >> 23) & 255u);
    if (E == 0 || e == FP_SENT) return 0;
    uint32_t M = (u & 0x7fffffu) | 0x800000u;
    int sh = 150 - E + e;
    int r;
    if (sh <= 0) r = 127;
    else if (sh > 24) r = 0;
    else r = (int)((M + (1u << (sh - 1))) >> sh);
    if (r > 127) r = 127;
    return (u >> 31) ? -r : r;
}

// floor(x * 2^se), magnitude clamped to 2^20
static int floor_s(float x, int se) {
#pragma HLS INLINE
    uint32_t u = f2u(x);
    int E = (int)((u >> 23) & 255u);
    if (E == 0) return 0;
    uint32_t M = (u & 0x7fffffu) | 0x800000u;
    int sh = 150 - E - se;
    uint32_t mag;
    bool frac;
    if (sh <= 0) {
        mag = (sh < -7) ? (1u << 20) : (M << (-sh));
        frac = false;
    } else if (sh >= 25) {
        mag = 0;
        frac = true;
    } else {
        mag = M >> sh;
        frac = (M & ((1u << sh) - 1u)) != 0;
    }
    if (mag > (1u << 20)) mag = 1u << 20;
    int m = (int)mag;
    return (u >> 31) ? -m - (frac ? 1 : 0) : m;
}

static inline uint32_t fabs_bits(float f) { return f2u(f) & 0x7fffffffu; }

// kw * x for one FP4 code (kw = 2*E2M1 value in {0,1,2,3,4,6,8,12}), no multiplier
static inline ap_int<13> wmul(ap_uint<4> code, s8 x) {
#pragma HLS INLINE
    ap_int<13> xx = x, m;
    switch ((int)code(2, 0)) {
    case 0: m = 0; break;
    case 1: m = xx; break;
    case 2: m = xx * 2; break;
    case 3: m = xx * 3; break;
    case 4: m = xx * 4; break;
    case 5: m = xx * 6; break;
    case 6: m = xx * 8; break;
    default: m = xx * 12; break;
    }
    return code[3] ? (ap_int<13>)(-m) : m;
}

// same as wmul, with x3 = 3*x precomputed once per activation (shared by all weight rows)
static inline ap_int<13> wmul3(ap_uint<4> code, s8 x, ap_int<10> x3) {
#pragma HLS INLINE
    ap_int<13> xx = x, t = x3;
    ap_int<13> tab[8] = {0, xx, (ap_int<13>)(xx << 1), t, (ap_int<13>)(xx << 2),
                         (ap_int<13>)(t << 1), (ap_int<13>)(xx << 3), (ap_int<13>)(t << 2)};
#pragma HLS ARRAY_PARTITION variable = tab complete
    ap_int<13> m = tab[(int)code(2, 0)];
    return code[3] ? (ap_int<13>)(-m) : m;
}

// ------------------------------------------------------------------ on-chip state
static float x[FP_DIM];             // residual stream            (8 banks)
static float o[FP_DIM];             // attention output           (8 banks)
static float ybufE[FP_DIM / 2];     // matmul rows for residual: even rows
static float ybufO[FP_DIM / 2];     //                            odd rows
static float qkvbE[16][4];          // q heads 0-7, k 8-11, v 12-15: even elements
static float qkvbO[16][4];          //                               odd elements
static float hb[FP_HIDP];           // SwiGLU output              (8 banks)
static s8 mq0[64], mq1[64], mq2[64];             // quantized matmul input, 3 slices
static ap_int<10> m30[64], m31[64], m32[64];     // 3 * the same values
static int s2;                      // sum q^2 of the last quant()
static s8 qi[FP_DIM];               // quantized, rotated q       (registers)
static int eq[FP_NH];
// K/V cache rows: 32 int8 + 4 exponents (288-bit words); layers 0-3 and layer 4 are separate
// memories so each packs into block RAM without a mostly-empty power-of-two depth.
static uint8_t kc[4 * FP_SEQ][36], kc4[FP_SEQ][36];
static uint8_t vc[4 * FP_SEQ][36], vc4[FP_SEQ][36];
static int ekmax[FP_NL][FP_NKV];    // max key exponent over cached positions
static int evmin[FP_NL][FP_NKV];    // min value exponent over cached positions (127 if none)
static ap_int<19> sp[2][FP_SEQ][4]; // rescaled attention scores, per group of 4 heads

// ------------------------------------------------------------------ blocks
// byte k of ROM word w (0..2175) of lane 0 (even chunks) or lane 1 (odd chunks)
static inline ap_uint<8> rom(int lane, int w, int k) {
#pragma HLS INLINE
    ap_uint<8> m = lane ? CROMB_M[w & 2047][k] : CROMA_M[w & 2047][k];
    ap_uint<8> t = CROMT[(lane << 7) | (w & 127)][k];
    return w < 2048 ? m : t;
}
static inline void mq_set(int i, int q) {
#pragma HLS INLINE
    s8 v = (s8)q;
    ap_int<10> v3 = (ap_int<10>)(v * 3);
    if (i < 64) { mq0[i] = v; m30[i] = v3; }
    else if (i < 128) { mq1[i - 64] = v; m31[i - 64] = v3; }
    else { mq2[i - 128] = v; m32[i - 128] = v3; }
}

// FP4 dot product of chunk c (ROM lane c&1, word c>>1) with input slice seg
static acc_t chunk_dot(int c, int seg) {
#pragma HLS INLINE
    int lane = c & 1, w = c >> 1;
    acc_t acc = 0;
    for (int b = 0; b < 4; b++) {
#pragma HLS UNROLL
        ap_int<18> bs = 0;
        for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
            int j = 16 * b + i;
            ap_uint<8> byte = rom(lane, w, j >> 1);
            ap_uint<4> code = (j & 1) ? byte(7, 4) : byte(3, 0);
            s8 xv = seg == 0 ? mq0[j] : (seg == 1 ? mq1[j] : mq2[j]);
            ap_int<10> x3 = seg == 0 ? m30[j] : (seg == 1 ? m31[j] : m32[j]);
            bs += wmul3(code, xv, x3);
        }
        ap_uint<8> sc = rom(lane, w, 32 + b);
        ap_int<23> t = bs * (ap_int<6>)sc(3, 0);
        acc += ((acc_t)t) << (int)sc(7, 4);
    }
    return acc;
}

enum { MM_QKV = 0, MM_WO = 1, MM_W13 = 2, MM_W2 = 3, MM_CLS = 4 };

// The single matmul engine: one 64-weight chunk (64 FP4 MACs) per cycle + each mode's output.
//   QKV: chunk i = row i (q 0-63, k 64-95, v 96-127)        -> qkvbE/qkvbO
//   WO : chunk i = row i                                    -> ybufE/ybufO
//   W13: chunks alternate w1 row i/2, w3 row i/2            -> hb (SwiGLU)
//   W2 : row r = chunks 3r, 3r+1, 3r+2 (input slices 0,1,2) -> ybufE/ybufO
//   CLS: chunk i = row i, returns the argmax row
static int matmul(int mode, int cbase, int nchunks, int ex, float c0, float c1, float c2) {
#pragma HLS INLINE off
    acc_t best = 0, racc = 0;
    int besti = 0, ph = 0, r2 = 0;
    float h1 = 0.0f;
mm_loop:
    for (int i = 0; i < nchunks; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 64 max = 512
        int seg = mode == MM_W2 ? ph : 0;
        acc_t a = chunk_dot(cbase + i, seg);
        acc_t conv = a;
        if (mode == MM_W2) {
            conv = ph == 0 ? a : (acc_t)(racc + a);
            racc = conv;
        }
        float c = mode == MM_QKV ? (i < 64 ? c0 : (i < 96 ? c1 : c2))
                                 : (mode == MM_W13 && (i & 1) ? c1 : c0);
        float y = ldexp_b(i2f48(conv), ex) * c;
        if (mode == MM_QKV) {
            if (i & 1) qkvbO[i >> 3][(i & 7) >> 1] = y;
            else qkvbE[i >> 3][(i & 7) >> 1] = y;
        } else if (mode == MM_WO) {
            if (i & 1) ybufO[i >> 1] = y;
            else ybufE[i >> 1] = y;
        } else if (mode == MM_W2) {
            if (ph == 2) {
                if (r2 & 1) ybufO[r2 >> 1] = y;
                else ybufE[r2 >> 1] = y;
            }
        } else if (mode == MM_W13) {
            if ((i & 1) == 0) {
                h1 = y;
            } else {
                float h3 = y;
                float u = h1 * LOG2E;
                int kk = floor_s(u2f(f2u(u) ^ 0x80000000u), 8); // floor(-h1*log2e*256)
                if (kk < -4096) kk = -4096;
                if (kk > 4095) kk = 4095;
                float e2 = ldexp_b(TF[kk & 255], kk >> 8);      // ~ 2^(-h1*log2e) = exp(-h1)
                float den = 1.0f + e2;
                float sig = 1.0f / den;
                hb[i >> 1] = (h1 * sig) * h3;
            }
        } else if (i == 0 || a > best) { // MM_CLS: first index of the maximum (run.c argmax)
            best = a;
            besti = i;
        }
        if (ph == 2) r2++;
        ph = ph == 2 ? 0 : ph + 1;
    }
    return besti;
}

static void residual() {
#pragma HLS INLINE off
res:
    for (int j = 0; j < FP_DIM / 2; j++) {
#pragma HLS PIPELINE II = 1
        x[2 * j] = x[2 * j] + ybufE[j];
        x[2 * j + 1] = x[2 * j + 1] + ybufO[j];
    }
}

// quantize x (src 0, 64), o (src 1, 64) or hb (src 2, 192) into the matmul input slices;
// returns the power-of-two exponent and sets s2 = sum q^2
static int quant(int src, int ngroups) {
#pragma HLS INLINE off
    uint32_t amax = 0;
q_max:
    for (int g = 0; g < 4 * ngroups; g++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 32 max = 96
        uint32_t m = 0;
        for (int k = 0; k < 2; k++) {
            int j = 2 * g + k;
            float v = src == 2 ? hb[j] : (src == 1 ? o[j & 63] : x[j & 63]);
            uint32_t a = fabs_bits(v);
            m = a > m ? a : m;
        }
        amax = m > amax ? m : amax;
    }
    int e = pow2exp(amax);
    int ss = 0;
q_q:
    for (int g = 0; g < 4 * ngroups; g++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 32 max = 96
        int part = 0;
        for (int k = 0; k < 2; k++) {
            int j = 2 * g + k;
            float v = src == 2 ? hb[j] : (src == 1 ? o[j & 63] : x[j & 63]);
            int q = roundq(v, e);
            mq_set(j, q);
            ap_int<15> sq = (ap_int<8>)q * (ap_int<8>)q;
#pragma HLS BIND_OP variable = sq op = mul impl = fabric
            part += (int)sq;
        }
        ss += part;
    }
    s2 = ss;
    return e;
}

// RMSNorm scalar 1/sqrt(mean(x^2) + 1e-5) from the quantized x (exact: x = q * 2^e)
static float rms_scale(int e) {
#pragma HLS INLINE off
    float ss = ldexp_b(i2f48(s2), 2 * e - 6); // sum(q^2) * 2^(2e) / 64
    ss = ss + 1e-5f;
    return 1.0f / sqrtf(ss);
}

// x = dequantized embedding row = classifier row `token` (ROM word 1920 + token/2)
static void load_embedding(int token) {
#pragma HLS INLINE off
    int w = 1920 + (token >> 1), lane = token & 1;
emb:
    for (int j = 0; j < FP_DIM; j++) {
#pragma HLS PIPELINE II = 1
        ap_uint<8> byte = rom(lane, w, j >> 1);
        ap_uint<4> code = (j & 1) ? byte(7, 4) : byte(3, 0);
        ap_uint<8> sc = rom(lane, w, 32 + (j >> 4));
        int kw = (int)wmul(code, 1) * (int)sc(3, 0);
        x[j] = ldexp_b(i2f48(kw), (int)sc(7, 4)) * CLS_BH;
    }
}

// o = x * final RMSNorm gain (the classifier weights are the unscaled embedding)
static void final_gain() {
#pragma HLS INLINE off
fg:
    for (int j = 0; j < FP_DIM; j++) {
#pragma HLS PIPELINE II = 1
        o[j] = x[j] * GF[j];
    }
}

// RoPE on q, k; quantize q (per head), k and v (per kv head); store k, v in the cache
static void rope_store(int l, int pos) {
#pragma HLS INLINE off
    float cs[8];
#pragma HLS ARRAY_PARTITION variable = cs complete
ld_cs:
    for (int i = 0; i < 8; i++) {
#pragma HLS PIPELINE II = 1
        cs[i] = ROPE[pos * 8 + i];
    }
    uint8_t krow[36], vrow[36];
#pragma HLS ARRAY_PARTITION variable = krow complete
#pragma HLS ARRAY_PARTITION variable = vrow complete
    float rv[16][8];
    int ge[16];
#pragma HLS ARRAY_PARTITION variable = rv complete dim = 2
#pragma HLS ARRAY_PARTITION variable = ge complete
grp:
    for (int g = 0; g < 16; g++) {
#pragma HLS PIPELINE II = 4
        float v[8];
#pragma HLS ARRAY_PARTITION variable = v complete
        for (int k = 0; k < 4; k++) {
            v[2 * k] = qkvbE[g][k];
            v[2 * k + 1] = qkvbO[g][k];
        }
        if (g < 12) { // q heads and k groups get RoPE; freq index = pair index within the head
            for (int f = 0; f < 4; f++) {
                float v0 = v[2 * f], v1 = v[2 * f + 1];
                float c = cs[2 * f], s = cs[2 * f + 1];
                v[2 * f] = v0 * c - v1 * s;
                v[2 * f + 1] = v0 * s + v1 * c;
            }
        }
        uint32_t a = 0;
        for (int k = 0; k < 8; k++) {
            uint32_t b = fabs_bits(v[k]);
            a = b > a ? b : a;
            rv[g][k] = v[k];
        }
        ge[g] = pow2exp(a);
    }
qnt:
    for (int i = 0; i < 64; i++) { // 2 values per cycle
#pragma HLS PIPELINE II = 1
        int g = i >> 2, e = ge[g];
        for (int kk = 0; kk < 2; kk++) {
            int k = 2 * (i & 3) + kk;
            int q = roundq(rv[g][k], e);
            if (g < 8) qi[8 * g + k] = (s8)q;
            else if (g < 12) krow[8 * (g - 8) + k] = (uint8_t)(int8_t)q;
            else vrow[8 * (g - 12) + k] = (uint8_t)(int8_t)q;
        }
    }
    for (int g = 0; g < 16; g++) {
#pragma HLS UNROLL
        if (g < 8) eq[g] = ge[g];
        else if (g < 12) krow[32 + g - 8] = (uint8_t)(int8_t)ge[g];
        else vrow[32 + g - 12] = (uint8_t)(int8_t)ge[g];
    }
    int row = (l & 3) * FP_SEQ + pos;
    for (int k = 0; k < 36; k++) {
#pragma HLS UNROLL
        if (l < 4) {
            kc[row][k] = krow[k];
            vc[row][k] = vrow[k];
        } else {
            kc4[pos][k] = krow[k];
            vc4[pos][k] = vrow[k];
        }
    }
    for (int g = 0; g < FP_NKV; g++) {
#pragma HLS UNROLL
        int ek = (int8_t)krow[32 + g], ev = (int8_t)vrow[32 + g];
        int evm = ev == FP_SENT ? 127 : ev;
        if (pos == 0) {
            ekmax[l][g] = ek;
            evmin[l][g] = evm;
        } else {
            ekmax[l][g] = ek > ekmax[l][g] ? ek : ekmax[l][g];
            evmin[l][g] = evm < evmin[l][g] ? evm : evmin[l][g];
        }
    }
}

// multi-head attention over positions 0..pos of layer l; result in o[]
static void attention(int l, int pos) {
#pragma HLS INLINE off
    int base = (l & 3) * FP_SEQ;
    bool top = l == 4;
    int ekm[FP_NKV], evm[FP_NKV];
#pragma HLS ARRAY_PARTITION variable = ekm complete
#pragma HLS ARRAY_PARTITION variable = evm complete
    for (int g = 0; g < FP_NKV; g++) {
#pragma HLS UNROLL
        ekm[g] = ekmax[l][g];
        evm[g] = evmin[l][g];
    }
    // Two passes of 4 heads (heads 4hg..4hg+3 read kv groups 2hg, 2hg+1).
headgrp:
    for (int hg = 0; hg < 2; hg++) {
        ap_int<19> mxg[4];
#pragma HLS ARRAY_PARTITION variable = mxg complete
        // pass A: integer scores rescaled to the layer's max key exponent; running max per head
    passA:
        for (int t = 0; t <= pos; t++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256 avg = 128
            uint8_t kr[36];
#pragma HLS ARRAY_PARTITION variable = kr complete
            for (int k = 0; k < 36; k++) kr[k] = top ? kc4[t][k] : kc[base + t][k];
            for (int hh = 0; hh < 4; hh++) {
                int h = 4 * hg + hh, g = h >> 1;
                ap_int<18> s = 0;
                for (int i = 0; i < FP_HS; i++) {
                    ap_int<16> prod = qi[8 * h + i] * (s8)(int8_t)kr[8 * g + i];
#pragma HLS BIND_OP variable = prod op = mul impl = dsp
                    s += prod;
                }
                int sh = ekm[g] - (int)(int8_t)kr[32 + g];
                if (sh > 24) sh = 24;
                ap_int<19> sv = s >> sh;
                sp[hg][t][hh] = sv;
                mxg[hh] = (t == 0 || sv > mxg[hh]) ? sv : mxg[hh];
            }
        }
        // pass C: softmax numerators from the 2^(i/256) table, integer weighted sum of V
        // per head: k = floor(d * A) with A = C0 * 2^(ekmax + eq + 8) = mA * 2^eA exactly
        ap_uint<24> mA[4];
        int eA[4], evg[4];
        static ap_int<41> acc[4][FP_HS];
        static ap_uint<25> S[4];
#pragma HLS ARRAY_PARTITION variable = mA complete
#pragma HLS ARRAY_PARTITION variable = eA complete
#pragma HLS ARRAY_PARTITION variable = evg complete
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0
#pragma HLS ARRAY_PARTITION variable = S complete
        for (int hh = 0; hh < 4; hh++) {
#pragma HLS UNROLL
            int h = 4 * hg + hh, g = h >> 1;
            uint32_t ab = f2u(ldexp_b(C0, ekm[g] + eq[h] + 8));
            int E = (int)((ab >> 23) & 255u);
            bool zero = E == 0 || eq[h] == FP_SENT || ekm[g] == FP_SENT;
            mA[hh] = zero ? 0u : ((ab & 0x7fffffu) | 0x800000u);
            eA[hh] = zero ? 0 : E - 150;
            evg[hh] = evm[g];
            S[hh] = 0;
            for (int i = 0; i < FP_HS; i++) acc[hh][i] = 0;
        }
    passC:
        for (int t = 0; t <= pos; t++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256 avg = 128
            uint8_t vr[36];
#pragma HLS ARRAY_PARTITION variable = vr complete
            for (int k = 0; k < 36; k++) vr[k] = top ? vc4[t][k] : vc[base + t][k];
            for (int hh = 0; hh < 4; hh++) {
                int h = 4 * hg + hh, g = h >> 1;
                ap_int<20> d = sp[hg][t][hh] - mxg[hh];        // <= 0, |d| < 2^19
                ap_int<45> prod = d * mA[hh];                  // exact, |prod| < 2^43
                int k;
                if (eA[hh] >= 0) k = prod == 0 ? 0 : -4097;
                else {
                    int r = -eA[hh] > 44 ? 44 : -eA[hh];
                    ap_int<45> kq = prod >> r;
                    k = kq < -4097 ? -4097 : (int)kq;
                }
                ap_uint<17> p = 0;
                ap_uint<24> pp = 0;
                if (k >= -4096) {
                    ap_uint<16> T = T16[k & 255];
                    int hi = k >> 8; // -16 .. 0
                    p = T >> (-hi);
                    int ev = (int8_t)vr[32 + g];
                    if (ev != FP_SENT) {
                        int sh = ev - evg[hh] + hi;
                        pp = sh >= 0 ? (ap_uint<24>)((ap_uint<24>)T << (sh > 8 ? 8 : sh))
                                     : (ap_uint<24>)(T >> (-sh > 16 ? 16 : -sh));
                    }
                }
                S[hh] += p;
                for (int i = 0; i < FP_HS; i++) acc[hh][i] += pp * (s8)(int8_t)vr[8 * g + i];
            }
        }
        float rS[4];
#pragma HLS ARRAY_PARTITION variable = rS complete
    rs:
        for (int hh = 0; hh < 4; hh++) {
#pragma HLS PIPELINE II = 1
            rS[hh] = 1.0f / i2f48(S[hh]);
        }
    outp:
        for (int j = 0; j < 32; j++) {
#pragma HLS PIPELINE II = 1
            int hh = j >> 3;
            o[32 * hg + j] = ldexp_b(i2f48(acc[hh][j & 7]), evg[hh]) * rS[hh];
        }
    }
}

int fp4_forward(int token, int pos) {
#pragma HLS INLINE off
    load_embedding(token);
layers:
    for (int l = 0; l < FP_NL; l++) {
        int cl = l * 768; // chunks per layer: qkv 128, wo 64, w13 384, w2 192
        int ex = quant(0, 8);
        float rs = rms_scale(ex);
        matmul(MM_QKV, cl, 128, ex, BH[l][0] * rs, BH[l][1] * rs, BH[l][2] * rs);
        rope_store(l, pos);
        attention(l, pos);
        int eo = quant(1, 8);
        matmul(MM_WO, cl + 128, 64, eo, BH[l][3], 0.0f, 0.0f);
        residual();
        int ex2 = quant(0, 8);
        float rs2 = rms_scale(ex2);
        matmul(MM_W13, cl + 192, 384, ex2, BH[l][4] * rs2, BH[l][5] * rs2, 0.0f);
        int eh = quant(2, 24);
        matmul(MM_W2, cl + 576, 192, eh, BH[l][6], 0.0f, 0.0f);
        residual();
    }
    final_gain();
    int exf = quant(1, 8); // argmax is invariant to the positive RMSNorm scale
    return matmul(MM_CLS, 3840, 512, exf, 0.0f, 0.0f, 0.0f);
}

// ------------------------------------------------------------------ stream mode helpers
static ap_uint<9> hist[64]; // last tokens of the current story (circular)
static bool seen[FP_VOCAB];

// Mode 1 emits ring entries: (seq & 0x7fff) << 16 | flag << 12 | token, seq = entry index.
// flag 1 marks the first token of a story opener (the host starts a new paragraph there).
static void emit(int tokens[FP_RING], int &produced, int tok, int flag) {
#pragma HLS INLINE
    tokens[produced & (FP_RING - 1)] = ((produced & 0x7fff) << 16) | (flag << 12) | tok;
    produced++;
}

void llama_fp4(const int prompt[FP_PARAMS], int n_prompt, int steps, int tokens[FP_RING],
               int *n_out, int mode, int ctx_len, int window, int max_new) {
#pragma HLS INTERFACE mode = s_axilite port = prompt bundle = control
#pragma HLS INTERFACE mode = s_axilite port = n_prompt bundle = control
#pragma HLS INTERFACE mode = s_axilite port = steps bundle = control
#pragma HLS INTERFACE mode = s_axilite port = tokens bundle = control
#pragma HLS INTERFACE mode = s_axilite port = n_out bundle = control
#pragma HLS INTERFACE mode = s_axilite port = mode bundle = control
#pragma HLS INTERFACE mode = s_axilite port = ctx_len bundle = control
#pragma HLS INTERFACE mode = s_axilite port = window bundle = control
#pragma HLS INTERFACE mode = s_axilite port = max_new bundle = control
#pragma HLS INTERFACE mode = s_axilite port = return bundle = control
#pragma HLS ALLOCATION function instances = fp4_forward limit = 1
#pragma HLS BIND_STORAGE variable = hist type = ram_2p impl = lutram
#pragma HLS BIND_STORAGE variable = seen type = ram_2p impl = lutram
#pragma HLS ARRAY_PARTITION variable = x cyclic factor = 8
#pragma HLS ARRAY_PARTITION variable = o cyclic factor = 8
#pragma HLS ARRAY_PARTITION variable = hb cyclic factor = 8
#pragma HLS ARRAY_PARTITION variable = qkvbE complete dim = 2
#pragma HLS ARRAY_PARTITION variable = qkvbO complete dim = 2
#pragma HLS ARRAY_PARTITION variable = mq0 complete
#pragma HLS ARRAY_PARTITION variable = mq1 complete
#pragma HLS ARRAY_PARTITION variable = mq2 complete
#pragma HLS ARRAY_PARTITION variable = m30 complete
#pragma HLS ARRAY_PARTITION variable = m31 complete
#pragma HLS ARRAY_PARTITION variable = m32 complete
#pragma HLS ARRAY_PARTITION variable = qi complete
#pragma HLS ARRAY_PARTITION variable = eq complete
#pragma HLS ARRAY_PARTITION variable = ekmax complete dim = 2
#pragma HLS ARRAY_PARTITION variable = evmin complete dim = 2
#pragma HLS ARRAY_RESHAPE variable = sp complete dim = 3
#pragma HLS ARRAY_RESHAPE variable = kc complete dim = 2
#pragma HLS ARRAY_RESHAPE variable = vc complete dim = 2
#pragma HLS ARRAY_RESHAPE variable = kc4 complete dim = 2
#pragma HLS ARRAY_RESHAPE variable = vc4 complete dim = 2
#pragma HLS ARRAY_RESHAPE variable = CROMA_M complete dim = 2
#pragma HLS ARRAY_RESHAPE variable = CROMB_M complete dim = 2
#pragma HLS ARRAY_RESHAPE variable = CROMT complete dim = 2
#pragma HLS BIND_STORAGE variable = CROMA_M type = rom_1p impl = bram
#pragma HLS BIND_STORAGE variable = CROMB_M type = rom_1p impl = bram
#pragma HLS BIND_STORAGE variable = CROMT type = rom_1p impl = bram
#pragma HLS ARRAY_PARTITION variable = T16 complete
    if (mode == 0) {
        // ---- one generate() run: tokens[pos] = argmax at pos (llama2.c run.c loop)
        if (n_prompt < 1 || n_prompt > FP_MAXPROMPT || steps < 1 || steps > FP_SEQ) {
            *n_out = -1;
            return;
        }
        for (int i = 0; i < n_prompt; i++) {
            if (prompt[i] < 0 || prompt[i] >= FP_VOCAB) {
                *n_out = -1;
                return;
            }
        }
        int token = prompt[0], n = 0;
    gen:
        for (int pos = 0; pos < steps; pos++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256 avg = 256
            int a = fp4_forward(token, pos);
            tokens[pos] = a;
            n = pos + 1;
            int next = pos < n_prompt - 1 ? prompt[pos + 1] : a;
            if (next == 1) break;
            token = next;
        }
        *n_out = n;
        return;
    }

    // ---- mode 1: endless stream, the same algorithm as host/fp4_host.py `stream`:
    // each run is BOS + the last (ctx_len-1) story tokens, `window` positions; new tokens go to
    // the ring; a story ends at BOS, when the last 64 tokens hold <= 12 distinct ones, or after
    // 800 positions, and the next story starts from the next opener. Stops once >= max_new new
    // tokens were emitted (checked between runs). prompt[] holds the openers:
    // [count, len0, tokens0..., len1, tokens1..., ...] (tokens without BOS, all < FP_VOCAB).
    int nop = prompt[0];
    if (ctx_len < 1 || ctx_len > FP_MAXPROMPT || window <= ctx_len || window > FP_SEQ ||
        max_new < 1 || nop < 1 || nop > 32) {
        *n_out = -1;
        return;
    }
    int produced = 0, emitted_new = 0, story_len = 0, k = 0;
    int head = 0, hl = 0, hl_total = 0;
    // load opener k into the history and emit it (first token flagged)
    {
        int off = 1;
        int len = prompt[off];
        head = 0;
        hl = 0;
        hl_total = 0;
    op0:
        for (int j = 0; j < len && j < 63; j++) {
            int t = prompt[off + 1 + j];
            hist[head] = t;
            head = (head + 1) & 63;
            hl = hl < 64 ? hl + 1 : 64;
            hl_total = hl_total < 64 ? hl_total + 1 : 64;
            emit(tokens, produced, t, j == 0 ? 1 : 0);
        }
    }
stream:
    while (emitted_new < max_new) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1000
        int m = hl < ctx_len - 1 ? hl : ctx_len - 1;
        int np = 1 + m;
        int base = head - m; // prompt = BOS, hist[base .. base+m) (hist only grows after it)
        int token = 1, ncomp = 0;
        bool ended = false;
    sgen:
        for (int pos = 0; pos < window; pos++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256 avg = 256
            int a = fp4_forward(token, pos);
            ncomp = pos + 1;
            int next;
            if (pos < np - 1) {
                next = (int)hist[(base + pos) & 63];
            } else {
                next = a;
                if (a == 1) {
                    ended = true;
                } else {
                    emit(tokens, produced, a, 0);
                    hist[head] = a;
                    head = (head + 1) & 63;
                    hl = hl < 64 ? hl + 1 : 64;
                    hl_total = hl_total < 64 ? hl_total + 1 : 64;
                    emitted_new++;
                }
            }
            if (next == 1) break;
            token = next;
        }
        story_len += ncomp - (np - 1);
        if (!ended && hl_total == 64) { // distinct tokens among the last 64
            int distinct = 0;
        dc:
            for (int j = 0; j < 64; j++) {
                int t = (int)hist[j];
                if (!seen[t]) {
                    seen[t] = true;
                    distinct++;
                }
            }
        dclr:
            for (int j = 0; j < 64; j++) seen[(int)hist[j]] = false;
            if (distinct <= 12) ended = true;
        }
        if (!ended && story_len > 800) ended = true;
        if (ended) {
            story_len = 0;
            k = k + 1 == nop ? 0 : k + 1;
            int off = 1;
        fo:
            for (int j = 0; j < k; j++) off += 1 + prompt[off];
            int len = prompt[off];
            head = 0;
            hl = 0;
            hl_total = 0;
        op:
            for (int j = 0; j < len && j < 63; j++) {
                int t = prompt[off + 1 + j];
                hist[head] = t;
                head = (head + 1) & 63;
                hl = hl < 64 ? hl + 1 : 64;
                hl_total = hl_total < 64 ? hl_total + 1 : 64;
                emit(tokens, produced, t, j == 0 ? 1 : 0);
            }
        }
    }
    *n_out = produced;
}
