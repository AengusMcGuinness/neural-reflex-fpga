// llama2.c forward pass for Vitis HLS. See llama_kernel.h.
//
// Numerics (why this matches run.c compiled with `gcc -O3`, no -march, no fast-math):
//  * every dot product accumulates j = 0..n-1 sequentially from 0.0f, with a separately
//    rounded multiply and add (x86-64 baseline has no FMA, so gcc emits mul+add too);
//  * matmul computes ROWS output rows interleaved, so each row keeps its own sequential
//    order while the adder pipeline stays full;
//  * RoPE cos/sin come from a host table built with run.c's exact expression;
//  * exp/sqrt use hls:: so C simulation is bit-accurate to the generated RTL.
// Area: matmul and rmsnorm take only scalar arguments and work on fixed global buffers
// (mm_x in, mm_y out), so HLS builds exactly one of each; one m_axi port; no integer
// dividers; only the loops that dominate run time are pipelined (-pipeline_loops 0).
#include "llama_kernel.h"
#include <hls_math.h>

static float wbuf[ROWS * HID_MAX]; // matmul weight rows
static float mm_x[HID_MAX];        // matmul input  (also rmsnorm output, attention output)
static float mm_y[HID_MAX];        // matmul output
static float st_x[DIM_MAX];        // residual stream x

// mm_x[0..size) = rmsnorm(st_x, weight @ mem[woff])   [run.c rmsnorm]
static void rmsnorm(const float *mem, int woff, int size) {
#pragma HLS INLINE off
    float ss = 0.0f;
rms_ss:
    for (int j = 0; j < size; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 64 max = 288
        ss += st_x[j] * st_x[j];
    }
    ss /= (float)size;
    ss += 1e-5f;
    ss = 1.0f / hls::sqrt(ss);
rms_scale:
    for (int j = 0; j < size; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 64 max = 288
        mm_x[j] = mem[woff + j] * (ss * st_x[j]);
    }
}

// mm_y = W (d,n) @ mm_x (n,)   [run.c matmul]
// mode 1 = classifier: returns run.c sample_argmax over the d results instead of writing
// mm_y and, if write_logits, streams them to mem[logits_off..].
static int matmul(float *mem, int woff, int n, int d, int mode, int logits_off, int write_logits) {
#pragma HLS INLINE off
    float acc[ROWS];
#pragma HLS BIND_STORAGE variable = acc type = ram_2p impl = lutram
    int max_i = 0;
    float max_p = 0.0f;
    int blk_off = woff;
    int nrows = n * ROWS;
mm_blocks:
    for (int i0 = 0; i0 < d; i0 += ROWS) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 2000
        int rows = (d - i0) < ROWS ? (d - i0) : ROWS;
        int cnt = rows * n;
    mm_load:
        for (int k = 0; k < cnt; k++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1024 max = 12288
            wbuf[k] = mem[blk_off + k];
        }
    mm_init:
        for (int r = 0; r < ROWS; r++) {
#pragma HLS PIPELINE II = 1
            acc[r] = 0.0f;
        }
        // flattened (j, r) loop: idx = r*n + j; acc[r] is revisited exactly ROWS iterations later
        int r = 0, j = 0, idx = 0;
    mm_dot:
        for (int it = 0; it < nrows; it++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1024 max = 12288
#pragma HLS DEPENDENCE variable = acc inter RAW distance = 16 true
            float wv = wbuf[idx];
            float xv = mm_x[j];
            if (r < rows) acc[r] += wv * xv;
            if (r == ROWS - 1) {
                r = 0;
                j++;
                idx = j;
            } else {
                r++;
                idx += n;
            }
        }
    mm_out:
        for (int rr = 0; rr < ROWS; rr++) {
#pragma HLS PIPELINE II = 1
            if (rr < rows) {
                int i = i0 + rr;
                float p = acc[rr];
                if (mode == 0) {
                    mm_y[i] = p;
                } else {
                    if (write_logits) mem[logits_off + i] = p;
                    if (i == 0) {
                        max_p = p;
                    } else if (p > max_p) {
                        max_i = i;
                        max_p = p;
                    }
                }
            }
        }
        blk_off += nrows;
    }
    return max_i;
}

int llama_forward(float *mem, int w_off, int kv_off, int rope_off, int logits_off,
                  int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads,
                  int vocab_size, int seq_len, int shared_weights,
                  int head_size, int kv_dim, int kv_mul,
                  int token, int pos, int write_logits) {
    // depth= only sizes the co-simulation buffer (stories260K layout from the testbench).
#pragma HLS INTERFACE mode = m_axi port = mem offset = slave bundle = gmem depth = 432640 max_read_burst_length = 64 max_write_burst_length = 16 num_read_outstanding = 4 num_write_outstanding = 2 latency = 32
#pragma HLS INTERFACE mode = s_axilite port = mem bundle = control
#pragma HLS INTERFACE mode = s_axilite port = w_off bundle = control
#pragma HLS INTERFACE mode = s_axilite port = kv_off bundle = control
#pragma HLS INTERFACE mode = s_axilite port = rope_off bundle = control
#pragma HLS INTERFACE mode = s_axilite port = logits_off bundle = control
#pragma HLS INTERFACE mode = s_axilite port = dim bundle = control
#pragma HLS INTERFACE mode = s_axilite port = hidden_dim bundle = control
#pragma HLS INTERFACE mode = s_axilite port = n_layers bundle = control
#pragma HLS INTERFACE mode = s_axilite port = n_heads bundle = control
#pragma HLS INTERFACE mode = s_axilite port = n_kv_heads bundle = control
#pragma HLS INTERFACE mode = s_axilite port = vocab_size bundle = control
#pragma HLS INTERFACE mode = s_axilite port = seq_len bundle = control
#pragma HLS INTERFACE mode = s_axilite port = shared_weights bundle = control
#pragma HLS INTERFACE mode = s_axilite port = head_size bundle = control
#pragma HLS INTERFACE mode = s_axilite port = kv_dim bundle = control
#pragma HLS INTERFACE mode = s_axilite port = kv_mul bundle = control
#pragma HLS INTERFACE mode = s_axilite port = token bundle = control
#pragma HLS INTERFACE mode = s_axilite port = pos bundle = control
#pragma HLS INTERFACE mode = s_axilite port = write_logits bundle = control
#pragma HLS INTERFACE mode = s_axilite port = return bundle = control
#pragma HLS ALLOCATION function instances = matmul limit = 1
#pragma HLS ALLOCATION function instances = rmsnorm limit = 1
#pragma HLS ALLOCATION operation instances = fexp limit = 1
#pragma HLS ALLOCATION operation instances = fdiv limit = 1
#pragma HLS ALLOCATION operation instances = fsqrt limit = 1
#pragma HLS ALLOCATION operation instances = mul limit = 2

    // ---- reject anything run.c would not compute identically or the buffers can't hold ----
    if (dim <= 0 || dim > DIM_MAX || hidden_dim <= 0 || hidden_dim > HID_MAX || n_layers <= 0 ||
        n_heads <= 0 || n_kv_heads <= 0 || seq_len <= 0 || seq_len > SEQ_MAX || vocab_size <= 0 ||
        head_size <= 0 || head_size > HS_MAX || (head_size & 1) || head_size * n_heads != dim ||
        kv_mul <= 0 || kv_mul * n_kv_heads != n_heads || kv_dim != n_kv_heads * head_size ||
        token < 0 || token >= vocab_size || pos < 0 || pos >= seq_len)
        return LLAMA_ERR_CONFIG;

    // ---- weight offsets in floats, exactly as run.c memory_map_weights ----
    int dd = dim * dim, dkv = dim * kv_dim, dh = dim * hidden_dim;
    int o_rms_att = w_off + vocab_size * dim;
    int o_wq = o_rms_att + n_layers * dim;
    int o_wk = o_wq + n_layers * dd;
    int o_wv = o_wk + n_layers * dkv;
    int o_wo = o_wv + n_layers * dkv;
    int o_rms_ffn = o_wo + n_layers * dd;
    int o_w1 = o_rms_ffn + n_layers * dim;
    int o_w2 = o_w1 + n_layers * dh;
    int o_w3 = o_w2 + n_layers * dh;
    int o_rms_final = o_w3 + n_layers * dh;
    int skip = (seq_len * head_size) >> 1; // freq_cis_real, then freq_cis_imag
    int o_wcls = shared_weights ? w_off : o_rms_final + dim + skip + skip;
    int layer_kv = seq_len * kv_dim;
    int o_vcache = kv_off + n_layers * layer_kv;
    int pos_kv = pos * kv_dim;

    float q[DIM_MAX], k[DIM_MAX], v[DIM_MAX];
    float hb[HID_MAX];
    float att[SEQ_MAX];
    float kvrow[HS_MAX];
    float cs[HS_MAX]; // interleaved cos,sin for this pos, indexed by head_dim

    int emb = w_off + token * dim;
load_x:
    for (int i = 0; i < dim; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 64 max = 288
        st_x[i] = mem[emb + i];
    }
    int rrow = rope_off + pos * head_size;
load_cs:
    for (int i = 0; i < head_size; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 64
        cs[i] = mem[rrow + i];
    }
    float sqrt_hs = hls::sqrt((float)head_size);

    int loff = 0; // l * seq_len * kv_dim
layers:
    for (int l = 0; l < n_layers; l++) {
#pragma HLS LOOP_TRIPCOUNT min = 5 max = 6
        rmsnorm(mem, o_rms_att + l * dim, dim); // -> mm_x (= run.c s->xb)
        matmul(mem, o_wq + l * dd, dim, dim, 0, 0, 0);
    cp_q:
        for (int i = 0; i < dim; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 64 max = 288
            q[i] = mm_y[i];
        }
        matmul(mem, o_wk + l * dkv, dim, kv_dim, 0, 0, 0);
    cp_k:
        for (int i = 0; i < kv_dim; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 32 max = 288
            k[i] = mm_y[i];
        }
        matmul(mem, o_wv + l * dkv, dim, kv_dim, 0, 0, 0);
    cp_v:
        for (int i = 0; i < kv_dim; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 32 max = 288
            v[i] = mm_y[i];
        }

        // RoPE: rotate q (all of dim) and k (first kv_dim); head_dim = i % head_size
        int head_dim = 0;
    rope_loop:
        for (int i = 0; i < dim; i += 2) {
#pragma HLS LOOP_TRIPCOUNT min = 32 max = 144
            float fcr = cs[head_dim];
            float fci = cs[head_dim + 1];
            float q0 = q[i], q1 = q[i + 1];
            q[i] = q0 * fcr - q1 * fci;
            q[i + 1] = q0 * fci + q1 * fcr;
            if (i < kv_dim) {
                float k0 = k[i], k1 = k[i + 1];
                k[i] = k0 * fcr - k1 * fci;
                k[i + 1] = k0 * fci + k1 * fcr;
            }
            head_dim += 2;
            if (head_dim == head_size) head_dim = 0;
        }

        // store this position's k,v into the caches
        int kc = kv_off + loff + pos_kv, vc = o_vcache + loff + pos_kv;
    kv_store:
        for (int i = 0; i < kv_dim; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 32 max = 288
            mem[kc + i] = k[i];
        }
    kv_store_v:
        for (int i = 0; i < kv_dim; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 32 max = 288
            mem[vc + i] = v[i];
        }

        // multihead attention; the per-head outputs go to mm_x (= run.c s->xb)
        int kvh = 0, kvh_cnt = 0, qbase = 0; // kvh = (h / kv_mul) * head_size
    heads:
        for (int h = 0; h < n_heads; h++) {
#pragma HLS LOOP_TRIPCOUNT min = 6 max = 8
            int krow = kv_off + loff + kvh;
        scores:
            for (int t = 0; t <= pos; t++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
                // the current position is taken from on-chip k (no read-after-write via DDR)
                if (t == pos) {
                cur_k:
                    for (int i = 0; i < head_size; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 64
                        kvrow[i] = k[kvh + i];
                    }
                } else {
                ld_k:
                    for (int i = 0; i < head_size; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 64
                        kvrow[i] = mem[krow + i];
                    }
                }
                float score = 0.0f;
            score_dot:
                for (int i = 0; i < head_size; i++) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 64
                    score += q[qbase + i] * kvrow[i];
                }
                score /= sqrt_hs;
                att[t] = score;
                krow += kv_dim;
            }

            // softmax(att, pos + 1)
            float max_val = att[0];
        sm_max:
            for (int i = 1; i <= pos; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
                if (att[i] > max_val) max_val = att[i];
            }
            float sum = 0.0f;
        sm_exp:
            for (int i = 0; i <= pos; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
                att[i] = hls::exp(att[i] - max_val);
                sum += att[i];
            }
        sm_div:
            for (int i = 0; i <= pos; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
                att[i] /= sum;
            }

        zero_xb:
            for (int i = 0; i < head_size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 64
                mm_x[qbase + i] = 0.0f;
            }
            int vrow = o_vcache + loff + kvh;
        weighted:
            for (int t = 0; t <= pos; t++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
                if (t == pos) {
                cur_v:
                    for (int i = 0; i < head_size; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 64
                        kvrow[i] = v[kvh + i];
                    }
                } else {
                ld_v:
                    for (int i = 0; i < head_size; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 64
                        kvrow[i] = mem[vrow + i];
                    }
                }
                float a = att[t];
            acc_v:
                for (int i = 0; i < head_size; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 64
                    // each iteration owns a distinct element mm_x[qbase + i]
#pragma HLS DEPENDENCE variable = mm_x inter false
                    mm_x[qbase + i] += a * kvrow[i];
                }
                vrow += kv_dim;
            }

            qbase += head_size;
            if (++kvh_cnt == kv_mul) {
                kvh_cnt = 0;
                kvh += head_size;
            }
        }

        matmul(mem, o_wo + l * dd, dim, dim, 0, 0, 0); // mm_y = s->xb2
    resid1:
        for (int i = 0; i < dim; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 64 max = 288
            st_x[i] += mm_y[i];
        }

        rmsnorm(mem, o_rms_ffn + l * dim, dim); // -> mm_x
        matmul(mem, o_w1 + l * dh, dim, hidden_dim, 0, 0, 0);
    cp_hb:
        for (int i = 0; i < hidden_dim; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 172 max = 768
            hb[i] = mm_y[i];
        }
        matmul(mem, o_w3 + l * dh, dim, hidden_dim, 0, 0, 0); // mm_y = s->hb2
    swiglu:
        for (int i = 0; i < hidden_dim; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 172 max = 768
            float val = hb[i];
            val *= (1.0f / (1.0f + hls::exp(-val)));
            val *= mm_y[i];
            mm_x[i] = val; // = s->hb, the input of the w2 matmul
        }
        matmul(mem, o_w2 + l * dh, hidden_dim, dim, 0, 0, 0); // mm_y = s->xb
    resid2:
        for (int i = 0; i < dim; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 64 max = 288
            st_x[i] += mm_y[i];
        }
        loff += layer_kv;
    }

    rmsnorm(mem, o_rms_final, dim); // run.c does this in place on x; same values, into mm_x
    return matmul(mem, o_wcls, dim, vocab_size, 1, logits_off, write_logits);
}
