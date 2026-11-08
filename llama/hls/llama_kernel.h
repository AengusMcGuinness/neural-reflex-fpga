// llama2.c forward pass as a Vitis HLS kernel (fp32, greedy decoding on chip).
// Mirrors karpathy/llama2.c run.c forward() operation-for-operation, including the
// summation order of every dot product, so greedy tokens match the CPU reference.
#pragma once

// Compile-time capacity. stories260K (dim 64, hidden 172, seq 512) and
// stories15M (dim 288, hidden 768, seq 256) both fit. Larger configs return -1.
#define DIM_MAX 288
#define HID_MAX 768
#define SEQ_MAX 512
#define HS_MAX 64
#define ROWS 16 // matmul rows processed together; must be >= fadd recurrence latency

#define LLAMA_ERR_CONFIG -1

// All tensors live in one memory `mem` (DDR3 on the board); offsets are in floats.
//   w_off      : start of the float weights = checkpoint file start + 28 bytes (7-int header)
//   kv_off     : key cache (n_layers*seq_len*kv_dim floats), value cache right after it
//   rope_off   : [seq_len][head_size/2][2] = {cosf, sinf}(pos*freq), run.c's expression
//   logits_off : vocab_size floats, written only when write_logits != 0
// head_size = dim/n_heads, kv_dim = dim*n_kv_heads/n_heads, kv_mul = n_heads/n_kv_heads are
// passed in (checked by multiplication) so the hardware needs no integer dividers.
// Returns run.c sample_argmax(logits) (first index of the maximum), or a negative error.
int llama_forward(float *mem, int w_off, int kv_off, int rope_off, int logits_off,
                  int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads,
                  int vocab_size, int seq_len, int shared_weights,
                  int head_size, int kv_dim, int kv_mul,
                  int token, int pos, int write_logits);
