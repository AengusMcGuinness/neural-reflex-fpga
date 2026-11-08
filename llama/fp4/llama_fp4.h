// FP4 llama2 (stories260K) accelerator: whole model on chip, generate loop on chip.
// This file + llama_fp4.cpp are both the hardware source (Vitis HLS) and the bit-exact
// golden model (any C++ compiler with -ffp-contract=off): every operation is an exact
// integer operation or a single IEEE-754 float op (add, mul, div, sqrt), never fused.
#pragma once
#include <stdint.h>

#define FP_DIM 64
#define FP_HIDP 192      // hidden 172 zero-padded to 3 chunks of 64
#define FP_NL 5
#define FP_NH 8
#define FP_NKV 4
#define FP_HS 8
#define FP_KVD 32
#define FP_VOCAB 512
#define FP_SEQ 256       // positions supported by the on-chip KV cache
#define FP_MAXPROMPT 64
#define FP_PARAMS 256    // prompt[] size: mode 0 prompt, or mode 1 openers table
#define FP_RING 2048     // tokens[] size: mode 0 per-position argmax, or mode 1 ring buffer
#define FP_SENT (-128)   // exponent of an all-zero vector

// mode 0: run the llama2.c generate() loop on chip (greedy). tokens[pos] = argmax at pos.
//         Stops after `steps` positions or when the next token is BOS (1).
//         *n_out = positions computed, or -1 if the arguments are invalid.
// mode 1: endless stream (see llama_fp4.cpp): tokens[] is a ring of tagged entries,
//         prompt[] the openers table; *n_out = entries written.
void llama_fp4(const int prompt[FP_PARAMS], int n_prompt, int steps, int tokens[FP_RING],
               int *n_out, int mode, int ctx_len, int window, int max_new);

// One forward pass (exposed for the testbench's teacher-forced quality check).
int fp4_forward(int token, int pos);
