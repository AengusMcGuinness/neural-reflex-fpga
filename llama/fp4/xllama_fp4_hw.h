// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
// control
// 0x0000 : Control signals
//          bit 0  - ap_start (Read/Write/COH)
//          bit 1  - ap_done (Read/COR)
//          bit 2  - ap_idle (Read)
//          bit 3  - ap_ready (Read/COR)
//          bit 7  - auto_restart (Read/Write)
//          bit 9  - interrupt (Read)
//          others - reserved
// 0x0004 : Global Interrupt Enable Register
//          bit 0  - Global Interrupt Enable (Read/Write)
//          others - reserved
// 0x0008 : IP Interrupt Enable Register (Read/Write)
//          bit 0 - enable ap_done interrupt (Read/Write)
//          bit 1 - enable ap_ready interrupt (Read/Write)
//          others - reserved
// 0x000c : IP Interrupt Status Register (Read/TOW)
//          bit 0 - ap_done (Read/TOW)
//          bit 1 - ap_ready (Read/TOW)
//          others - reserved
// 0x0010 : Data signal of n_prompt
//          bit 31~0 - n_prompt[31:0] (Read/Write)
// 0x0014 : reserved
// 0x0018 : Data signal of steps
//          bit 31~0 - steps[31:0] (Read/Write)
// 0x001c : reserved
// 0x0020 : Data signal of n_out
//          bit 31~0 - n_out[31:0] (Read)
// 0x0024 : Control signal of n_out
//          bit 0  - n_out_ap_vld (Read/COR)
//          others - reserved
// 0x0030 : Data signal of mode
//          bit 31~0 - mode[31:0] (Read/Write)
// 0x0034 : reserved
// 0x0038 : Data signal of ctx_len
//          bit 31~0 - ctx_len[31:0] (Read/Write)
// 0x003c : reserved
// 0x0040 : Data signal of window
//          bit 31~0 - window[31:0] (Read/Write)
// 0x0044 : reserved
// 0x0048 : Data signal of max_new
//          bit 31~0 - max_new[31:0] (Read/Write)
// 0x004c : reserved
// 0x0400 ~
// 0x07ff : Memory 'prompt' (256 * 32b)
//          Word n : bit [31:0] - prompt[n]
// 0x2000 ~
// 0x3fff : Memory 'tokens' (2048 * 32b)
//          Word n : bit [31:0] - tokens[n]
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XLLAMA_FP4_CONTROL_ADDR_AP_CTRL       0x0000
#define XLLAMA_FP4_CONTROL_ADDR_GIE           0x0004
#define XLLAMA_FP4_CONTROL_ADDR_IER           0x0008
#define XLLAMA_FP4_CONTROL_ADDR_ISR           0x000c
#define XLLAMA_FP4_CONTROL_ADDR_N_PROMPT_DATA 0x0010
#define XLLAMA_FP4_CONTROL_BITS_N_PROMPT_DATA 32
#define XLLAMA_FP4_CONTROL_ADDR_STEPS_DATA    0x0018
#define XLLAMA_FP4_CONTROL_BITS_STEPS_DATA    32
#define XLLAMA_FP4_CONTROL_ADDR_N_OUT_DATA    0x0020
#define XLLAMA_FP4_CONTROL_BITS_N_OUT_DATA    32
#define XLLAMA_FP4_CONTROL_ADDR_N_OUT_CTRL    0x0024
#define XLLAMA_FP4_CONTROL_ADDR_MODE_DATA     0x0030
#define XLLAMA_FP4_CONTROL_BITS_MODE_DATA     32
#define XLLAMA_FP4_CONTROL_ADDR_CTX_LEN_DATA  0x0038
#define XLLAMA_FP4_CONTROL_BITS_CTX_LEN_DATA  32
#define XLLAMA_FP4_CONTROL_ADDR_WINDOW_DATA   0x0040
#define XLLAMA_FP4_CONTROL_BITS_WINDOW_DATA   32
#define XLLAMA_FP4_CONTROL_ADDR_MAX_NEW_DATA  0x0048
#define XLLAMA_FP4_CONTROL_BITS_MAX_NEW_DATA  32
#define XLLAMA_FP4_CONTROL_ADDR_PROMPT_BASE   0x0400
#define XLLAMA_FP4_CONTROL_ADDR_PROMPT_HIGH   0x07ff
#define XLLAMA_FP4_CONTROL_WIDTH_PROMPT       32
#define XLLAMA_FP4_CONTROL_DEPTH_PROMPT       256
#define XLLAMA_FP4_CONTROL_ADDR_TOKENS_BASE   0x2000
#define XLLAMA_FP4_CONTROL_ADDR_TOKENS_HIGH   0x3fff
#define XLLAMA_FP4_CONTROL_WIDTH_TOKENS       32
#define XLLAMA_FP4_CONTROL_DEPTH_TOKENS       2048

