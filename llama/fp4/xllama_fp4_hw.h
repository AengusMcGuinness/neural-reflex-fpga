// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
// control
// 0x000 : Control signals
//         bit 0  - ap_start (Read/Write/COH)
//         bit 1  - ap_done (Read/COR)
//         bit 2  - ap_idle (Read)
//         bit 3  - ap_ready (Read/COR)
//         bit 7  - auto_restart (Read/Write)
//         bit 9  - interrupt (Read)
//         others - reserved
// 0x004 : Global Interrupt Enable Register
//         bit 0  - Global Interrupt Enable (Read/Write)
//         others - reserved
// 0x008 : IP Interrupt Enable Register (Read/Write)
//         bit 0 - enable ap_done interrupt (Read/Write)
//         bit 1 - enable ap_ready interrupt (Read/Write)
//         others - reserved
// 0x00c : IP Interrupt Status Register (Read/TOW)
//         bit 0 - ap_done (Read/TOW)
//         bit 1 - ap_ready (Read/TOW)
//         others - reserved
// 0x010 : Data signal of n_prompt
//         bit 31~0 - n_prompt[31:0] (Read/Write)
// 0x014 : reserved
// 0x018 : Data signal of steps
//         bit 31~0 - steps[31:0] (Read/Write)
// 0x01c : reserved
// 0x020 : Data signal of n_out
//         bit 31~0 - n_out[31:0] (Read)
// 0x024 : Control signal of n_out
//         bit 0  - n_out_ap_vld (Read/COR)
//         others - reserved
// 0x100 ~
// 0x1ff : Memory 'prompt' (64 * 32b)
//         Word n : bit [31:0] - prompt[n]
// 0x400 ~
// 0x7ff : Memory 'tokens' (256 * 32b)
//         Word n : bit [31:0] - tokens[n]
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XLLAMA_FP4_CONTROL_ADDR_AP_CTRL       0x000
#define XLLAMA_FP4_CONTROL_ADDR_GIE           0x004
#define XLLAMA_FP4_CONTROL_ADDR_IER           0x008
#define XLLAMA_FP4_CONTROL_ADDR_ISR           0x00c
#define XLLAMA_FP4_CONTROL_ADDR_N_PROMPT_DATA 0x010
#define XLLAMA_FP4_CONTROL_BITS_N_PROMPT_DATA 32
#define XLLAMA_FP4_CONTROL_ADDR_STEPS_DATA    0x018
#define XLLAMA_FP4_CONTROL_BITS_STEPS_DATA    32
#define XLLAMA_FP4_CONTROL_ADDR_N_OUT_DATA    0x020
#define XLLAMA_FP4_CONTROL_BITS_N_OUT_DATA    32
#define XLLAMA_FP4_CONTROL_ADDR_N_OUT_CTRL    0x024
#define XLLAMA_FP4_CONTROL_ADDR_PROMPT_BASE   0x100
#define XLLAMA_FP4_CONTROL_ADDR_PROMPT_HIGH   0x1ff
#define XLLAMA_FP4_CONTROL_WIDTH_PROMPT       32
#define XLLAMA_FP4_CONTROL_DEPTH_PROMPT       64
#define XLLAMA_FP4_CONTROL_ADDR_TOKENS_BASE   0x400
#define XLLAMA_FP4_CONTROL_ADDR_TOKENS_HIGH   0x7ff
#define XLLAMA_FP4_CONTROL_WIDTH_TOKENS       32
#define XLLAMA_FP4_CONTROL_DEPTH_TOKENS       256

