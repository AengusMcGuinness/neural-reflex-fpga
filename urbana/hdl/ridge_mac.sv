`timescale 1ns / 1ps
`default_nettype none

// Fixed-point ridge decoder: for each output j
//   acc = b[j] + sum_i f[i] * w[j*N_FEAT + i]          (48-bit signed)
//   y[j] = clamp((acc + round_bias) >>> shift[j], lo[j], hi[j])
// Bit-exact with python/fixed_point.py predict_int(). One MAC per cycle,
// about N_OUT * (N_FEAT + 6) cycles per frame (~3 us at 100 MHz for 40x6).
// Weights, bias and cfg start from the .mem files and can be rewritten at runtime.
module ridge_mac #(
    parameter int N_FEAT = 40,
    parameter int N_OUT = 6,
    parameter WEIGHTS_FILE = "weights.mem",
    parameter BIAS_FILE = "bias.mem",
    parameter CFG_FILE = "cfg.mem"
)(
    input wire clk,
    input wire rst,

    // write ports (from the UART frame receiver)
    input wire                                 feat_we,
    input wire [$clog2(N_FEAT)-1:0]            feat_addr,
    input wire signed [15:0]                   feat_data,
    input wire                                 w_we,
    input wire [$clog2(N_FEAT*N_OUT)-1:0]      w_addr,
    input wire signed [15:0]                   w_data,
    input wire                                 b_we,
    input wire [$clog2(N_OUT)-1:0]             b_addr,
    input wire signed [47:0]                   b_data,
    input wire                                 cfg_we,
    input wire [$clog2(3*N_OUT)-1:0]           cfg_addr,  // 3*j + {0: shift, 1: lo, 2: hi}
    input wire signed [15:0]                   cfg_data,

    input wire start,                                     // compute all outputs from current features
    output logic                               y_valid,   // one pulse per output, in order
    output logic [$clog2(N_OUT)-1:0]           y_idx,
    output logic signed [15:0]                 y_data,
    output logic                               done,      // pulse after the last output
    output logic                               busy,
    output logic signed [15:0]                 lo0,       // output 0 clamp range, for the LED bar
    output logic signed [15:0]                 hi0
);
    localparam int N_W = N_FEAT * N_OUT;
    localparam int FW = $clog2(N_FEAT);
    localparam int OW = $clog2(N_OUT);
    localparam int WW = $clog2(N_W);

    // ---------------- storage ----------------
    logic signed [15:0] feat [N_FEAT];
    logic signed [15:0] wmem [N_W];       // inferred as BRAM (simple dual port)
    logic signed [47:0] bias [N_OUT];
    logic signed [15:0] cfg  [3*N_OUT];

    initial begin
        $readmemh(WEIGHTS_FILE, wmem);
        $readmemh(BIAS_FILE, bias);
        $readmemh(CFG_FILE, cfg);
        for (int k = 0; k < N_FEAT; k++) feat[k] = 0;
    end

    always_ff @(posedge clk) begin
        if (feat_we) feat[feat_addr] <= feat_data;
        if (w_we)    wmem[w_addr]    <= w_data;
        if (b_we)    bias[b_addr]    <= b_data;
        if (cfg_we)  cfg[cfg_addr]   <= cfg_data;
    end

    assign lo0 = cfg[1];
    assign hi0 = cfg[2];

    // ---------------- control ----------------
    typedef enum logic [2:0] {IDLE, ISSUE, DRAIN, ROUND, SHIFT, CLAMP} state_t;
    state_t state;

    logic [OW-1:0] j;          // current output
    logic [FW-1:0] i;          // feature being issued
    logic [WW-1:0] base;       // j * N_FEAT

    // stage 1: weight RAM read + feature select
    logic signed [15:0] w_rd, f_s1;
    logic v1, first1, last1;
    // stage 2: product
    logic signed [31:0] prod;
    logic v2, first2, last2;
    // stage 3: accumulate
    logic signed [47:0] acc;
    logic acc_done;

    logic issuing;
    assign issuing = (state == ISSUE);

    always_ff @(posedge clk) begin
        w_rd   <= wmem[base + i];
        f_s1   <= feat[i];
        v1     <= issuing;
        first1 <= issuing && (i == 0);
        last1  <= issuing && (i == FW'(N_FEAT - 1));

        prod   <= f_s1 * w_rd;
        v2     <= v1;
        first2 <= first1;
        last2  <= last1;

        if (v2) acc <= (first2 ? bias[j] : acc) + $signed({{16{prod[31]}}, prod});
        acc_done <= v2 && last2;
    end

    // post-processing registers
    logic [5:0]         sh;
    logic signed [47:0] rounded, shifted;
    logic signed [15:0] lo, hi;
    always_comb begin
        sh = cfg[3*j][5:0];
        lo = cfg[3*j + 1];
        hi = cfg[3*j + 2];
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            j <= 0;
            i <= 0;
            base <= 0;
            y_valid <= 0;
            done <= 0;
            busy <= 0;
        end else begin
            y_valid <= 0;
            done <= 0;
            case (state)
                IDLE: if (start) begin
                    state <= ISSUE;
                    busy <= 1;
                    j <= 0;
                    i <= 0;
                    base <= 0;
                end
                ISSUE: begin
                    if (i == FW'(N_FEAT - 1)) begin
                        state <= DRAIN;
                    end else begin
                        i <= i + 1;
                    end
                end
                DRAIN: if (acc_done) state <= ROUND;
                ROUND: begin
                    rounded <= acc + ((sh == 0) ? 48'sd0 : (48'sd1 <<< (sh - 1)));
                    state <= SHIFT;
                end
                SHIFT: begin
                    shifted <= rounded >>> sh;
                    state <= CLAMP;
                end
                CLAMP: begin
                    y_valid <= 1;
                    y_idx <= j;
                    if (shifted < $signed({{32{lo[15]}}, lo}))      y_data <= lo;
                    else if (shifted > $signed({{32{hi[15]}}, hi})) y_data <= hi;
                    else                        y_data <= shifted[15:0];
                    if (j == OW'(N_OUT - 1)) begin
                        state <= IDLE;
                        done <= 1;
                        busy <= 0;
                    end else begin
                        j <= j + 1;
                        i <= 0;
                        base <= base + WW'(N_FEAT);
                        state <= ISSUE;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
