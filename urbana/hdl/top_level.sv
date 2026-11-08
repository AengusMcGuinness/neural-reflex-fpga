`timescale 1ns / 1ps
`default_nettype none

// Urbana (Spartan-7) EMG decoder: the Mac sends feature frames over the USB UART, the FPGA
// runs the fixed-point ridge decoder and sends the outputs (robot servo ticks) back.
//   btn[0]   reset
//   led      bar graph of output 0 between its clamp limits
//   rgb0     flashes on each decoded frame; rgb1 stays lit after a bad frame until the next good one
//   debug_*  PMOD B pins for the logic analyzer: UART lines and decoder busy (compute latency)
// N_FEAT / N_OUT must match build/quant_params.json (export_weights.py prints them).
module top_level #(
    parameter int N_FEAT = 175,
    parameter int N_OUT = 2,
    parameter int BAUD_RATE = 921_600,
    parameter int CLK_HZ = 100_000_000,
    parameter WEIGHTS_FILE = "weights.mem",
    parameter BIAS_FILE = "bias.mem",
    parameter CFG_FILE = "cfg.mem"
)(
    input wire          clk_100mhz,
    input wire [3:0]    btn,
    output logic [15:0] led,
    output logic [2:0]  rgb0,
    output logic [2:0]  rgb1,
    input wire          uart_rxd,
    output logic        uart_txd,
    output logic        debug_uart_rxd,
    output logic        debug_uart_txd,
    output logic        debug_busy
);
    logic sys_rst;
    assign sys_rst = btn[0];

    // ---------------- UART in ----------------
    logic [1:0] rxd_sync;
    always_ff @(posedge clk_100mhz) rxd_sync <= {rxd_sync[0], uart_rxd};

    logic       rx_valid;
    logic [7:0] rx_byte;
    uart_receive #(.INPUT_CLOCK_FREQ(CLK_HZ), .BAUD_RATE(BAUD_RATE)) urx (
        .clk(clk_100mhz), .rst(sys_rst), .din(rxd_sync[1]),
        .dout_valid(rx_valid), .dout(rx_byte));

    // ---------------- frames -> decoder ----------------
    logic                              feat_we, w_we, b_we, cfg_we;
    logic [$clog2(N_FEAT)-1:0]         feat_addr;
    logic [$clog2(N_FEAT*N_OUT)-1:0]   w_addr;
    logic [$clog2(N_OUT)-1:0]          b_addr;
    logic [$clog2(3*N_OUT)-1:0]        cfg_addr;
    logic [15:0]                       feat_data, w_data, cfg_data;
    logic [47:0]                       b_data;
    logic                              frame_ok, frame_bad;
    logic [7:0]                        frame_cmd;

    frame_rx #(.N_FEAT(N_FEAT), .N_OUT(N_OUT), .TIMEOUT_CYCLES(CLK_HZ / 100)) frx (
        .clk(clk_100mhz), .rst(sys_rst), .byte_valid(rx_valid), .byte_in(rx_byte),
        .feat_we(feat_we), .feat_addr(feat_addr), .feat_data(feat_data),
        .w_we(w_we), .w_addr(w_addr), .w_data(w_data),
        .b_we(b_we), .b_addr(b_addr), .b_data(b_data),
        .cfg_we(cfg_we), .cfg_addr(cfg_addr), .cfg_data(cfg_data),
        .frame_ok(frame_ok), .frame_bad(frame_bad), .frame_cmd(frame_cmd));

    logic                       y_valid, mac_done, mac_busy;
    logic [$clog2(N_OUT)-1:0]   y_idx;
    logic signed [15:0]         y_data, lo0, hi0;

    ridge_mac #(.N_FEAT(N_FEAT), .N_OUT(N_OUT), .WEIGHTS_FILE(WEIGHTS_FILE),
                .BIAS_FILE(BIAS_FILE), .CFG_FILE(CFG_FILE)) mac (
        .clk(clk_100mhz), .rst(sys_rst),
        .feat_we(feat_we), .feat_addr(feat_addr), .feat_data(feat_data),
        .w_we(w_we), .w_addr(w_addr), .w_data(w_data),
        .b_we(b_we), .b_addr(b_addr), .b_data(b_data),
        .cfg_we(cfg_we), .cfg_addr(cfg_addr), .cfg_data(cfg_data),
        .start(frame_ok && frame_cmd == 8'h01),
        .y_valid(y_valid), .y_idx(y_idx), .y_data(y_data),
        .done(mac_done), .busy(mac_busy), .lo0(lo0), .hi0(hi0));

    logic [16*N_OUT-1:0] y_flat;
    logic                result_ready;   // done delayed a cycle so the last y is in y_flat
    always_ff @(posedge clk_100mhz) begin
        if (y_valid) y_flat[16*y_idx +: 16] <= y_data;
        result_ready <= mac_done;
    end

    // ---------------- decoder -> UART out ----------------
    logic       tx_trigger, tx_busy;
    logic [7:0] tx_byte;
    frame_tx #(.N_OUT(N_OUT)) ftx (
        .clk(clk_100mhz), .rst(sys_rst),
        .send_result(result_ready), .y_flat(y_flat),
        .send_load_ok(frame_ok && frame_cmd == 8'h02),
        .send_error(frame_bad), .error_cmd(frame_cmd),
        .tx_trigger(tx_trigger), .tx_byte(tx_byte), .tx_busy(tx_busy), .busy());

    uart_transmit #(.INPUT_CLOCK_FREQ(CLK_HZ), .BAUD_RATE(BAUD_RATE)) utx (
        .clk(clk_100mhz), .rst(sys_rst), .din(tx_byte), .trigger(tx_trigger),
        .busy(tx_busy), .dout(uart_txd));

    // ---------------- LEDs and debug ----------------
    // bar graph: led[k] lights when y0 passes lo + (k+1)/16 of the clamp range
    logic signed [16:0] span;
    assign span = hi0 - lo0;
    always_ff @(posedge clk_100mhz) begin
        for (int k = 0; k < 16; k++)
            led[k] <= $signed(y_flat[15:0]) >= lo0 + ((span * (k + 1)) >>> 4);
    end

    logic [23:0] activity, error_hold;
    always_ff @(posedge clk_100mhz) begin
        if (sys_rst) begin
            activity <= 0;
            error_hold <= 0;
        end else begin
            activity <= result_ready ? 24'd5_000_000 : (activity != 0 ? activity - 1 : 0);
            if (frame_bad) error_hold <= 1;
            else if (frame_ok) error_hold <= 0;
        end
    end
    assign rgb0 = {3{activity != 0}};
    assign rgb1 = {3{error_hold[0]}};

    assign debug_uart_rxd = uart_rxd;
    assign debug_uart_txd = uart_txd;
    assign debug_busy = mac_busy;
endmodule

`default_nettype wire
