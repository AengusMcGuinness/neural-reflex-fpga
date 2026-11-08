`timescale 1ns / 1ps
`default_nettype none
// FP4 llama2 (stories260K) accelerator on the Opal Kelly XEM7310-A75 (Artix-7 xc7a75t).
// Whole model + KV cache on chip, as on the Urbana, with a wider kernel and a USB host link.
//
//   host --USB/FrontPanel--> PipeIn 0x80 --async FIFO--> fp_cmd_proc --AXI-Lite--> llama_fp4
//   host <-----------------  PipeOut 0xA0 <-async FIFO--      |
//                                                      busy-cycle counter, status block
// Clocks: okClk (100.8 MHz, from the host interface) runs the FrontPanel endpoints; the
// kernel and command processor run at CLK_HZ from the board's 200 MHz oscillator via an MMCM.
//
// Endpoints:
//   WireIn  0x00  bit 0: reset (kernel, command processor, FIFOs)
//   WireOut 0x20  words waiting in the response FIFO
//   WireOut 0x21  [0] MMCM locked, [1] kernel busy, [2] command FIFO overflowed (sticky),
//                 [3] response FIFO overflowed (sticky), [31:16] 0xF4F4
//   PipeIn  0x80  command words (see fp_cmd_proc.sv)
//   PipeOut 0xA0  response words
// LEDs (active low, open drain): [0] locked, [1] busy, [2] command activity,
//   [3] response activity, [4] an overflow happened.
module llama_xem7310_top #(
    parameter real CLKFBOUT_MULT = 5.0,   // 200 MHz * 5 = 1000 MHz VCO
    parameter real CLKOUT_DIVIDE = 7.5,   // 1000 MHz / 7.5 = 133.3 MHz
    parameter int  CLK_HZ = 133_333_333,
    parameter int  CTRL_AW = 11
) (
    input  wire  [4:0]  okUH,
    output wire  [2:0]  okHU,
    inout  wire  [31:0] okUHU,
    inout  wire         okAA,
    input  wire         sys_clkp,
    input  wire         sys_clkn,
    output wire  [7:0]  led
);
    // ---------------- FrontPanel host interface (okClk domain) ----------------
    wire okClk;
    wire [112:0] okHE;
    wire [64:0] okEH;
    localparam int NEH = 4;
    wire [NEH*65-1:0] okEHx;

    okHost host (
        .okUH(okUH), .okHU(okHU), .okUHU(okUHU), .okAA(okAA),
        .okClk(okClk), .okHE(okHE), .okEH(okEH)
    );
    okWireOR #(.N(NEH)) wire_or (.okEH(okEH), .okEHx(okEHx));

    wire [31:0] ep00;
    okWireIn wi00 (.okHE(okHE), .ep_addr(8'h00), .ep_dataout(ep00));
    wire host_rst = ep00[0];

    wire pipe_wr;
    wire [31:0] pipe_wdata;
    okPipeIn pi80 (.okHE(okHE), .okEH(okEHx[0*65 +: 65]), .ep_addr(8'h80),
                   .ep_write(pipe_wr), .ep_dataout(pipe_wdata));

    wire pipe_rd;
    wire [31:0] pipe_rdata;
    okPipeOut poA0 (.okHE(okHE), .okEH(okEHx[1*65 +: 65]), .ep_addr(8'hA0),
                    .ep_read(pipe_rd), .ep_datain(pipe_rdata));

    wire [11:0] resp_count;
    logic [31:0] wo20, wo21;
    okWireOut wo_20 (.okHE(okHE), .okEH(okEHx[2*65 +: 65]), .ep_addr(8'h20), .ep_datain(wo20));
    okWireOut wo_21 (.okHE(okHE), .okEH(okEHx[3*65 +: 65]), .ep_addr(8'h21), .ep_datain(wo21));

    // ---------------- kernel clock: 200 MHz -> MMCM -> CLK_HZ ----------------
    wire sys_clk, clkfb, clk_mmcm, clk, locked;
    IBUFDS ibufds_clk (.I(sys_clkp), .IB(sys_clkn), .O(sys_clk));
    MMCME2_BASE #(
        .CLKIN1_PERIOD(5.0),
        .CLKFBOUT_MULT_F(CLKFBOUT_MULT),
        .DIVCLK_DIVIDE(1),
        .CLKOUT0_DIVIDE_F(CLKOUT_DIVIDE)
    ) mmcm (
        .CLKIN1(sys_clk), .CLKFBIN(clkfb), .CLKFBOUT(clkfb), .CLKOUT0(clk_mmcm),
        .CLKFBOUTB(), .CLKOUT0B(), .CLKOUT1(), .CLKOUT1B(), .CLKOUT2(), .CLKOUT2B(),
        .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6(),
        .LOCKED(locked), .PWRDWN(1'b0), .RST(1'b0)
    );
    BUFG bufg_clk (.I(clk_mmcm), .O(clk));

    (* ASYNC_REG = "TRUE" *) logic [2:0] rst_sync;
    always_ff @(posedge clk) rst_sync <= {rst_sync[1:0], host_rst | ~locked};
    wire rst = rst_sync[2];

    // okClk-side reset for the command FIFO (its write side lives in okClk)
    (* ASYNC_REG = "TRUE" *) logic [1:0] okrst_sync;
    always_ff @(posedge okClk) okrst_sync <= {okrst_sync[0], host_rst | ~locked};
    wire ok_rst = okrst_sync[1];

    // ---------------- clock-crossing FIFOs ----------------
    wire cmd_full, cmd_empty, cmd_rd, cmd_wr_busy, cmd_rd_busy;
    wire [31:0] cmd_data;
    xpm_fifo_async #(
        .FIFO_MEMORY_TYPE("block"), .FIFO_WRITE_DEPTH(2048),
        .WRITE_DATA_WIDTH(32), .READ_DATA_WIDTH(32),
        .READ_MODE("fwft"), .FIFO_READ_LATENCY(0),
        .WR_DATA_COUNT_WIDTH(12), .RD_DATA_COUNT_WIDTH(12),
        .USE_ADV_FEATURES("0000"), .CDC_SYNC_STAGES(2), .RELATED_CLOCKS(0),
        .ECC_MODE("no_ecc"), .DOUT_RESET_VALUE("0"), .FULL_RESET_VALUE(0),
        .PROG_EMPTY_THRESH(10), .PROG_FULL_THRESH(10), .SIM_ASSERT_CHK(0), .WAKEUP_TIME(0)
    ) cmd_fifo (
        .sleep(1'b0), .rst(ok_rst),
        .wr_clk(okClk), .wr_en(pipe_wr & ~cmd_full & ~cmd_wr_busy), .din(pipe_wdata), .full(cmd_full),
        .wr_rst_busy(cmd_wr_busy),
        .rd_clk(clk), .rd_en(cmd_rd), .dout(cmd_data), .empty(cmd_empty), .rd_rst_busy(cmd_rd_busy),
        .overflow(), .prog_full(), .wr_data_count(), .almost_full(), .wr_ack(), .underflow(),
        .prog_empty(), .rd_data_count(), .almost_empty(), .data_valid(),
        .injectsbiterr(1'b0), .injectdbiterr(1'b0), .sbiterr(), .dbiterr()
    );

    wire resp_full, resp_empty, resp_wr_busy, resp_wr;
    wire [31:0] resp_data;
    xpm_fifo_async #(
        .FIFO_MEMORY_TYPE("block"), .FIFO_WRITE_DEPTH(2048),
        .WRITE_DATA_WIDTH(32), .READ_DATA_WIDTH(32),
        .READ_MODE("std"), .FIFO_READ_LATENCY(1),
        .WR_DATA_COUNT_WIDTH(12), .RD_DATA_COUNT_WIDTH(12),
        .USE_ADV_FEATURES("0400"), .CDC_SYNC_STAGES(2), .RELATED_CLOCKS(0),
        .ECC_MODE("no_ecc"), .DOUT_RESET_VALUE("0"), .FULL_RESET_VALUE(0),
        .PROG_EMPTY_THRESH(10), .PROG_FULL_THRESH(10), .SIM_ASSERT_CHK(0), .WAKEUP_TIME(0)
    ) resp_fifo (
        .sleep(1'b0), .rst(rst),
        .wr_clk(clk), .wr_en(resp_wr & ~resp_full & ~resp_wr_busy), .din(resp_data), .full(resp_full),
        .wr_rst_busy(resp_wr_busy),
        .rd_clk(okClk), .rd_en(pipe_rd), .dout(pipe_rdata), .empty(resp_empty), .rd_rst_busy(),
        .rd_data_count(resp_count),
        .overflow(), .prog_full(), .wr_data_count(), .almost_full(), .wr_ack(), .underflow(),
        .prog_empty(), .almost_empty(), .data_valid(),
        .injectsbiterr(1'b0), .injectdbiterr(1'b0), .sbiterr(), .dbiterr()
    );

    // ---------------- command processor + kernel (clk domain) ----------------
    logic [CTRL_AW-1:0] awaddr, araddr;
    logic [31:0] wdata, rdata;
    logic awvalid, awready, wvalid, wready, bvalid, bready, arvalid, arready, rvalid, rready;
    logic [1:0] bresp, rresp;
    logic busy;
    logic [63:0] cycles;

    fp_cmd_proc #(.CLK_HZ(CLK_HZ), .CTRL_AW(CTRL_AW)) proc (
        .clk, .rst,
        .cmd_data, .cmd_empty(cmd_empty | cmd_rd_busy), .cmd_rd,
        .resp_data, .resp_wr,
        .awaddr, .awvalid, .awready, .wdata, .wvalid, .wready, .bvalid, .bready,
        .araddr, .arvalid, .arready, .rdata, .rvalid, .rready,
        .busy, .cycles
    );

    llama_fp4_0 kernel (
        .ap_clk(clk),
        .ap_rst_n(~rst),
        .s_axi_control_AWVALID(awvalid), .s_axi_control_AWREADY(awready),
        .s_axi_control_AWADDR(awaddr),
        .s_axi_control_WVALID(wvalid), .s_axi_control_WREADY(wready),
        .s_axi_control_WDATA(wdata), .s_axi_control_WSTRB(4'hF),
        .s_axi_control_ARVALID(arvalid), .s_axi_control_ARREADY(arready),
        .s_axi_control_ARADDR(araddr),
        .s_axi_control_RVALID(rvalid), .s_axi_control_RREADY(rready),
        .s_axi_control_RDATA(rdata), .s_axi_control_RRESP(rresp),
        .s_axi_control_BVALID(bvalid), .s_axi_control_BREADY(bready),
        .s_axi_control_BRESP(bresp),
        .interrupt()
    );

    // ---------------- status wires (okClk domain) ----------------
    logic cmd_ovf, resp_ovf;
    (* ASYNC_REG = "TRUE" *) logic [1:0] busy_s, lock_s, rovf_s;
    logic resp_ovf_k;  // clk domain, sticky
    always_ff @(posedge clk) begin
        if (rst) resp_ovf_k <= 1'b0;
        else if (resp_wr && resp_full) resp_ovf_k <= 1'b1;
    end
    always_ff @(posedge okClk) begin
        busy_s <= {busy_s[0], busy};
        lock_s <= {lock_s[0], locked};
        rovf_s <= {rovf_s[0], resp_ovf_k};
        if (ok_rst) cmd_ovf <= 1'b0;
        else if (pipe_wr && cmd_full) cmd_ovf <= 1'b1;
        wo20 <= {20'd0, resp_count};
        wo21 <= {16'hF4F4, 12'd0, rovf_s[1], cmd_ovf, busy_s[1], lock_s[1]};
    end

    // ---------------- LEDs (XEM7310: drive 0 to light, Z otherwise) ----------------
    logic [22:0] cmd_stretch, resp_stretch;
    always_ff @(posedge okClk) begin
        cmd_stretch <= pipe_wr ? '1 : (cmd_stretch != 0 ? cmd_stretch - 1'b1 : '0);
        resp_stretch <= pipe_rd ? '1 : (resp_stretch != 0 ? resp_stretch - 1'b1 : '0);
    end
    wire [7:0] led_on = {3'd0, cmd_ovf | rovf_s[1], resp_stretch != 0, cmd_stretch != 0,
                         busy_s[1], lock_s[1]};
    genvar gi;
    generate
        for (gi = 0; gi < 8; gi++) begin : g_led
            assign led[gi] = led_on[gi] ? 1'b0 : 1'bz;
        end
    endgenerate
endmodule
`default_nettype wire
