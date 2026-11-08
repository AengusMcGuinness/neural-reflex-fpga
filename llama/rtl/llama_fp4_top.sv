`timescale 1ns / 1ps
`default_nettype none
// FP4 llama2 (stories260K) accelerator on the Urbana: whole model + KV cache on chip.
//
//   host --UART 2 Mbaud--> uart_axi_bridge --+-- 0x4000_0000+ : status registers (below)
//                                            +-- otherwise    : llama_fp4 AXI-Lite control
//                                                                (prompt[], n_prompt, steps,
//                                                                 tokens[], n_out, ap_ctrl)
// Status registers (read only): +0x0 busy-cycle counter low, +0x4 high, +0x8 kernel clock in
// Hz, +0xC magic 0xF4F4_0001. The counter runs from the host's ap_start write until the
// kernel raises ap_done (seen on `interrupt`, which the host enables via GIE/IER), so the
// tokens/second the host reports is measured by the FPGA itself.
// LEDs: [0] MMCM locked, [1] kernel busy, [2] UART rx, [3] UART tx, [4] AXI error.
module llama_fp4_top #(
    parameter real CLKFBOUT_MULT = 10.0,  // 100 MHz * 10 = 1000 MHz VCO
    parameter real CLKOUT_DIVIDE = 8.0,   // 1000 MHz / 8 = 125 MHz
    parameter int  CLK_HZ = 125_000_000,
    parameter int  CTRL_AW = 14           // width of the kernel's s_axi_control address (C_S_AXI_CONTROL_ADDR_WIDTH)
) (
    input  wire         clk_100mhz,
    input  wire  [3:0]  btn,
    output logic [15:0] led,
    input  wire         uart_rxd,
    output wire         uart_txd
);
    // ---------------- clock: 100 MHz -> MMCM -> CLK_HZ (125 MHz) ----------------
    wire clkfb, clk_mmcm, clk, locked, clk_in;
    IBUF ibuf_clk (.I(clk_100mhz), .O(clk_in));
    MMCME2_BASE #(
        .CLKIN1_PERIOD(10.0),
        .CLKFBOUT_MULT_F(CLKFBOUT_MULT),
        .DIVCLK_DIVIDE(1),
        .CLKOUT0_DIVIDE_F(CLKOUT_DIVIDE)
    ) mmcm (
        .CLKIN1(clk_in), .CLKFBIN(clkfb), .CLKFBOUT(clkfb), .CLKOUT0(clk_mmcm),
        .CLKFBOUTB(), .CLKOUT0B(), .CLKOUT1(), .CLKOUT1B(), .CLKOUT2(), .CLKOUT2B(),
        .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6(),
        .LOCKED(locked), .PWRDWN(1'b0), .RST(btn[0])
    );
    BUFG bufg_clk (.I(clk_mmcm), .O(clk));
    logic [1:0] rst_sync;
    always_ff @(posedge clk) rst_sync <= {rst_sync[0], btn[0] | ~locked};
    wire rst = rst_sync[1];

    // ---------------- UART bridge ----------------
    logic [31:0] m_awaddr, m_wdata, m_araddr, m_rdata;
    logic [7:0] m_awlen, m_arlen;
    logic [2:0] m_awsize, m_arsize;
    logic [1:0] m_awburst, m_arburst, m_bresp, m_rresp;
    logic [3:0] m_wstrb;
    logic m_awvalid, m_awready, m_wlast, m_wvalid, m_wready, m_bvalid, m_bready;
    logic m_arvalid, m_arready, m_rlast, m_rvalid, m_rready;
    logic bridge_err;
    uart_axi_bridge #(.CLK_HZ(CLK_HZ), .BAUD(2_000_000)) bridge (
        .clk, .rst, .uart_rx(uart_rxd), .uart_tx(uart_txd),
        .m_axi_awaddr(m_awaddr), .m_axi_awlen(m_awlen), .m_axi_awsize(m_awsize), .m_axi_awburst(m_awburst),
        .m_axi_awvalid(m_awvalid), .m_axi_awready(m_awready), .m_axi_wdata(m_wdata), .m_axi_wstrb(m_wstrb),
        .m_axi_wlast(m_wlast), .m_axi_wvalid(m_wvalid), .m_axi_wready(m_wready), .m_axi_bresp(m_bresp),
        .m_axi_bvalid(m_bvalid), .m_axi_bready(m_bready), .m_axi_araddr(m_araddr), .m_axi_arlen(m_arlen),
        .m_axi_arsize(m_arsize), .m_axi_arburst(m_arburst), .m_axi_arvalid(m_arvalid), .m_axi_arready(m_arready),
        .m_axi_rdata(m_rdata), .m_axi_rresp(m_rresp), .m_axi_rlast(m_rlast), .m_axi_rvalid(m_rvalid),
        .m_axi_rready(m_rready), .err_resp(bridge_err));

    // ---------------- decode (addresses are held stable for the whole transaction) -------
    wire st_w = m_awaddr[30];
    wire st_r = m_araddr[30];

    // status block: writes accepted and ignored, reads return the registers
    logic s_bvalid, s_rvalid;
    logic [31:0] s_rdata;
    logic [63:0] cycles;
    always_ff @(posedge clk) begin
        if (rst) begin
            s_bvalid <= 1'b0;
            s_rvalid <= 1'b0;
        end else begin
            if (m_awvalid && m_wvalid && st_w && !s_bvalid) s_bvalid <= 1'b1;
            else if (s_bvalid && m_bready) s_bvalid <= 1'b0;
            if (m_arvalid && st_r && !s_rvalid) begin
                s_rvalid <= 1'b1;
                case (m_araddr[3:2])
                    2'd0: s_rdata <= cycles[31:0];
                    2'd1: s_rdata <= cycles[63:32];
                    2'd2: s_rdata <= CLK_HZ;
                    default: s_rdata <= 32'hF4F4_0001;
                endcase
            end else if (s_rvalid && m_rready) s_rvalid <= 1'b0;
        end
    end
    // single-beat handshakes for the status block: accept AW/W together with the response
    wire s_awready = st_w & m_awvalid & m_wvalid & ~s_bvalid;
    wire s_arready = st_r & m_arvalid & ~s_rvalid;

    // kernel control
    logic c_awready, c_wready, c_bvalid, c_arready, c_rvalid;
    logic [1:0] c_bresp, c_rresp;
    logic [31:0] c_rdata;
    wire kernel_irq;
    llama_fp4_0 kernel (
        .ap_clk(clk),
        .ap_rst_n(~rst),
        .s_axi_control_AWVALID(m_awvalid & ~st_w), .s_axi_control_AWREADY(c_awready),
        .s_axi_control_AWADDR(m_awaddr[CTRL_AW-1:0]),
        .s_axi_control_WVALID(m_wvalid & ~st_w), .s_axi_control_WREADY(c_wready),
        .s_axi_control_WDATA(m_wdata), .s_axi_control_WSTRB(m_wstrb),
        .s_axi_control_ARVALID(m_arvalid & ~st_r), .s_axi_control_ARREADY(c_arready),
        .s_axi_control_ARADDR(m_araddr[CTRL_AW-1:0]),
        .s_axi_control_RVALID(c_rvalid), .s_axi_control_RREADY(m_rready & ~st_r),
        .s_axi_control_RDATA(c_rdata), .s_axi_control_RRESP(c_rresp),
        .s_axi_control_BVALID(c_bvalid), .s_axi_control_BREADY(m_bready & ~st_w),
        .s_axi_control_BRESP(c_bresp),
        .interrupt(kernel_irq)
    );

    assign m_awready = st_w ? s_awready : c_awready;
    assign m_wready = st_w ? s_awready : c_wready;
    assign m_bvalid = st_w ? s_bvalid : c_bvalid;
    assign m_bresp = st_w ? 2'b00 : c_bresp;
    assign m_arready = st_r ? s_arready : c_arready;
    assign m_rvalid = st_r ? s_rvalid : c_rvalid;
    assign m_rdata = st_r ? s_rdata : c_rdata;
    assign m_rresp = st_r ? 2'b00 : c_rresp;
    assign m_rlast = 1'b1;

    // ---------------- busy-cycle counter ----------------
    // start: the host's write of ap_start (control offset 0, bit 0); stop: interrupt (ap_done)
    wire start_wr = m_awvalid & m_awready & ~st_w & (m_awaddr[CTRL_AW-1:0] == '0) & m_wdata[0];
    logic busy;
    always_ff @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0;
            cycles <= '0;
        end else if (start_wr) begin
            busy <= 1'b1;
            cycles <= '0;
        end else if (busy) begin
            if (kernel_irq) busy <= 1'b0;
            else cycles <= cycles + 1'b1;
        end
    end

    // ---------------- LEDs ----------------
    logic [22:0] rx_stretch, tx_stretch;
    always_ff @(posedge clk) begin
        rx_stretch <= ~uart_rxd ? '1 : (rx_stretch != 0 ? rx_stretch - 1'b1 : '0);
        tx_stretch <= ~uart_txd ? '1 : (tx_stretch != 0 ? tx_stretch - 1'b1 : '0);
    end
    assign led = {11'd0, bridge_err, tx_stretch != 0, rx_stretch != 0, busy, locked};
endmodule
`default_nettype wire
