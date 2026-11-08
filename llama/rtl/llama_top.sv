`timescale 1ns / 1ps
`default_nettype none
// llama2.c on the RealDigital Urbana (xc7s50csga324-1).
//
//   host (Mac) --UART 2 Mbaud--> uart_axi_bridge --+-- addr[31]=1 --> llama_forward s_axi_control
//                                                  +-- addr[31]=0 --> axi_wb_ddr port 0 --+
//   llama_forward m_axi_gmem --------------------------------------> axi_wb_ddr port 1 --+--> ddr3_top (UberDDR3, 256 MB)
//
// Clocks come from the 6.205 lab06 clock wizard: 83.33 MHz controller/user clock (everything
// here runs on it), 333.33 MHz DDR3 clock (+90 deg), 200 MHz IDELAY reference.
// ddr3_top parameters are exactly the ones in the lab06 high_definition_frame_buffer.
// LEDs: [0] MMCM locked, [1] DDR3 calibrated, [2] AXI error seen, [3] kernel busy,
//       [4] UART rx activity, [5] UART tx activity.  btn[0] = reset.
module llama_top (
    input  wire         clk_100mhz,
    input  wire  [3:0]  btn,
    output logic [15:0] led,
    input  wire         uart_rxd,
    output wire         uart_txd,
    inout  wire  [15:0] ddr3_dq,
    inout  wire  [1:0]  ddr3_dqs_n,
    inout  wire  [1:0]  ddr3_dqs_p,
    output wire  [13:0] ddr3_addr,
    output wire  [2:0]  ddr3_ba,
    output wire         ddr3_ras_n,
    output wire         ddr3_cas_n,
    output wire         ddr3_we_n,
    output wire         ddr3_reset_n,
    output wire         ddr3_clk_p,
    output wire         ddr3_clk_n,
    output wire         ddr3_clke,
    output wire  [1:0]  ddr3_dm,
    output wire         ddr3_odt
);
    // ---------------- clocks and reset ----------------
    wire clk, clk_ddr3, clk_ddr3_90, clk_ref200, locked;
    lab06_clk_wiz lcw (
        .reset(btn[0]),
        .clk_in1(clk_100mhz),
        .clk_camera(clk_ref200),
        .clk_xc(),
        .clk_passthrough(),
        .clk_controller(clk),
        .clk_ddr3(clk_ddr3),
        .clk_ddr3_90(clk_ddr3_90),
        .locked(locked)
    );
    logic [1:0] rst_sync;
    always_ff @(posedge clk) rst_sync <= {rst_sync[0], btn[0] | ~locked};
    wire rst = rst_sync[1];

    // ---------------- UART bridge (AXI4 master) ----------------
    logic [31:0] m_awaddr, m_wdata, m_araddr, m_rdata;
    logic [7:0] m_awlen, m_arlen;
    logic [2:0] m_awsize, m_arsize;
    logic [1:0] m_awburst, m_arburst, m_bresp, m_rresp;
    logic [3:0] m_wstrb;
    logic m_awvalid, m_awready, m_wlast, m_wvalid, m_wready, m_bvalid, m_bready;
    logic m_arvalid, m_arready, m_rlast, m_rvalid, m_rready;
    logic bridge_err;
    uart_axi_bridge #(.CLK_HZ(83_333_333), .BAUD(2_000_000)) bridge (
        .clk, .rst, .uart_rx(uart_rxd), .uart_tx(uart_txd),
        .m_axi_awaddr(m_awaddr), .m_axi_awlen(m_awlen), .m_axi_awsize(m_awsize), .m_axi_awburst(m_awburst),
        .m_axi_awvalid(m_awvalid), .m_axi_awready(m_awready), .m_axi_wdata(m_wdata), .m_axi_wstrb(m_wstrb),
        .m_axi_wlast(m_wlast), .m_axi_wvalid(m_wvalid), .m_axi_wready(m_wready), .m_axi_bresp(m_bresp),
        .m_axi_bvalid(m_bvalid), .m_axi_bready(m_bready), .m_axi_araddr(m_araddr), .m_axi_arlen(m_arlen),
        .m_axi_arsize(m_arsize), .m_axi_arburst(m_arburst), .m_axi_arvalid(m_arvalid), .m_axi_arready(m_arready),
        .m_axi_rdata(m_rdata), .m_axi_rresp(m_rresp), .m_axi_rlast(m_rlast), .m_axi_rvalid(m_rvalid),
        .m_axi_rready(m_rready), .err_resp(bridge_err));

    // ---------------- address decode: bit 31 selects the kernel control registers ----------
    // The bridge holds awaddr/araddr constant for the whole transaction, so a combinational
    // select is stable from VALID to the response handshake.
    wire sel_w = m_awaddr[31];
    wire sel_r = m_araddr[31];

    // kernel AXI-Lite control
    logic c_awready, c_wready, c_bvalid, c_arready, c_rvalid;
    logic [1:0] c_bresp, c_rresp;
    logic [31:0] c_rdata;

    // DDR adapter ports (0 = bridge, 1 = kernel)
    logic [0:0] p_awid[2], p_bid[2], p_arid[2], p_rid[2];
    logic [31:0] p_awaddr[2], p_wdata[2], p_araddr[2], p_rdata[2];
    logic [7:0] p_awlen[2], p_arlen[2];
    logic [3:0] p_wstrb[2];
    logic [1:0] p_bresp[2], p_rresp[2];
    logic p_awvalid[2], p_awready[2], p_wlast[2], p_wvalid[2], p_wready[2], p_bvalid[2], p_bready[2];
    logic p_arvalid[2], p_arready[2], p_rlast[2], p_rvalid[2], p_rready[2];

    assign p_awid[0] = 1'b0;
    assign p_awaddr[0] = m_awaddr;
    assign p_awlen[0] = m_awlen;
    assign p_awvalid[0] = m_awvalid & ~sel_w;
    assign p_wdata[0] = m_wdata;
    assign p_wstrb[0] = m_wstrb;
    assign p_wlast[0] = m_wlast;
    assign p_wvalid[0] = m_wvalid & ~sel_w;
    assign p_bready[0] = m_bready & ~sel_w;
    assign p_arid[0] = 1'b0;
    assign p_araddr[0] = m_araddr;
    assign p_arlen[0] = m_arlen;
    assign p_arvalid[0] = m_arvalid & ~sel_r;
    assign p_rready[0] = m_rready & ~sel_r;

    assign m_awready = sel_w ? c_awready : p_awready[0];
    assign m_wready = sel_w ? c_wready : p_wready[0];
    assign m_bvalid = sel_w ? c_bvalid : p_bvalid[0];
    assign m_bresp = sel_w ? c_bresp : p_bresp[0];
    assign m_arready = sel_r ? c_arready : p_arready[0];
    assign m_rvalid = sel_r ? c_rvalid : p_rvalid[0];
    assign m_rdata = sel_r ? c_rdata : p_rdata[0];
    assign m_rresp = sel_r ? c_rresp : p_rresp[0];
    assign m_rlast = sel_r ? 1'b1 : p_rlast[0];

    // ---------------- the llama2 kernel (Vitis HLS IP) ----------------
    wire [1:0] g_awlock, g_arlock;
    wire [3:0] g_awcache, g_arcache, g_awqos, g_arqos, g_awregion, g_arregion;
    wire [2:0] g_awprot, g_arprot, g_awsize, g_arsize;
    wire [1:0] g_awburst, g_arburst;
    wire [0:0] g_wid;
    wire kernel_irq;
    llama_forward_0 kernel (
        .ap_clk(clk),
        .ap_rst_n(~rst),
        .m_axi_gmem_AWVALID(p_awvalid[1]), .m_axi_gmem_AWREADY(p_awready[1]),
        .m_axi_gmem_AWADDR(p_awaddr[1]), .m_axi_gmem_AWID(p_awid[1]), .m_axi_gmem_AWLEN(p_awlen[1]),
        .m_axi_gmem_AWSIZE(g_awsize), .m_axi_gmem_AWBURST(g_awburst), .m_axi_gmem_AWLOCK(g_awlock),
        .m_axi_gmem_AWCACHE(g_awcache), .m_axi_gmem_AWPROT(g_awprot), .m_axi_gmem_AWQOS(g_awqos),
        .m_axi_gmem_AWREGION(g_awregion),
        .m_axi_gmem_WVALID(p_wvalid[1]), .m_axi_gmem_WREADY(p_wready[1]), .m_axi_gmem_WDATA(p_wdata[1]),
        .m_axi_gmem_WSTRB(p_wstrb[1]), .m_axi_gmem_WLAST(p_wlast[1]), .m_axi_gmem_WID(g_wid),
        .m_axi_gmem_ARVALID(p_arvalid[1]), .m_axi_gmem_ARREADY(p_arready[1]),
        .m_axi_gmem_ARADDR(p_araddr[1]), .m_axi_gmem_ARID(p_arid[1]), .m_axi_gmem_ARLEN(p_arlen[1]),
        .m_axi_gmem_ARSIZE(g_arsize), .m_axi_gmem_ARBURST(g_arburst), .m_axi_gmem_ARLOCK(g_arlock),
        .m_axi_gmem_ARCACHE(g_arcache), .m_axi_gmem_ARPROT(g_arprot), .m_axi_gmem_ARQOS(g_arqos),
        .m_axi_gmem_ARREGION(g_arregion),
        .m_axi_gmem_RVALID(p_rvalid[1]), .m_axi_gmem_RREADY(p_rready[1]), .m_axi_gmem_RDATA(p_rdata[1]),
        .m_axi_gmem_RLAST(p_rlast[1]), .m_axi_gmem_RID(p_rid[1]),
        .m_axi_gmem_RRESP(p_rresp[1]),
        .m_axi_gmem_BVALID(p_bvalid[1]), .m_axi_gmem_BREADY(p_bready[1]), .m_axi_gmem_BRESP(p_bresp[1]),
        .m_axi_gmem_BID(p_bid[1]),
        .s_axi_control_AWVALID(m_awvalid & sel_w), .s_axi_control_AWREADY(c_awready),
        .s_axi_control_AWADDR(m_awaddr[7:0]),
        .s_axi_control_WVALID(m_wvalid & sel_w), .s_axi_control_WREADY(c_wready),
        .s_axi_control_WDATA(m_wdata), .s_axi_control_WSTRB(m_wstrb),
        .s_axi_control_ARVALID(m_arvalid & sel_r), .s_axi_control_ARREADY(c_arready),
        .s_axi_control_ARADDR(m_araddr[7:0]),
        .s_axi_control_RVALID(c_rvalid), .s_axi_control_RREADY(m_rready & sel_r),
        .s_axi_control_RDATA(c_rdata), .s_axi_control_RRESP(c_rresp),
        .s_axi_control_BVALID(c_bvalid), .s_axi_control_BREADY(m_bready & sel_w),
        .s_axi_control_BRESP(c_bresp),
        .interrupt(kernel_irq)
    );

    // ---------------- AXI -> Wishbone -> UberDDR3 ----------------
    logic wb_stb, wb_we, wb_stall, wb_ack;
    logic [23:0] wb_addr;
    logic [127:0] wb_wdata, wb_rdata;
    logic [15:0] wb_sel;
    axi_wb_ddr #(.ID_W(1), .WB_AW(24)) adapter (
        .clk, .rst,
        .s_awid(p_awid), .s_awaddr(p_awaddr), .s_awlen(p_awlen), .s_awvalid(p_awvalid), .s_awready(p_awready),
        .s_wdata(p_wdata), .s_wstrb(p_wstrb), .s_wlast(p_wlast), .s_wvalid(p_wvalid), .s_wready(p_wready),
        .s_bid(p_bid), .s_bresp(p_bresp), .s_bvalid(p_bvalid), .s_bready(p_bready),
        .s_arid(p_arid), .s_araddr(p_araddr), .s_arlen(p_arlen), .s_arvalid(p_arvalid), .s_arready(p_arready),
        .s_rid(p_rid), .s_rdata(p_rdata), .s_rresp(p_rresp), .s_rlast(p_rlast), .s_rvalid(p_rvalid), .s_rready(p_rready),
        .wb_stb, .wb_we, .wb_addr, .wb_data(wb_wdata), .wb_sel, .wb_stall, .wb_ack, .wb_rdata);

    wire calib_done;
    ddr3_top #(
        .CONTROLLER_CLK_PERIOD(12_000),
        .DDR3_CLK_PERIOD(3_000),
        .ROW_BITS(14),
        .COL_BITS(10),
        .BA_BITS(3),
        .BYTE_LANES(2),
        .AUX_WIDTH(16),
        .WB2_ADDR_BITS(32),
        .WB2_DATA_BITS(32),
        .MICRON_SIM(0),
        .ODELAY_SUPPORTED(0),
        .SECOND_WISHBONE(0),
        .ECC_ENABLE(0),
        .WB_ERROR(0)
    ) ddr3 (
        .i_controller_clk(clk),
        .i_ddr3_clk(clk_ddr3),
        .i_ref_clk(clk_ref200),
        .i_ddr3_clk_90(clk_ddr3_90),
        .i_rst_n(~btn[0] & locked),
        .i_wb_cyc(1'b1),
        .i_wb_stb(wb_stb),
        .i_wb_we(wb_we),
        .i_wb_addr(wb_addr),
        .i_wb_data(wb_wdata),
        .i_wb_sel(wb_sel),
        .i_aux({15'd0, wb_we}),
        .o_wb_stall(wb_stall),
        .o_wb_ack(wb_ack),
        .o_wb_err(),
        .o_wb_data(wb_rdata),
        .o_aux(),
        .i_wb2_cyc(1'b0),
        .i_wb2_stb(1'b0),
        .i_wb2_we(1'b0),
        .i_wb2_addr(32'd0),
        .i_wb2_data(32'd0),
        .i_wb2_sel(4'd0),
        .o_wb2_stall(),
        .o_wb2_ack(),
        .o_wb2_data(),
        .o_ddr3_clk_p(ddr3_clk_p),
        .o_ddr3_clk_n(ddr3_clk_n),
        .o_ddr3_reset_n(ddr3_reset_n),
        .o_ddr3_cke(ddr3_clke),
        .o_ddr3_cs_n(), // not routed on the Urbana
        .o_ddr3_ras_n(ddr3_ras_n),
        .o_ddr3_cas_n(ddr3_cas_n),
        .o_ddr3_we_n(ddr3_we_n),
        .o_ddr3_addr(ddr3_addr),
        .o_ddr3_ba_addr(ddr3_ba),
        .io_ddr3_dq(ddr3_dq),
        .io_ddr3_dqs(ddr3_dqs_p),
        .io_ddr3_dqs_n(ddr3_dqs_n),
        .o_ddr3_dm(ddr3_dm),
        .o_ddr3_odt(ddr3_odt),
        .o_calib_complete(calib_done),
        .o_debug1(),
        .i_user_self_refresh(1'b0),
        .uart_tx()
    );

    // ---------------- status LEDs ----------------
    logic [22:0] rx_stretch, tx_stretch;
    always_ff @(posedge clk) begin
        rx_stretch <= ~uart_rxd ? '1 : (rx_stretch != 0 ? rx_stretch - 1'b1 : '0);
        tx_stretch <= ~uart_txd ? '1 : (tx_stretch != 0 ? tx_stretch - 1'b1 : '0);
    end
    logic kernel_busy;
    always_ff @(posedge clk) begin
        if (rst) kernel_busy <= 1'b0;
        else if (m_awvalid && sel_w && m_awready && m_awaddr[7:0] == 8'h00 && m_wdata[0]) kernel_busy <= 1'b1;
        else if (kernel_irq || (m_rvalid && sel_r && m_araddr[7:0] == 8'h00 && m_rdata[1])) kernel_busy <= 1'b0;
    end
    assign led = {10'd0, tx_stretch != 0, rx_stretch != 0, kernel_busy, bridge_err, calib_done, locked};
endmodule
`default_nettype wire
