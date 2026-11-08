`timescale 1ns / 1ps
// Two AXI4 slave ports (32-bit data, INCR bursts, size = 4 bytes) -> one pipelined
// Wishbone master for the UberDDR3 controller (ddr3_top: 128-bit lines, i_wb_cyc tied 1).
//
// One AXI transaction is served at a time (round-robin between ports, reads before writes
// within a port). Each transaction completes fully before the next one starts, so every
// Wishbone ack belongs unambiguously to the transaction in flight.
//  * read burst : all lines it touches are requested back to back; returned lines land in a
//                 32-deep FIFO (>= 17 lines, the most a 64-beat burst can touch, so acks can
//                 never overflow it) and are emitted as 32-bit beats.
//  * write beat : one 128-bit Wishbone write per beat with the 4 byte selects of its lane,
//                 waiting for the ack before taking the next beat.
// Byte address A maps to line A[27:4] (256 MB) and 32-bit lane A[3:2].
module axi_wb_ddr #(
    parameter int ID_W = 1,
    parameter int WB_AW = 24
) (
    input  logic clk,
    input  logic rst,
    // ---- AXI slave ports [0] and [1] ----
    input  logic [ID_W-1:0] s_awid[2],
    input  logic [31:0]     s_awaddr[2],
    input  logic [7:0]      s_awlen[2],
    input  logic            s_awvalid[2],
    output logic            s_awready[2],
    input  logic [31:0]     s_wdata[2],
    input  logic [3:0]      s_wstrb[2],
    input  logic            s_wlast[2],
    input  logic            s_wvalid[2],
    output logic            s_wready[2],
    output logic [ID_W-1:0] s_bid[2],
    output logic [1:0]      s_bresp[2],
    output logic            s_bvalid[2],
    input  logic            s_bready[2],
    input  logic [ID_W-1:0] s_arid[2],
    input  logic [31:0]     s_araddr[2],
    input  logic [7:0]      s_arlen[2],
    input  logic            s_arvalid[2],
    output logic            s_arready[2],
    output logic [ID_W-1:0] s_rid[2],
    output logic [31:0]     s_rdata[2],
    output logic [1:0]      s_rresp[2],
    output logic            s_rlast[2],
    output logic            s_rvalid[2],
    input  logic            s_rready[2],
    // ---- Wishbone master (pipelined, cyc tied high outside) ----
    output logic             wb_stb,
    output logic             wb_we,
    output logic [WB_AW-1:0] wb_addr,
    output logic [127:0]     wb_data,
    output logic [15:0]      wb_sel,
    input  logic             wb_stall,
    input  logic             wb_ack,
    input  logic [127:0]     wb_rdata
);
    typedef enum logic [2:0] {IDLE, RD, WR_BEAT, WR_REQ, WR_ACK, WR_RESP} st_t;
    st_t st;
    logic port;            // port being served
    logic rr;              // round-robin: port that gets first look next time
    logic [ID_W-1:0] id;
    logic [31:0] addr;     // byte address of the current beat
    logic [8:0] beats_left;

    // read bookkeeping
    logic [WB_AW-1:0] line_next;  // next line to request
    logic [5:0] lines_to_req;     // lines not yet requested
    logic [1:0] lane;             // lane of the next beat in the FIFO head line

    // ---- line FIFO (32 x 128) ----
    logic [127:0] fifo[32];
    logic [5:0] f_wr, f_rd;
    wire f_empty = (f_wr == f_rd);
    logic f_pop;
    always_ff @(posedge clk) begin
        if (rst) f_wr <= '0;
        else if (wb_ack && st == RD) begin
            fifo[f_wr[4:0]] <= wb_rdata;
            f_wr <= f_wr + 1'b1;
        end
    end
    always_ff @(posedge clk) begin
        if (rst) f_rd <= '0;
        else if (f_pop) f_rd <= f_rd + 1'b1;
    end
    wire [127:0] head = fifo[f_rd[4:0]];

    // ---- outputs per port ----
    logic g_rd, g_wr, g_port;
    wire rbeat = (st == RD) && !f_empty;
    wire last_beat = (beats_left == 9'd1);
    always_comb begin
        for (int p = 0; p < 2; p++) begin
            s_awready[p] = 1'b0;
            s_arready[p] = 1'b0;
            s_wready[p] = (st == WR_BEAT) && (port == p);
            s_bvalid[p] = (st == WR_RESP) && (port == p);
            s_bid[p] = id;
            s_bresp[p] = 2'b00;
            s_rvalid[p] = rbeat && (port == p);
            s_rid[p] = id;
            s_rdata[p] = head[32*lane +: 32];
            s_rresp[p] = 2'b00;
            s_rlast[p] = last_beat;
        end
        // arbitration (IDLE only): look at rr first, then the other port
        g_rd = 1'b0;
        g_wr = 1'b0;
        g_port = rr;
        if (st == IDLE) begin
            if (s_arvalid[rr]) begin g_rd = 1'b1; g_port = rr; end
            else if (s_awvalid[rr]) begin g_wr = 1'b1; g_port = rr; end
            else if (s_arvalid[!rr]) begin g_rd = 1'b1; g_port = !rr; end
            else if (s_awvalid[!rr]) begin g_wr = 1'b1; g_port = !rr; end
        end
        if (g_rd) s_arready[g_port] = 1'b1;
        if (g_wr) s_awready[g_port] = 1'b1;
    end
    // granted request, as seen by the sequential logic
    wire [ID_W-1:0] g_arid = s_arid[g_port];
    wire [31:0] g_araddr = s_araddr[g_port];
    wire [7:0] g_arlen = s_arlen[g_port];
    wire [ID_W-1:0] g_awid = s_awid[g_port];
    wire [31:0] g_awaddr = s_awaddr[g_port];
    wire rhand = rbeat && s_rready[port];
    assign f_pop = rhand && (lane == 2'd3 || last_beat);

    logic [31:0] wd;
    logic [3:0] ws;
    logic wl;

    always_ff @(posedge clk) begin
        if (rst) begin
            st <= IDLE;
            rr <= 1'b0;
            wb_stb <= 1'b0;
            wb_we <= 1'b0;
        end else begin
            case (st)
                IDLE: begin
                    if (g_rd) begin
                        port <= g_port;
                        id <= g_arid;
                        addr <= g_araddr;
                        beats_left <= {1'b0, g_arlen} + 9'd1;
                        line_next <= g_araddr[WB_AW+3:4];
                        // lines touched = last_line - first_line + 1
                        lines_to_req <= 6'(((g_araddr[WB_AW+3:0] + {g_arlen, 2'b00}) >> 4)
                                            - (g_araddr[WB_AW+3:0] >> 4) + 1);
                        lane <= g_araddr[3:2];
                        rr <= !g_port;
                        st <= RD;
                    end else if (g_wr) begin
                        port <= g_port;
                        id <= g_awid;
                        addr <= g_awaddr;
                        rr <= !g_port;
                        st <= WR_BEAT;
                    end
                end
                RD: begin
                    // request side
                    if (wb_stb && !wb_stall) begin
                        line_next <= line_next + 1'b1;
                        lines_to_req <= lines_to_req - 1'b1;
                        if (lines_to_req == 6'd1) wb_stb <= 1'b0;
                    end else if (!wb_stb && lines_to_req != 0) begin
                        wb_stb <= 1'b1;
                        wb_we <= 1'b0;
                    end
                    // beat side
                    if (rhand) begin
                        lane <= lane + 1'b1;
                        beats_left <= beats_left - 1'b1;
                        if (last_beat) st <= IDLE;
                    end
                end
                WR_BEAT: if (s_wvalid[port]) begin
                    wd <= s_wdata[port];
                    ws <= s_wstrb[port];
                    wl <= s_wlast[port];
                    st <= WR_REQ;
                end
                WR_REQ: begin
                    if (!wb_stb) begin
                        wb_stb <= 1'b1;
                        wb_we <= 1'b1;
                    end else if (!wb_stall) begin
                        wb_stb <= 1'b0;
                        if (wb_ack) begin // zero-latency ack: this write is already done
                            addr <= addr + 32'd4;
                            st <= wl ? WR_RESP : WR_BEAT;
                        end else st <= WR_ACK;
                    end
                end
                WR_ACK: if (wb_ack) begin
                    addr <= addr + 32'd4;
                    st <= wl ? WR_RESP : WR_BEAT;
                end
                WR_RESP: if (s_bready[port]) st <= IDLE;
                default: st <= IDLE;
            endcase
        end
    end

    // Wishbone address/data: read requests walk line_next; writes use the beat's lane.
    always_comb begin
        wb_addr = (st == RD) ? line_next : addr[WB_AW+3:4];
        wb_data = {4{wd}};
        wb_sel = (st == RD) ? 16'h0000 : (16'(ws) << (4 * addr[3:2]));
    end
endmodule
