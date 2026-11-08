`timescale 1ns / 1ps
`default_nettype none
// Command processor for the XEM7310 FP4 accelerator (kernel clock domain).
//
// The host sends a stream of 32-bit words through FrontPanel PipeIn 0x80 (via an async FIFO,
// first-word-fall-through) and reads results from PipeOut 0xA0 (via another async FIFO).
// A whole run - prompt writes, ap_start, wait for ap_done, reads of n_out, tokens and the
// cycle counter - is one PipeIn write and one PipeOut read: no host round trips in between.
//
// Header word: [31:28] op, [27] status block, [26:16] count-1, [15:0] byte address
//   op 0 NOP    padding (FrontPanel transfers are multiples of 16 bytes)
//   op 1 WRITE  followed by count data words, written to address, address+4, ...
//   op 2 READ   pushes count words read from address, address+4, ... to the response FIFO
//   op 3 WAIT   polls ap_ctrl (offset 0) until ap_done; this also stops the cycle counter
//   other       ignored
// Status block (status bit set, read only; writes are dropped):
//   +0x0 busy-cycle counter low, +0x4 high, +0x8 kernel clock in Hz, +0xC magic.
// The counter runs from the write of ap_start (offset 0, bit 0) until WAIT sees ap_done, so
// tokens/second is measured on the FPGA (it can overcount by the few cycles of one poll).
module fp_cmd_proc #(
    parameter int CLK_HZ = 125_000_000,
    parameter int CTRL_AW = 11,
    parameter logic [31:0] MAGIC = 32'hF4F4_0002
) (
    input  wire         clk,
    input  wire         rst,
    // command FIFO (first-word fall-through)
    input  wire  [31:0] cmd_data,
    input  wire         cmd_empty,
    output logic        cmd_rd,
    // response FIFO
    output logic [31:0] resp_data,
    output logic        resp_wr,
    // AXI-Lite master to the kernel's s_axi_control
    output logic [CTRL_AW-1:0] awaddr,
    output logic        awvalid,
    input  wire         awready,
    output logic [31:0] wdata,
    output logic        wvalid,
    input  wire         wready,
    input  wire         bvalid,
    output logic        bready,
    output logic [CTRL_AW-1:0] araddr,
    output logic        arvalid,
    input  wire         arready,
    input  wire  [31:0] rdata,
    input  wire         rvalid,
    output logic        rready,
    // status
    output logic        busy,
    output logic [63:0] cycles
);
    localparam logic [3:0] OP_WRITE = 4'd1, OP_READ = 4'd2, OP_WAIT = 4'd3;

    typedef enum logic [2:0] {IDLE, WR_DATA, WR_AXI, RD_ISSUE, RD_DATA, WAIT_ISSUE, WAIT_DATA} state_t;
    state_t state;
    logic [15:0] addr;
    logic [11:0] cnt;          // words left in this command
    logic stat;
    logic aw_done, w_done;
    logic start_seen;

    wire [3:0] op = cmd_data[31:28];

    always_comb begin
        cmd_rd = 1'b0;
        if (!cmd_empty) begin
            if (state == IDLE) cmd_rd = 1'b1;
            if (state == WR_DATA) cmd_rd = 1'b1;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            awvalid <= 1'b0;
            wvalid <= 1'b0;
            bready <= 1'b0;
            arvalid <= 1'b0;
            rready <= 1'b0;
            resp_wr <= 1'b0;
            busy <= 1'b0;
            cycles <= '0;
            aw_done <= 1'b0;
            w_done <= 1'b0;
            start_seen <= 1'b0;
        end else begin
            resp_wr <= 1'b0;
            start_seen <= 1'b0;
            if (busy) cycles <= cycles + 1'b1;
            if (start_seen) begin
                busy <= 1'b1;
                cycles <= '0;
            end
            case (state)
                IDLE: if (!cmd_empty) begin
                    addr <= cmd_data[15:0];
                    cnt <= {1'b0, cmd_data[26:16]} + 12'd1;
                    stat <= cmd_data[27];
                    case (op)
                        OP_WRITE: state <= WR_DATA;
                        OP_READ:  state <= RD_ISSUE;
                        OP_WAIT:  state <= WAIT_ISSUE;
                        default:  state <= IDLE;
                    endcase
                end
                WR_DATA: if (!cmd_empty) begin
                    if (stat) begin
                        addr <= addr + 16'd4;
                        cnt <= cnt - 1'b1;
                        if (cnt == 12'd1) state <= IDLE;
                    end else begin
                        awaddr <= addr[CTRL_AW-1:0];
                        wdata <= cmd_data;
                        awvalid <= 1'b1;
                        wvalid <= 1'b1;
                        aw_done <= 1'b0;
                        w_done <= 1'b0;
                        state <= WR_AXI;
                        if (addr[CTRL_AW-1:0] == '0 && cmd_data[0]) start_seen <= 1'b1;
                    end
                end
                WR_AXI: begin
                    if (awvalid && awready) begin
                        awvalid <= 1'b0;
                        aw_done <= 1'b1;
                    end
                    if (wvalid && wready) begin
                        wvalid <= 1'b0;
                        w_done <= 1'b1;
                        bready <= 1'b1;
                    end
                    if (bready && bvalid) begin
                        bready <= 1'b0;
                        addr <= addr + 16'd4;
                        cnt <= cnt - 1'b1;
                        state <= cnt == 12'd1 ? IDLE : WR_DATA;
                    end
                end
                RD_ISSUE: begin
                    if (stat) begin
                        resp_wr <= 1'b1;
                        case (addr[3:2])
                            2'd0: resp_data <= cycles[31:0];
                            2'd1: resp_data <= cycles[63:32];
                            2'd2: resp_data <= CLK_HZ;
                            default: resp_data <= MAGIC;
                        endcase
                        addr <= addr + 16'd4;
                        cnt <= cnt - 1'b1;
                        if (cnt == 12'd1) state <= IDLE;
                    end else begin
                        araddr <= addr[CTRL_AW-1:0];
                        arvalid <= 1'b1;
                        rready <= 1'b1;
                        state <= RD_DATA;
                    end
                end
                RD_DATA: begin
                    if (arvalid && arready) arvalid <= 1'b0;
                    if (rready && rvalid) begin
                        rready <= 1'b0;
                        resp_wr <= 1'b1;
                        resp_data <= rdata;
                        addr <= addr + 16'd4;
                        cnt <= cnt - 1'b1;
                        state <= cnt == 12'd1 ? IDLE : RD_ISSUE;
                    end
                end
                WAIT_ISSUE: begin
                    araddr <= '0;
                    arvalid <= 1'b1;
                    rready <= 1'b1;
                    state <= WAIT_DATA;
                end
                WAIT_DATA: begin
                    if (arvalid && arready) arvalid <= 1'b0;
                    if (rready && rvalid) begin
                        rready <= 1'b0;
                        if (rdata[1]) begin   // ap_done (clear on read)
                            busy <= 1'b0;
                            state <= IDLE;
                        end else begin
                            state <= WAIT_ISSUE;
                        end
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
`default_nettype wire
