`timescale 1ns / 1ps
// UART -> AXI4 master bridge. The host drives everything (DDR3 loads, HLS kernel control)
// through three commands, all little-endian:
//
//   'W' addr[4] n[4] data[4*n]  write n words at addr, addr+4, ...   reply: 'K' sum32(data)
//   'R' addr[4] n[4]            read n words                         reply: data[4*n]
//   'S' addr[4] n[4]            sum32 of n words read from memory    reply: 'S' sum32
//
// addr must be 4-byte aligned. Every AXI access is a single 32-bit beat (len 0, size 2).
// A receive FIFO absorbs AXI stalls (e.g. DDR3 refresh) because UART cannot be paused.
// Unknown command bytes are answered with '?' and ignored.
module uart_axi_bridge #(
    parameter int CLK_HZ = 83_333_333,
    parameter int BAUD = 2_000_000,
    parameter int FIFO_AW = 11 // 2 KiB receive FIFO
) (
    input  logic        clk,
    input  logic        rst, // synchronous, active high
    input  logic        uart_rx,
    output logic        uart_tx,
    // AXI4 master (write)
    output logic [31:0] m_axi_awaddr,
    output logic [7:0]  m_axi_awlen,
    output logic [2:0]  m_axi_awsize,
    output logic [1:0]  m_axi_awburst,
    output logic        m_axi_awvalid,
    input  logic        m_axi_awready,
    output logic [31:0] m_axi_wdata,
    output logic [3:0]  m_axi_wstrb,
    output logic        m_axi_wlast,
    output logic        m_axi_wvalid,
    input  logic        m_axi_wready,
    input  logic [1:0]  m_axi_bresp,
    input  logic        m_axi_bvalid,
    output logic        m_axi_bready,
    // AXI4 master (read)
    output logic [31:0] m_axi_araddr,
    output logic [7:0]  m_axi_arlen,
    output logic [2:0]  m_axi_arsize,
    output logic [1:0]  m_axi_arburst,
    output logic        m_axi_arvalid,
    input  logic        m_axi_arready,
    input  logic [31:0] m_axi_rdata,
    input  logic [1:0]  m_axi_rresp,
    input  logic        m_axi_rlast,
    input  logic        m_axi_rvalid,
    output logic        m_axi_rready,
    output logic        err_resp // sticky: an AXI SLVERR/DECERR was seen
);
    assign m_axi_awlen = 8'd0;
    assign m_axi_awsize = 3'd2;
    assign m_axi_awburst = 2'b01;
    assign m_axi_wstrb = 4'hF;
    assign m_axi_wlast = 1'b1;
    assign m_axi_arlen = 8'd0;
    assign m_axi_arsize = 3'd2;
    assign m_axi_arburst = 2'b01;

    // ---------------- UART receive -> FIFO ----------------
    logic [7:0] rx_byte;
    logic rx_valid;
    uart_rx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_rx (.clk, .rst, .rx(uart_rx), .data(rx_byte), .valid(rx_valid));

    logic [7:0] fifo_mem[2**FIFO_AW];
    logic [FIFO_AW:0] wr_ptr, rd_ptr;
    logic [7:0] fifo_q;
    wire fifo_empty = (wr_ptr == rd_ptr);
    logic fifo_pop;
    logic overflow; // sticky; reported as '!' instead of 'K'
    always_ff @(posedge clk) begin
        if (rst) begin
            wr_ptr <= '0;
            overflow <= 1'b0;
        end else if (rx_valid) begin
            if ((wr_ptr - rd_ptr) == (FIFO_AW + 1)'(2**FIFO_AW)) overflow <= 1'b1;
            else begin
                fifo_mem[wr_ptr[FIFO_AW-1:0]] <= rx_byte;
                wr_ptr <= wr_ptr + 1'b1;
            end
        end
    end
    assign fifo_q = fifo_mem[rd_ptr[FIFO_AW-1:0]];
    always_ff @(posedge clk) begin
        if (rst) rd_ptr <= '0;
        else if (fifo_pop) rd_ptr <= rd_ptr + 1'b1;
    end

    // ---------------- UART transmit ----------------
    logic [7:0] tx_byte;
    logic tx_start, tx_busy;
    uart_tx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_tx (.clk, .rst, .data(tx_byte), .start(tx_start), .busy(tx_busy), .tx(uart_tx));

    // ---------------- command engine ----------------
    typedef enum logic [3:0] {
        S_CMD, S_HDR, S_WDATA, S_AXI_W, S_AXI_B, S_AXI_AR, S_AXI_R, S_TX_WORD, S_TX_WAIT, S_NEXT
    } state_t;
    state_t st;
    logic [7:0] cmd;
    logic [63:0] hdr;      // addr (low) + n (high), shifted in LSB first
    logic [3:0] cnt;       // byte counter within header / word
    logic [31:0] addr, nleft, word, sum;
    logic [39:0] txsh;     // up to 5 bytes to send, LSB first
    logic [2:0] txn;       // bytes left in txsh

    always_ff @(posedge clk) begin
        if (rst) begin
            st <= S_CMD;
            m_axi_awvalid <= 1'b0;
            m_axi_wvalid <= 1'b0;
            m_axi_bready <= 1'b0;
            m_axi_arvalid <= 1'b0;
            m_axi_rready <= 1'b0;
            tx_start <= 1'b0;
            err_resp <= 1'b0;
            cnt <= '0;
        end else begin
            tx_start <= 1'b0;
            case (st)
                S_CMD: if (!fifo_empty) begin
                    cmd <= fifo_q;
                    if (fifo_q == "W" || fifo_q == "R" || fifo_q == "S") begin
                        cnt <= '0;
                        st <= S_HDR;
                    end else begin
                        txsh <= {32'd0, 8'h3F}; // '?'
                        txn <= 3'd1;
                        st <= S_TX_WAIT;
                    end
                end
                S_HDR: if (!fifo_empty) begin
                    hdr <= {fifo_q, hdr[63:8]};
                    cnt <= cnt + 1'b1;
                    if (cnt == 4'd7) begin
                        addr <= hdr[39:8];               // bytes 0..3 (byte 7 arriving now)
                        nleft <= {fifo_q, hdr[63:40]};    // bytes 4..7
                        sum <= '0;
                        cnt <= '0;
                        st <= S_NEXT;
                    end
                end
                S_NEXT: begin
                    if (nleft == 0) begin
                        if (cmd == "W") begin
                            txsh <= {sum, overflow ? 8'h21 : 8'h4B}; // 'K' (or '!' on overflow)
                            txn <= 3'd5;
                            st <= S_TX_WAIT;
                        end else if (cmd == "S") begin
                            txsh <= {sum, 8'h53}; // 'S'
                            txn <= 3'd5;
                            st <= S_TX_WAIT;
                        end else st <= S_CMD;
                    end else if (cmd == "W") st <= S_WDATA;
                    else begin
                        m_axi_araddr <= addr;
                        m_axi_arvalid <= 1'b1;
                        st <= S_AXI_AR;
                    end
                end
                S_WDATA: if (!fifo_empty) begin
                    word <= {fifo_q, word[31:8]};
                    cnt <= cnt + 1'b1;
                    if (cnt == 4'd3) begin
                        cnt <= '0;
                        m_axi_awaddr <= addr;
                        m_axi_awvalid <= 1'b1;
                        m_axi_wdata <= {fifo_q, word[31:8]};
                        m_axi_wvalid <= 1'b1;
                        sum <= sum + {fifo_q, word[31:8]};
                        st <= S_AXI_W;
                    end
                end
                S_AXI_W: begin
                    if (m_axi_awready) m_axi_awvalid <= 1'b0;
                    if (m_axi_wready) m_axi_wvalid <= 1'b0;
                    if ((!m_axi_awvalid || m_axi_awready) && (!m_axi_wvalid || m_axi_wready)) begin
                        m_axi_bready <= 1'b1;
                        st <= S_AXI_B;
                    end
                end
                S_AXI_B: if (m_axi_bvalid) begin
                    m_axi_bready <= 1'b0;
                    if (m_axi_bresp[1]) err_resp <= 1'b1;
                    addr <= addr + 32'd4;
                    nleft <= nleft - 1'b1;
                    st <= S_NEXT;
                end
                S_AXI_AR: if (m_axi_arready) begin
                    m_axi_arvalid <= 1'b0;
                    m_axi_rready <= 1'b1;
                    st <= S_AXI_R;
                end
                S_AXI_R: if (m_axi_rvalid) begin
                    m_axi_rready <= 1'b0;
                    if (m_axi_rresp[1]) err_resp <= 1'b1;
                    addr <= addr + 32'd4;
                    nleft <= nleft - 1'b1;
                    if (cmd == "R") begin
                        txsh <= {8'd0, m_axi_rdata};
                        txn <= 3'd4;
                        st <= S_TX_WAIT;
                    end else begin
                        sum <= sum + m_axi_rdata;
                        st <= S_NEXT;
                    end
                end
                S_TX_WAIT: if (!tx_busy && !tx_start) begin
                    if (txn == 0) st <= (cmd == "R" || cmd == "W" || cmd == "S") && nleft != 0 ? S_NEXT : S_CMD;
                    else begin
                        tx_byte <= txsh[7:0];
                        tx_start <= 1'b1;
                        txsh <= {8'd0, txsh[39:8]};
                        txn <= txn - 1'b1;
                    end
                end
                default: st <= S_CMD;
            endcase
        end
    end
    // pop exactly when a byte is consumed
    assign fifo_pop = !fifo_empty && ((st == S_CMD) || (st == S_HDR) || (st == S_WDATA));
endmodule

// 8N1 receiver, 16x oversampling from a fractional tick generator.
module uart_rx #(parameter int CLK_HZ = 83_333_333, parameter int BAUD = 2_000_000) (
    input  logic clk, rst, rx,
    output logic [7:0] data,
    output logic valid
);
    localparam longint INC = 16 * longint'(BAUD);
    logic [31:0] acc;
    logic tick;
    always_ff @(posedge clk) begin
        if (rst) begin acc <= '0; tick <= 1'b0; end
        else if (acc + INC >= CLK_HZ) begin acc <= acc + INC - CLK_HZ; tick <= 1'b1; end
        else begin acc <= acc + INC; tick <= 1'b0; end
    end
    logic [2:0] sync;
    always_ff @(posedge clk) sync <= {sync[1:0], rx};
    wire rxs = sync[2];
    logic busy;
    logic [3:0] os;   // oversample counter
    logic [3:0] bitn; // 0 = start, 1..8 data, 9 stop
    logic [7:0] sh;
    always_ff @(posedge clk) begin
        valid <= 1'b0;
        if (rst) busy <= 1'b0;
        else if (tick) begin
            if (!busy) begin
                if (!rxs) begin busy <= 1'b1; os <= 4'd0; bitn <= 4'd0; end
            end else begin
                os <= os + 1'b1;
                if (os == 4'd7) begin // middle of the bit
                    if (bitn == 4'd0) begin
                        if (rxs) busy <= 1'b0; // false start
                    end else if (bitn <= 4'd8) sh <= {rxs, sh[7:1]};
                    else begin
                        busy <= 1'b0;
                        if (rxs) begin data <= sh; valid <= 1'b1; end // drop framing errors
                    end
                    bitn <= bitn + 1'b1;
                end
            end
        end
    end
endmodule

// 8N1 transmitter with a fractional bit-period generator.
module uart_tx #(parameter int CLK_HZ = 83_333_333, parameter int BAUD = 2_000_000) (
    input  logic clk, rst,
    input  logic [7:0] data,
    input  logic start,
    output logic busy,
    output logic tx
);
    logic [31:0] acc;
    logic tick;
    always_ff @(posedge clk) begin
        if (rst || !busy) begin acc <= '0; tick <= 1'b0; end
        else if (acc + BAUD >= CLK_HZ) begin acc <= acc + BAUD - CLK_HZ; tick <= 1'b1; end
        else begin acc <= acc + BAUD; tick <= 1'b0; end
    end
    logic [9:0] sh;
    logic [3:0] n;
    always_ff @(posedge clk) begin
        if (rst) begin busy <= 1'b0; tx <= 1'b1; end
        else if (!busy) begin
            tx <= 1'b1;
            if (start) begin sh <= {1'b1, data, 1'b0}; n <= 4'd0; busy <= 1'b1; tx <= 1'b0; end
        end else if (tick) begin
            n <= n + 1'b1;
            if (n == 4'd9) begin busy <= 1'b0; tx <= 1'b1; end
            else begin tx <= sh[n + 1]; end
        end
    end
endmodule
