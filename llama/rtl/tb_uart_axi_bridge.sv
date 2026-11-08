// Self-checking testbench: a UART host model talks to uart_axi_bridge, which talks to an
// AXI memory model with random ready/valid stalls. Run: xvlog -sv ...; xelab; xsim -R
`timescale 1ns / 1ps
module tb_uart_axi_bridge;
    localparam int CLK_HZ = 83_333_333;
    localparam int BAUD = 2_000_000;
    localparam real BIT_NS = 1.0e9 / BAUD;

    logic clk = 0, rst = 1;
    always #6 clk = ~clk; // 83.33 MHz
    logic rx = 1'b1; // host -> fpga
    wire tx;         // fpga -> host

    logic [31:0] awaddr, wdata, araddr, rdata;
    logic [7:0] awlen, arlen;
    logic [2:0] awsize, arsize;
    logic [1:0] awburst, arburst, bresp, rresp;
    logic [3:0] wstrb;
    logic awvalid, awready, wvalid, wready, wlast, bvalid, bready;
    logic arvalid, arready, rvalid, rready, rlast;
    logic err;

    uart_axi_bridge #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) dut (
        .clk, .rst, .uart_rx(rx), .uart_tx(tx),
        .m_axi_awaddr(awaddr), .m_axi_awlen(awlen), .m_axi_awsize(awsize), .m_axi_awburst(awburst),
        .m_axi_awvalid(awvalid), .m_axi_awready(awready), .m_axi_wdata(wdata), .m_axi_wstrb(wstrb),
        .m_axi_wlast(wlast), .m_axi_wvalid(wvalid), .m_axi_wready(wready), .m_axi_bresp(bresp),
        .m_axi_bvalid(bvalid), .m_axi_bready(bready), .m_axi_araddr(araddr), .m_axi_arlen(arlen),
        .m_axi_arsize(arsize), .m_axi_arburst(arburst), .m_axi_arvalid(arvalid), .m_axi_arready(arready),
        .m_axi_rdata(rdata), .m_axi_rresp(rresp), .m_axi_rlast(rlast), .m_axi_rvalid(rvalid),
        .m_axi_rready(rready), .err_resp(err));

    // ---------------- AXI memory model (word addressed, random stalls) ----------------
    logic [31:0] mem[int];
    logic [31:0] aw_q;
    logic aw_have, w_have;
    logic [31:0] w_q;
    always_ff @(posedge clk) begin
        if (rst) begin
            awready <= 0; wready <= 0; bvalid <= 0; aw_have <= 0; w_have <= 0;
            arready <= 0; rvalid <= 0;
        end else begin
            awready <= !aw_have && ($urandom % 4 != 0);
            wready <= !w_have && ($urandom % 3 != 0);
            if (awvalid && awready) begin aw_q <= awaddr; aw_have <= 1; awready <= 0; end
            if (wvalid && wready) begin w_q <= wdata; w_have <= 1; wready <= 0; end
            if (aw_have && w_have && !bvalid && ($urandom % 5 != 0)) begin
                mem[aw_q >> 2] = w_q;
                bvalid <= 1; bresp <= 2'b00;
                aw_have <= 0; w_have <= 0;
            end
            if (bvalid && bready) bvalid <= 0;
            arready <= !rvalid && ($urandom % 4 != 0);
            if (arvalid && arready) begin
                rdata <= mem.exists(araddr >> 2) ? mem[araddr >> 2] : 32'hDEADBEEF;
                rvalid <= 1; rresp <= 2'b00; rlast <= 1; arready <= 0;
            end
            if (rvalid && rready) rvalid <= 0;
        end
    end

    // ---------------- UART host model ----------------
    task automatic send_byte(input logic [7:0] b);
        rx = 0; #(BIT_NS);
        for (int i = 0; i < 8; i++) begin rx = b[i]; #(BIT_NS); end
        rx = 1; #(BIT_NS);
    endtask
    task automatic send_word(input logic [31:0] w);
        for (int i = 0; i < 4; i++) send_byte(w[8*i +: 8]);
    endtask
    logic [7:0] rxq[$];
    initial begin : uart_monitor
        logic [7:0] b;
        forever begin
            @(negedge tx);
            #(BIT_NS * 1.5);
            for (int i = 0; i < 8; i++) begin b[i] = tx; #(BIT_NS); end
            if (tx !== 1'b1) $error("framing error on tx");
            rxq.push_back(b);
        end
    end
    task automatic get_byte(output logic [7:0] b);
        int guard = 0;
        while (rxq.size() == 0) begin #(BIT_NS); guard++; if (guard > 200000) $fatal(1, "timeout"); end
        b = rxq.pop_front();
    endtask
    task automatic get_word(output logic [31:0] w);
        logic [7:0] b;
        for (int i = 0; i < 4; i++) begin get_byte(b); w[8*i +: 8] = b; end
    endtask

    int errors = 0;
    task automatic check(input string what, input logic [31:0] got, exp);
        if (got !== exp) begin errors++; $display("FAIL %s: got %h exp %h", what, got, exp); end
    endtask

    initial begin
        logic [7:0] b;
        logic [31:0] w, sum;
        logic [31:0] data[64];
        repeat (20) @(posedge clk);
        rst = 0;
        repeat (20) @(posedge clk);

        // W 64 words back to back at full UART rate
        sum = 0;
        for (int i = 0; i < 64; i++) begin data[i] = $urandom; sum += data[i]; end
        send_byte("W"); send_word(32'h0000_1000); send_word(64);
        for (int i = 0; i < 64; i++) send_word(data[i]);
        get_byte(b); check("W ack", b, "K");
        get_word(w); check("W sum", w, sum);
        for (int i = 0; i < 64; i++) check($sformatf("mem[%0d]", i), mem[(32'h1000 >> 2) + i], data[i]);

        // R 5 words starting at word 3
        send_byte("R"); send_word(32'h0000_1000 + 12); send_word(5);
        for (int i = 0; i < 5; i++) begin get_word(w); check($sformatf("R[%0d]", i), w, data[3 + i]); end

        // S over all 64 words
        send_byte("S"); send_word(32'h0000_1000); send_word(64);
        get_byte(b); check("S ack", b, "S");
        get_word(w); check("S sum", w, sum);

        // unknown command, then a zero-length write still acks
        send_byte("x");
        get_byte(b); check("unknown", b, "?");
        send_byte("W"); send_word(32'h2000); send_word(0);
        get_byte(b); check("W0 ack", b, "K");
        get_word(w); check("W0 sum", w, 0);

        check("err_resp", err, 0);
        if (rxq.size() != 0) begin errors++; $display("FAIL: %0d stray bytes", rxq.size()); end
        if (errors == 0) $display("TB_PASS uart_axi_bridge");
        else $display("TB_FAIL uart_axi_bridge errors=%0d", errors);
        $finish;
    end
endmodule
