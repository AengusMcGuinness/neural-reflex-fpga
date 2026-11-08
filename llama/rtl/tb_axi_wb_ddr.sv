`timescale 1ns / 1ps
// Self-checking testbench for axi_wb_ddr: two independent random AXI masters (HLS-like
// bursts: reads up to 64 beats, writes up to 16 beats with random strobes) on disjoint
// address regions, against a pipelined Wishbone memory with random stall and random,
// in-order ack latency. Every read beat is checked against a shadow copy.
module tb_axi_wb_ddr;
    logic clk = 0, rst = 1;
    always #6 clk = ~clk;

    logic [0:0] awid[2], bid[2], arid[2], rid[2];
    logic [31:0] awaddr[2], wdata[2], araddr[2], rdata[2];
    logic [7:0] awlen[2], arlen[2];
    logic awvalid[2], awready[2], wlast[2], wvalid[2], wready[2], bvalid[2], bready[2];
    logic arvalid[2], arready[2], rlast[2], rvalid[2], rready[2];
    logic [3:0] wstrb[2];
    logic [1:0] bresp[2], rresp[2];
    logic wb_stb, wb_we, wb_stall, wb_ack;
    logic [23:0] wb_addr;
    logic [127:0] wb_data, wb_rdata;
    logic [15:0] wb_sel;

    axi_wb_ddr dut (.clk, .rst,
        .s_awid(awid), .s_awaddr(awaddr), .s_awlen(awlen), .s_awvalid(awvalid), .s_awready(awready),
        .s_wdata(wdata), .s_wstrb(wstrb), .s_wlast(wlast), .s_wvalid(wvalid), .s_wready(wready),
        .s_bid(bid), .s_bresp(bresp), .s_bvalid(bvalid), .s_bready(bready),
        .s_arid(arid), .s_araddr(araddr), .s_arlen(arlen), .s_arvalid(arvalid), .s_arready(arready),
        .s_rid(rid), .s_rdata(rdata), .s_rresp(rresp), .s_rlast(rlast), .s_rvalid(rvalid), .s_rready(rready),
        .wb_stb, .wb_we, .wb_addr, .wb_data, .wb_sel, .wb_stall, .wb_ack, .wb_rdata);

    // ---------------- Wishbone memory: random stall, in-order acks after 2..25 cycles -------
    logic [127:0] wbmem[int];
    typedef struct { int due; logic [127:0] data; } resp_t;
    resp_t q[$];
    int cyc = 0;
    always_ff @(posedge clk) begin
        cyc <= cyc + 1;
        wb_ack <= 1'b0;
        if (wb_stb && !wb_stall) begin
            automatic resp_t r;
            automatic logic [127:0] old = wbmem.exists(wb_addr) ? wbmem[wb_addr] : '0;
            if (wb_we) begin
                for (int b = 0; b < 16; b++) if (wb_sel[b]) old[8*b +: 8] = wb_data[8*b +: 8];
                wbmem[wb_addr] = old;
            end
            r.due = (q.size() ? (q[$].due > cyc ? q[$].due : cyc) : cyc) + 2 + ($urandom % 24);
            r.data = old;
            q.push_back(r);
        end
        if (q.size() && q[0].due <= cyc) begin
            wb_ack <= 1'b1;
            wb_rdata <= q[0].data;
            void'(q.pop_front());
        end
        wb_stall <= ($urandom % 3 == 0);
    end

    // ---------------- shadow memory (word addressed) ----------------
    logic [31:0] shadow[int];
    int errors = 0, rd_beats[2], wr_beats[2];

    task automatic master(input int p, input int n_txn);
        int base = p ? 32'h0400_0000 : 32'h0000_1000; // disjoint regions
        for (int t = 0; t < n_txn; t++) begin
            int word = $urandom % 4096;
            logic [31:0] a = base + 4 * word;
            if ($urandom % 2) begin
                int len = $urandom % 64;
                @(posedge clk); arid[p] <= p; araddr[p] <= a; arlen[p] <= len; arvalid[p] <= 1;
                do @(posedge clk); while (!arready[p]);
                arvalid[p] <= 0;
                for (int i = 0; i <= len; i++) begin
                    rready[p] <= ($urandom % 4 != 0);
                    do begin @(posedge clk); if (!rready[p]) rready[p] <= 1; end while (!(rvalid[p] && rready[p]));
                    begin
                        logic [31:0] exp = shadow.exists((a >> 2) + i) ? shadow[(a >> 2) + i] : 32'h0;
                        if (rdata[p] !== exp) begin errors++; if (errors < 10) $display("RD MISMATCH p%0d addr %h got %h exp %h", p, a + 4*i, rdata[p], exp); end
                        if (rlast[p] !== (i == len)) begin errors++; $display("RLAST wrong p%0d beat %0d/%0d", p, i, len); end
                        if (rid[p] !== p[0]) begin errors++; $display("RID wrong"); end
                    end
                    rd_beats[p]++;
                end
                rready[p] <= 0;
            end else begin
                int len = $urandom % 16;
                @(posedge clk); awid[p] <= p; awaddr[p] <= a; awlen[p] <= len; awvalid[p] <= 1;
                do @(posedge clk); while (!awready[p]);
                awvalid[p] <= 0;
                for (int i = 0; i <= len; i++) begin
                    logic [31:0] d = $urandom;
                    logic [3:0] s = ($urandom % 3 == 0) ? 4'($urandom) : 4'hF;
                    logic [31:0] old = shadow.exists((a >> 2) + i) ? shadow[(a >> 2) + i] : 32'h0;
                    wdata[p] <= d; wstrb[p] <= s; wlast[p] <= (i == len); wvalid[p] <= 1;
                    do @(posedge clk); while (!wready[p]);
                    wvalid[p] <= 0;
                    for (int b = 0; b < 4; b++) if (s[b]) old[8*b +: 8] = d[8*b +: 8];
                    shadow[(a >> 2) + i] = old;
                    wr_beats[p]++;
                end
                bready[p] <= ($urandom % 2);
                do begin @(posedge clk); bready[p] <= 1; end while (!(bvalid[p] && bready[p]));
                if (bid[p] !== p[0]) begin errors++; $display("BID wrong"); end
                bready[p] <= 0;
            end
        end
    endtask

    initial begin
        for (int p = 0; p < 2; p++) begin
            arvalid[p] = 0; awvalid[p] = 0; wvalid[p] = 0; rready[p] = 0; bready[p] = 0;
            rd_beats[p] = 0; wr_beats[p] = 0;
        end
        repeat (10) @(posedge clk);
        rst = 0;
        fork
            master(0, 400);
            master(1, 400);
        join
        repeat (50) @(posedge clk);
        if (q.size() != 0) begin errors++; $display("stray wishbone responses: %0d", q.size()); end
        $display("read beats %0d/%0d write beats %0d/%0d", rd_beats[0], rd_beats[1], wr_beats[0], wr_beats[1]);
        if (errors == 0) $display("TB_PASS axi_wb_ddr"); else $display("TB_FAIL axi_wb_ddr errors=%0d", errors);
        $finish;
    end
    initial begin #50ms; $display("TB_FAIL timeout"); $finish; end
endmodule
