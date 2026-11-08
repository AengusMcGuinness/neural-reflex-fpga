`timescale 1ns / 1ps
`default_nettype none
// Testbench for fp_cmd_proc: a command stream (with NOP padding, burst writes, status reads,
// WAIT and burst reads) against a model of the HLS s_axi_control port. The model has a
// 2 KB register file, takes AW before W like the HLS adapter, and on an ap_start write
// raises ap_done (offset 0 bit 1, clear on read) DONE_AFTER cycles later, writing
// tokens[i] = prompt[i] + 1000 at 0x400 + 4i as the "result".
// Run: xvlog -sv fp_cmd_proc.sv tb_fp_cmd_proc.sv && xelab tb_fp_cmd_proc -s tb && xsim tb -R
module tb_fp_cmd_proc;
    localparam int AW = 11, DONE_AFTER = 300;
    logic clk = 0, rst = 1;
    always #4 clk = ~clk;

    // command FIFO model (FWFT)
    // (an array with pointers: continuous assigns do not track a queue's size in xsim)
    logic [31:0] cmdm[0:255];
    int cwp = 0, crp = 0;
    wire cmd_empty = cwp == crp;
    wire [31:0] cmd_data = cmdm[crp[7:0]];
    task automatic push(logic [31:0] w);
        cmdm[cwp[7:0]] = w;
        cwp = cwp + 1;
    endtask
    logic cmd_rd;
    logic [31:0] resp_data;
    logic resp_wr;
    logic [31:0] resp[$];

    logic [AW-1:0] awaddr, araddr;
    logic awvalid, awready, wvalid, wready, bvalid, bready, arvalid, arready, rvalid, rready;
    logic [31:0] wdata, rdata;
    logic busy;
    logic [63:0] cycles;

    fp_cmd_proc #(.CLK_HZ(125_000_000), .CTRL_AW(AW)) dut (.*);

    always @(posedge clk) if (resp_wr) resp.push_back(resp_data);
    always @(posedge clk) if (cmd_rd) #1 crp = crp + 1;

    // ---------------- s_axi_control model ----------------
    logic [31:0] regs[0:511];
    logic [AW-1:0] wa;
    logic wstate;  // 0 = waiting for AW, 1 = waiting for W
    int done_cnt = -1;
    logic ap_done = 0;
    initial for (int i = 0; i < 512; i++) regs[i] = 0;
    assign awready = !wstate && !bvalid;
    assign wready = wstate;
    always @(posedge clk) begin
        if (rst) begin
            wstate <= 0;
            bvalid <= 0;
            arready <= 1;
            rvalid <= 0;
        end else begin
            if (awvalid && awready) begin
                wa <= awaddr;
                wstate <= 1;
            end
            if (wvalid && wready) begin
                wstate <= 0;
                bvalid <= 1;
                regs[wa >> 2] <= wdata;
                if (wa == 0 && wdata[0]) done_cnt <= DONE_AFTER;
            end
            if (bvalid && bready) bvalid <= 0;
            if (arvalid && arready) begin
                arready <= 0;
                rvalid <= 1;
                rdata <= araddr == 0 ? {30'd0, ap_done, 1'b0} : regs[araddr >> 2];
                if (araddr == 0) ap_done <= 0;
            end
            if (rvalid && rready) begin
                rvalid <= 0;
                arready <= 1;
            end
            if (done_cnt > 0) done_cnt <= done_cnt - 1;
            if (done_cnt == 0) begin
                done_cnt <= -1;
                ap_done <= 1;
                for (int i = 0; i < 64; i++) regs[(12'h400 >> 2) + i] <= regs[(12'h100 >> 2) + i] + 1000;
                regs[12'h20 >> 2] <= 7;
            end
        end
    end

    function automatic logic [31:0] hdr(int op, int stat, int count, int addr);
        return (op << 28) | (stat << 27) | ((count - 1) << 16) | addr;
    endfunction

    int errors = 0;
    initial begin
        repeat (5) @(posedge clk);
        rst = 0;
        // write 5 prompt words with one burst, n_prompt, start; wait; read n_out, 5 tokens,
        // the cycle counter and the magic
        push(hdr(0, 0, 1, 0));
        push(hdr(1, 0, 5, 12'h100));
        for (int i = 0; i < 5; i++) push(10 + i);
        push(hdr(1, 0, 1, 12'h10));
        push(5);
        push(hdr(1, 1, 1, 0));      // write to the status block: dropped
        push(32'hDEAD_BEEF);
        push(hdr(1, 0, 1, 0));      // ap_start
        push(1);
        push(hdr(3, 0, 1, 0));      // WAIT
        push(hdr(2, 0, 1, 12'h20)); // n_out
        push(hdr(2, 0, 5, 12'h400));
        push(hdr(2, 1, 4, 0));      // cycles lo, hi, Hz, magic
        push(hdr(0, 0, 1, 0));
        while (resp.size() < 10) @(posedge clk);
        repeat (20) @(posedge clk);
        if (resp.size() != 10) begin $display("FAIL: %0d responses", resp.size()); errors++; end
        if (resp[0] !== 7) begin $display("FAIL: n_out %0d", resp[0]); errors++; end
        for (int i = 0; i < 5; i++)
            if (resp[1 + i] !== 1010 + i) begin $display("FAIL: token %0d = %0d", i, resp[1 + i]); errors++; end
        if (resp[6] < DONE_AFTER || resp[6] > DONE_AFTER + 20) begin
            $display("FAIL: cycles %0d (want about %0d)", resp[6], DONE_AFTER); errors++;
        end
        if (resp[7] !== 0) begin $display("FAIL: cycles hi"); errors++; end
        if (resp[8] !== 125_000_000) begin $display("FAIL: hz %0d", resp[8]); errors++; end
        if (resp[9] !== 32'hF4F4_0002) begin $display("FAIL: magic %h", resp[9]); errors++; end
        if (regs[0] !== 1 || regs[12'h10 >> 2] !== 5) begin $display("FAIL: register writes"); errors++; end
        if (busy) begin $display("FAIL: still busy"); errors++; end
        $display("cycles=%0d", resp[6]);
        // second run back to back (ap_done was cleared by WAIT's read)
        resp.delete();
        push(hdr(1, 0, 1, 12'h100));
        push(42);
        push(hdr(1, 0, 1, 0));
        push(1);
        push(hdr(3, 0, 1, 0));
        push(hdr(2, 0, 1, 12'h400));
        while (resp.size() < 1) @(posedge clk);
        if (resp[0] !== 1042) begin $display("FAIL: second run token %0d", resp[0]); errors++; end
        if (errors) $display("TB_CMD_PROC FAIL (%0d errors)", errors);
        else $display("TB_CMD_PROC PASS");
        $finish;
    end
    initial begin
        #200000;
        $display("TB_CMD_PROC FAIL timeout");
        $finish;
    end
endmodule
`default_nettype wire
