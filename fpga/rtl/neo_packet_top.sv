`default_nettype none
`timescale 1ns / 1ps

// XEM7310-A75 FrontPanel packet bring-up. The PipeIn transports exactly the
// little-endian 68-word packet defined in neural_reflex/fpga_packet.py.
module neo_packet_top (
    input  wire [4:0]  okUH,
    output wire [2:0]  okHU,
    inout  wire [31:0] okUHU,
    inout  wire        okAA
);
    localparam [31:0] MAGIC = 32'h4e524658;
    wire okClk;
    wire [112:0] okHE;
    wire [64:0] okEH;
    wire [9*65-1:0] okEHx;
    wire [31:0] control;
    wire pipe_write;
    wire [31:0] pipe_data;
    wire reset = control[0];

    okHost host (
        .okUH(okUH), .okHU(okHU), .okUHU(okUHU), .okAA(okAA),
        .okClk(okClk), .okHE(okHE), .okEH(okEH)
    );
    okWireOR #(.N(9)) wire_or (.okEH(okEH), .okEHx(okEHx));
    okWireIn ep00 (.okHE(okHE), .ep_addr(8'h00), .ep_dataout(control));
    okPipeIn ep80 (
        .okHE(okHE), .okEH(okEHx[8*65 +: 65]),
        .ep_addr(8'h80), .ep_write(pipe_write), .ep_dataout(pipe_data)
    );

    // A small elastic FIFO keeps endpoint timing independent of parsing.
    // The parser consumes one word per clock, matching the PipeIn peak rate.
    reg [31:0] fifo [0:127];
    reg [6:0] write_ptr = 0;
    reg [6:0] read_ptr = 0;
    reg [7:0] fifo_count = 0;
    wire pop = fifo_count != 0;
    wire push = pipe_write && (fifo_count != 8'd128 || pop);
    wire [31:0] word_in = fifo[read_ptr];

    reg [6:0] word_index = 0;
    reg [31:0] total_words = 0;
    reg [31:0] good_packets = 0;
    reg [31:0] last_magic = 0;
    reg [31:0] last_sequence = 0;
    reg [31:0] last_checksum = 0;
    reg [31:0] last_timestamp_lo = 0;
    reg [31:0] last_timestamp_hi = 0;
    reg [31:0] checksum = 0;
    reg [31:0] expected_sequence = 0;
    reg sequence_known = 0;
    reg magic_ok = 0;
    reg sequence_ok = 0;
    reg [2:0] errors = 0; // bit 0 magic, bit 1 sequence, bit 2 FIFO overflow

    always @(posedge okClk) begin
        if (reset) begin
            write_ptr <= 0;
            read_ptr <= 0;
            fifo_count <= 0;
            word_index <= 0;
            total_words <= 0;
            good_packets <= 0;
            last_magic <= 0;
            last_sequence <= 0;
            last_checksum <= 0;
            last_timestamp_lo <= 0;
            last_timestamp_hi <= 0;
            checksum <= 0;
            expected_sequence <= 0;
            sequence_known <= 0;
            magic_ok <= 0;
            sequence_ok <= 0;
            errors <= 0;
        end else begin
            if (push) begin
                fifo[write_ptr] <= pipe_data;
                write_ptr <= write_ptr + 1'b1;
            end
            if (pop)
                read_ptr <= read_ptr + 1'b1;
            case ({push, pop})
                2'b10: fifo_count <= fifo_count + 1'b1;
                2'b01: fifo_count <= fifo_count - 1'b1;
                default: ;
            endcase
            if (pipe_write && !push)
                errors[2] <= 1'b1;

            if (pop) begin
                total_words <= total_words + 1'b1;
                if (word_index == 0) begin
                    last_magic <= word_in;
                    magic_ok <= word_in == MAGIC;
                    checksum <= word_in;
                end else begin
                    checksum <= checksum ^ word_in;
                end
                if (word_index == 1) begin
                    last_sequence <= word_in;
                    sequence_ok <= !sequence_known || word_in == expected_sequence;
                    expected_sequence <= word_in + 1'b1;
                    sequence_known <= 1'b1;
                end
                if (word_index == 2)
                    last_timestamp_lo <= word_in;
                if (word_index == 3)
                    last_timestamp_hi <= word_in;
                if (word_index == 67) begin
                    word_index <= 0;
                    last_checksum <= checksum ^ word_in;
                    if (magic_ok && sequence_ok)
                        good_packets <= good_packets + 1'b1;
                    if (!magic_ok)
                        errors[0] <= 1'b1;
                    if (!sequence_ok)
                        errors[1] <= 1'b1;
                end else begin
                    word_index <= word_index + 1'b1;
                end
            end
        end
    end

    // UpdateWireOuts snapshots these registers after each host write.
    okWireOut ep20 (.okHE(okHE), .okEH(okEHx[0*65 +: 65]), .ep_addr(8'h20), .ep_datain(total_words));
    okWireOut ep21 (.okHE(okHE), .okEH(okEHx[1*65 +: 65]), .ep_addr(8'h21), .ep_datain(good_packets));
    okWireOut ep22 (.okHE(okHE), .okEH(okEHx[2*65 +: 65]), .ep_addr(8'h22), .ep_datain(last_magic));
    okWireOut ep23 (.okHE(okHE), .okEH(okEHx[3*65 +: 65]), .ep_addr(8'h23), .ep_datain(last_sequence));
    okWireOut ep24 (.okHE(okHE), .okEH(okEHx[4*65 +: 65]), .ep_addr(8'h24), .ep_datain(last_checksum));
    okWireOut ep25 (.okHE(okHE), .okEH(okEHx[5*65 +: 65]), .ep_addr(8'h25), .ep_datain(last_timestamp_lo));
    okWireOut ep26 (.okHE(okHE), .okEH(okEHx[6*65 +: 65]), .ep_addr(8'h26), .ep_datain(last_timestamp_hi));
    okWireOut ep27 (.okHE(okHE), .okEH(okEHx[7*65 +: 65]), .ep_addr(8'h27),
        .ep_datain({13'b0, errors, fifo_count, 1'b0, word_index}));
endmodule
`default_nettype wire
