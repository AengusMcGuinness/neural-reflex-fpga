`timescale 1ns / 1ps
`default_nettype none

// Parses host->FPGA frames from the UART byte stream (all multi-byte values little-endian):
//   A5 5A 01 <N_FEAT x int16 features>                                   CHK   features -> compute
//   A5 5A 02 <N_OUT*N_FEAT x int16 w> <N_OUT x int48 b> <3*N_OUT x int16 cfg> CHK   load weights
// CHK = sum of the command byte and payload bytes, mod 256.
// Payload words are written straight into ridge_mac as they arrive; the checksum
// only decides whether frame_ok or frame_bad fires at the end.
// A gap of TIMEOUT_CYCLES between bytes drops the frame and goes back to hunting for A5 5A.
module frame_rx #(
    parameter int N_FEAT = 40,
    parameter int N_OUT = 6,
    parameter int TIMEOUT_CYCLES = 1_000_000
)(
    input wire clk,
    input wire rst,
    input wire       byte_valid,
    input wire [7:0] byte_in,

    output logic                                 feat_we,
    output logic [$clog2(N_FEAT)-1:0]            feat_addr,
    output logic [15:0]                          feat_data,
    output logic                                 w_we,
    output logic [$clog2(N_FEAT*N_OUT)-1:0]      w_addr,
    output logic [15:0]                          w_data,
    output logic                                 b_we,
    output logic [$clog2(N_OUT)-1:0]             b_addr,
    output logic [47:0]                          b_data,
    output logic                                 cfg_we,
    output logic [$clog2(3*N_OUT)-1:0]           cfg_addr,
    output logic [15:0]                          cfg_data,

    output logic       frame_ok,    // pulse: complete frame with good checksum
    output logic       frame_bad,   // pulse: complete frame with bad checksum, or unknown command
    output logic [7:0] frame_cmd    // command of the frame that just ended
);
    localparam logic [7:0] SYNC0 = 8'hA5, SYNC1 = 8'h5A;
    localparam logic [7:0] CMD_FEAT = 8'h01, CMD_LOAD = 8'h02;
    localparam int FEAT_BYTES = 2 * N_FEAT;
    localparam int W_BYTES = 2 * N_FEAT * N_OUT;
    localparam int B_BYTES = 6 * N_OUT;
    localparam int C_BYTES = 6 * N_OUT;
    localparam int LOAD_BYTES = W_BYTES + B_BYTES + C_BYTES;

    typedef enum logic [2:0] {HUNT0, HUNT1, CMD, PAYLOAD, CHECK} state_t;
    state_t state;

    logic [15:0] cnt;          // payload byte index
    logic [15:0] payload_len;
    logic [7:0]  sum;
    logic [7:0]  cmd;
    logic [39:0] shreg;        // previous payload bytes of the current word
    logic [2:0]  sub;          // byte position within the current word
    logic [31:0] idle_cycles;

    // word counters for each section of a load frame
    logic [$clog2(N_FEAT)-1:0]        fi;
    logic [$clog2(N_FEAT*N_OUT)-1:0]  wi;
    logic [$clog2(N_OUT)-1:0]         bi;
    logic [$clog2(3*N_OUT)-1:0]       ci;

    logic in_w, in_b;
    assign in_w = (cnt < W_BYTES);
    assign in_b = (cnt >= W_BYTES) && (cnt < W_BYTES + B_BYTES);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= HUNT0;
            feat_we <= 0; w_we <= 0; b_we <= 0; cfg_we <= 0;
            frame_ok <= 0; frame_bad <= 0; frame_cmd <= 0;
            idle_cycles <= 0;
        end else begin
            feat_we <= 0; w_we <= 0; b_we <= 0; cfg_we <= 0;
            frame_ok <= 0; frame_bad <= 0;

            if (byte_valid) idle_cycles <= 0;
            else if (idle_cycles < TIMEOUT_CYCLES) idle_cycles <= idle_cycles + 1;

            if (!byte_valid && state != HUNT0 && idle_cycles >= TIMEOUT_CYCLES) begin
                state <= HUNT0;
            end else if (byte_valid) begin
                case (state)
                    HUNT0: if (byte_in == SYNC0) state <= HUNT1;
                    HUNT1: state <= (byte_in == SYNC1) ? CMD : (byte_in == SYNC0) ? HUNT1 : HUNT0;
                    CMD: begin
                        cmd <= byte_in;
                        sum <= byte_in;
                        cnt <= 0; sub <= 0;
                        fi <= 0; wi <= 0; bi <= 0; ci <= 0;
                        if (byte_in == CMD_FEAT) begin
                            payload_len <= FEAT_BYTES;
                            state <= PAYLOAD;
                        end else if (byte_in == CMD_LOAD) begin
                            payload_len <= LOAD_BYTES;
                            state <= PAYLOAD;
                        end else begin
                            frame_bad <= 1;
                            frame_cmd <= byte_in;
                            state <= HUNT0;
                        end
                    end
                    PAYLOAD: begin
                        sum <= sum + byte_in;
                        cnt <= cnt + 1;
                        if (cnt == payload_len - 1) state <= CHECK;
                        if (cmd == CMD_FEAT || in_w || !in_b) begin
                            // 16-bit words: features, weights, cfg
                            if (sub == 0) begin
                                shreg[7:0] <= byte_in;
                                sub <= 1;
                            end else begin
                                sub <= 0;
                                if (cmd == CMD_FEAT) begin
                                    feat_we <= 1; feat_addr <= fi; feat_data <= {byte_in, shreg[7:0]}; fi <= fi + 1;
                                end else if (in_w) begin
                                    w_we <= 1; w_addr <= wi; w_data <= {byte_in, shreg[7:0]}; wi <= wi + 1;
                                end else begin
                                    cfg_we <= 1; cfg_addr <= ci; cfg_data <= {byte_in, shreg[7:0]}; ci <= ci + 1;
                                end
                            end
                        end else begin
                            // 48-bit bias words
                            if (sub == 5) begin
                                sub <= 0;
                                b_we <= 1; b_addr <= bi; b_data <= {byte_in, shreg[39:0]}; bi <= bi + 1;
                            end else begin
                                shreg <= {byte_in, shreg[39:8]};
                                sub <= sub + 1;
                            end
                        end
                    end
                    CHECK: begin
                        frame_cmd <= cmd;
                        if (byte_in == sum) frame_ok <= 1;
                        else frame_bad <= 1;
                        state <= HUNT0;
                    end
                    default: state <= HUNT0;
                endcase
            end
        end
    end
endmodule

`default_nettype wire
