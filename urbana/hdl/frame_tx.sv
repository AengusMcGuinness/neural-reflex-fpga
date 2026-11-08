`timescale 1ns / 1ps
`default_nettype none

// Sends FPGA->host frames through uart_transmit:
//   5A A5 81 <N_OUT x int16 y, little-endian> CHK    decoder outputs
//   5A A5 82 00 CHK                                 weights loaded OK
//   5A A5 EE <cmd> CHK                              bad checksum / unknown command
// CHK = sum of the type byte and payload bytes, mod 256.
// Requests that arrive while a frame is being sent are dropped.
module frame_tx #(
    parameter int N_OUT = 6
)(
    input wire clk,
    input wire rst,
    input wire              send_result,
    input wire [16*N_OUT-1:0] y_flat,        // y[j] in bits [16*j +: 16]
    input wire              send_load_ok,
    input wire              send_error,
    input wire [7:0]        error_cmd,

    output logic            tx_trigger,
    output logic [7:0]      tx_byte,
    input wire              tx_busy,
    output logic            busy
);
    localparam int MAX_PAYLOAD = 2 * N_OUT;

    logic [7:0] type_b;
    logic [7:0] payload [MAX_PAYLOAD];
    logic [$clog2(MAX_PAYLOAD + 5)-1:0] k, n_bytes;  // byte index, total bytes in frame
    logic [7:0] sum;

    typedef enum logic [1:0] {IDLE, LOAD, WAIT_START, WAIT_DONE} state_t;
    state_t state;

    logic [7:0] cur;
    always_comb begin
        if (k == 0)                cur = 8'h5A;
        else if (k == 1)           cur = 8'hA5;
        else if (k == 2)           cur = type_b;
        else if (k == n_bytes - 1) cur = sum;
        else                       cur = payload[k - 3];
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            tx_trigger <= 0;
            busy <= 0;
        end else begin
            tx_trigger <= 0;
            case (state)
                IDLE: begin
                    k <= 0;
                    if (send_result) begin
                        type_b <= 8'h81;
                        for (int j = 0; j < N_OUT; j++) begin
                            payload[2*j]     <= y_flat[16*j +: 8];
                            payload[2*j + 1] <= y_flat[16*j + 8 +: 8];
                        end
                        n_bytes <= 2 * N_OUT + 4;
                        state <= LOAD;
                        busy <= 1;
                    end else if (send_load_ok || send_error) begin
                        type_b <= send_error ? 8'hEE : 8'h82;
                        payload[0] <= send_error ? error_cmd : 8'h00;
                        n_bytes <= 5;
                        state <= LOAD;
                        busy <= 1;
                    end
                end
                LOAD: begin
                    tx_byte <= cur;
                    if (!tx_busy) begin
                        // checksum over type + payload, accumulated once per byte sent
                        if (k == 2) sum <= cur;
                        else if (k > 2 && k < n_bytes - 1) sum <= sum + cur;
                        tx_trigger <= 1;
                        state <= WAIT_START;
                    end
                end
                WAIT_START: if (tx_busy) state <= WAIT_DONE;
                WAIT_DONE: if (!tx_busy) begin
                    if (k == n_bytes - 1) begin
                        state <= IDLE;
                        busy <= 0;
                    end else begin
                        k <= k + 1;
                        state <= LOAD;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
