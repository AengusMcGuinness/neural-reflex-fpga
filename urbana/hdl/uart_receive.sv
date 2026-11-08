`timescale 1ns / 1ps
`default_nettype none
 
module uart_receive
  #(
    parameter INPUT_CLOCK_FREQ = 100_000_000,
    parameter BAUD_RATE = 9600
    )
   (
    input wire 	       clk,
    input wire 	       rst,
    input wire 	       din,
    output logic       dout_valid,
    output logic [7:0] dout
    );
 
  typedef enum {
    IDLE = 0,
    START = 1,
    DATA = 2,
    STOP = 3,
    TRANSMIT
    // TODO: define the rest of the states your receiver needs to operate
  } uart_state;
 

localparam  UART_BIT_PERIOD = INPUT_CLOCK_FREQ/BAUD_RATE;
localparam DATA_WIDTH = 8;

  // note: for the online checker, don't rename this variable
  uart_state state;
 
   // TODO: module to read UART rx wire

// make a counter to determine/time uart_bit_period
logic [31:0] uart_count;
always_ff @( posedge clk ) begin : uart_bit_period_counter
    if(rst || (state == IDLE))begin
        uart_count <= 0;
    end else begin
        if(uart_count < UART_BIT_PERIOD - 1)begin
            uart_count <= uart_count + 1;
        end else begin
            uart_count <= 0;
        end
    end    
end
 
 logic half_period;
 assign half_period = (uart_count == UART_BIT_PERIOD/2); //make sure to check this again to make sure our period is not a fraction
 //now we write our states....
logic [3:0] data_count;
logic [7:0] Rx_data;
 always_ff@(posedge clk)begin
    if(rst)begin
        state <= IDLE;
        dout <= 0;
        Rx_data <= 0;
        data_count <= 0;
        dout_valid <= 0;

    end else begin
        case(state)
        IDLE: begin
            data_count <= 0;
            dout_valid <= 0;
            if(!din)begin
                state <= START;
            end else begin
                state <= IDLE;
            end
        end

        START: begin
            if(!din && (uart_count < UART_BIT_PERIOD/2))begin
                //wait untill we reach half period
                state <= START;
            end
            else if(!din && half_period)begin
                state <= DATA;
            end else begin
                state <= IDLE;
            end 
        end

        DATA: begin
            
            if(data_count == DATA_WIDTH)begin
                state <= STOP;
                data_count <= 0;
            end else begin
                if(half_period)begin
                    Rx_data <= {din, Rx_data[7:1]};
                    data_count <= data_count + 1;
                    // state <= DATA; we stay here by default anyways
                end
            end
        end
        // end
        STOP: begin
            //check if we read 1 on din after half period 
            if(din && half_period)begin
                state <= TRANSMIT;
            end
            else begin
                if(~din && half_period)begin
                    state <= IDLE;
                end
            end
        end

        TRANSMIT: begin
            dout <= Rx_data;
            dout_valid <= 1'b1;
            state <= IDLE;
        end
        endcase
    end
 end
 

endmodule // uart_receive
 
`default_nettype wire