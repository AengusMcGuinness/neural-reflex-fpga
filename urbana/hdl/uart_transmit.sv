module uart_transmit #(parameter int INPUT_CLOCK_FREQ = 100000000,
                        parameter int BAUD_RATE = 115_200)(input wire clk,
                    input wire rst,
                    input wire [7:0] din,
                    input wire trigger,
                    output logic busy,
                    output logic dout
);

// create a uart transmitter... we shall use our spi code for most of it....
localparam BAUD_BIT_PREIOD = INPUT_CLOCK_FREQ/BAUD_RATE;

//build a counter who signal is on half the time of baudrate and off half the time 
logic [31:0] period_count;

always_ff @(posedge clk) begin : bit_period_counter
    if(rst)begin
        period_count <= 0;
    end else begin
        if(busy && period_count < BAUD_BIT_PREIOD -1)begin
            period_count <= period_count + 1;
        end else begin
            period_count <= 0;
        end
    end    
end

//now another cunter to determine when the data has been fully transmitted! Note: this only tracks the data bits not the start bit
logic [15:0] data_period;
logic byte_complete;
always_ff@(posedge clk)begin : how_many_bits_did_we_send
    if(rst)begin
        data_period <= 0;
    end else begin
        if((next_bit) && data_period < 9)begin
            data_period <= data_period + 1;
        end else begin
            //lmao make sure it last one BAUD_BIT_PERIOD
            if((data_period == 9)&& next_bit)begin
                data_period <= 0;
            end
        end
    end
end

assign byte_complete = (data_period == 9) && next_bit; 

// when should I send the next bit? 
logic next_bit;
assign next_bit = (busy && period_count == BAUD_BIT_PREIOD - 1);

// determine our busy signal
always_ff@(posedge clk) begin : set_busy
    if(rst)begin
        busy <= 0;
    end else begin
        if(trigger && !busy)begin
            busy <= 1'b1;
        end else begin
            if(byte_complete)begin
                busy <= 0;
            end
        end
    end
end

//capture the transmission data as well
// now determine the serialization to out put our data!....
/* ther signal generated we hard.code the first and last bit to be 0 and 1 then serialize the last three */

logic [7:0] latched_din;
always_ff@(posedge clk)begin :capture_data_in_serialization
    if(rst)begin
        latched_din <= 0;
        dout <= 1'b1;
    end else if(!busy)begin
            if(trigger)begin
                latched_din <= din;
                dout <= 1'b0;
            end else begin
                dout <= 1'b1;
                latched_din <= 0;
            end
    end   
    else begin
        if(next_bit && data_period < 8) begin //transmit data by serialization make sure to use shift register!
            latched_din <= {latched_din[0], latched_din[7:1]};
            dout <= latched_din[0];
        end else begin
            if((data_period == 8) && next_bit)begin
                dout <= 1'b1;
            end
        end
        end
    end
endmodule