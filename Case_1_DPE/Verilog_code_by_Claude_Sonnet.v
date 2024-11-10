module jacobi_5ptr_sonnet (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [31:0] h,
    input wire [31:0] u_data_in,
    input wire u_data_valid,
    output reg [31:0] u_data_out,
    output reg u_data_out_valid,
    output reg done
);

// Parameters
parameter M = 4;  // Grid size
parameter MEM_SIZE = (M+2) * (M+2);  // Including boundary
parameter MAX_ITER = 1000;  // Maximum iterations
parameter INTEGER_BITS = 8;
parameter FRACTION_BITS = 24;

// Fixed-point arithmetic parameters
parameter THRESHOLD = 32'h00010000;  // ~0.0625 in fixed-point (lowered threshold)

// State definitions
localparam IDLE = 3'd0;
localparam LOAD = 3'd1;
localparam COMPUTE = 3'd2;
localparam CHECK = 3'd3;
localparam OUTPUT = 3'd4;

// Internal registers
reg [2:0] state;
reg [31:0] mem_u [0:MEM_SIZE-1];
reg [31:0] mem_unew [0:MEM_SIZE-1];
reg [9:0] write_addr;
reg [9:0] read_addr;
reg [9:0] compute_addr;
reg [9:0] iter_count;
reg [31:0] max_diff;

// Declare 'diff' outside the always block
reg [31:0] diff;

// Wire declarations
wire [31:0] new_value;
wire [9:0] row, col;
wire is_boundary;

// Calculate row and column from address
assign row = compute_addr / (M+2);
assign col = compute_addr % (M+2);
assign is_boundary = (row == 0) || (row == M+1) || (col == 0) || (col == M+1);

// Five-point stencil computation
wire [31:0] u_center, u_north, u_south, u_east, u_west;
assign u_center = mem_u[compute_addr];
assign u_north = mem_u[compute_addr - (M+2)];
assign u_south = mem_u[compute_addr + (M+2)];
assign u_east  = mem_u[compute_addr + 1];
assign u_west  = mem_u[compute_addr - 1];

// Fixed-point multiplication and division
wire [63:0] sum_neighbors;
assign sum_neighbors = {32'b0, u_north} + {32'b0, u_south} + {32'b0, u_east} + {32'b0, u_west};
assign new_value = sum_neighbors[31:0] >>> 2; // Divide by 4

// Initialize memories
integer i;
initial begin
    for (i = 0; i < MEM_SIZE; i = i + 1) begin
        mem_u[i] = 0;
        mem_unew[i] = 0;
    end
end

// Main state machine
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        write_addr <= 0;
        read_addr <= 0;
        compute_addr <= 0;
        iter_count <= 0;
        done <= 0;
        u_data_out_valid <= 0;
        max_diff <= 0;
        diff <= 0;
    end else begin
        case (state)
            IDLE: begin
                if (start) begin
                    state <= LOAD;
                    write_addr <= 0;
                    read_addr <= 0;
                    compute_addr <= 0;
                    iter_count <= 0;
                    done <= 0;
                    u_data_out_valid <= 0;
                    max_diff <= 0;
                    diff <= 0;
                end
            end

            LOAD: begin
                if (u_data_valid) begin
                    mem_u[write_addr] <= u_data_in;
                    write_addr <= write_addr + 1;
                    if (write_addr == MEM_SIZE-1) begin
                        state <= COMPUTE;
                        compute_addr <= 0;
                        max_diff <= 0;
                        diff <= 0;
                    end
                end
            end

            COMPUTE: begin
                if (!is_boundary) begin
                    mem_unew[compute_addr] <= new_value;
                end else begin
                    mem_unew[compute_addr] <= mem_u[compute_addr];
                end

                // Calculate the difference for convergence
                if (!is_boundary) begin
                    if (new_value > mem_u[compute_addr]) begin
                        diff <= new_value - mem_u[compute_addr];
                    end else begin
                        diff <= mem_u[compute_addr] - new_value;
                    end
                    if (diff > max_diff) begin
                        max_diff <= diff;
                    end
                end

                compute_addr <= compute_addr + 1;

                if (compute_addr == MEM_SIZE-1) begin
                    state <= CHECK;
                end
            end

            CHECK: begin
                if (max_diff < THRESHOLD || iter_count >= MAX_ITER-1) begin
                    state <= OUTPUT;
                    read_addr <= 0;
                end else begin
                    state <= COMPUTE;
                    compute_addr <= 0;
                    iter_count <= iter_count + 1;
                    max_diff <= 0;
                    diff <= 0;
                    // Copy mem_unew to mem_u
                    for (i = 0; i < MEM_SIZE; i = i + 1) begin
                        mem_u[i] <= mem_unew[i];
                    end
                end
            end

            OUTPUT: begin
                if (!done) begin
                    u_data_out <= mem_unew[read_addr];
                    u_data_out_valid <= 1;
                    read_addr <= read_addr + 1;
                    if (read_addr == MEM_SIZE-1) begin
                        state <= IDLE;
                        done <= 1;
                        u_data_out_valid <= 0;
                    end
                end
            end

            default: state <= IDLE;
        endcase
    end
end

// Debug information
always @(posedge clk) begin
    if (state == COMPUTE) begin
        $display("Time=%0t compute_addr=%0d row=%0d col=%0d is_boundary=%b new_value=%h", 
                 $time, compute_addr, row, col, is_boundary, new_value);
    end
    if (state == CHECK) begin
        $display("Iteration=%d max_diff=%h", iter_count, max_diff);
    end
end

endmodule
