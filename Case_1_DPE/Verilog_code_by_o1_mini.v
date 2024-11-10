`timescale 1ns / 1ps

module jacobi_5ptr_mini #
    (
        parameter M = 4,
        parameter WIDTH = 32,
        parameter FRAC = 16
    )
    (
        input clk,
        input rst,
        input start,
        input [((M+2)*(M+2)*WIDTH)-1:0] u_in_flat,
        output reg done,
        output reg [((M+2)*(M+2)*WIDTH)-1:0] u_out_flat
    );

// ????
localparam TOTAL_SIZE = (M + 2) * (M + 2);

// ???????
localparam IDLE           = 4'd0;
localparam LOAD_U         = 4'd1;
localparam CALC_NORM_B    = 4'd2;
localparam INIT_ITER      = 4'd3;
localparam CALC_UNEW      = 4'd4;
localparam CALC_NORM_R    = 4'd5;
localparam CHECK_CONV     = 4'd6;
localparam COPY_UNEW      = 4'd7;
localparam OUTPUT_RESULT  = 4'd8;

// ???????
reg [WIDTH-1:0] u_reg [0:TOTAL_SIZE-1];
reg [WIDTH-1:0] unew_reg [0:TOTAL_SIZE-1];
reg [31:0] norm_b;
reg [63:0] norm_r; // ??????????
reg [31:0] res;
reg [3:0] state;
reg [15:0] i, j; // ??16???????
reg [31:0] abs_val; // ???????
reg [63:0] temp_residual; // ??????

integer idx;

// ?????
always @(posedge clk or posedge rst) begin
    if (rst) begin
        // ???????
        state <= IDLE;
        done <= 1'b0;
        norm_b <= 32'd0;
        norm_r <= 64'd0;
        res <= 32'd0;
        i <= 16'd0;
        j <= 16'd0;
        u_out_flat <= {((M+2)*(M+2)*WIDTH){1'b0}};
        for (idx = 0; idx < TOTAL_SIZE; idx = idx + 1) begin
            u_reg[idx] <= 32'd0;
            unew_reg[idx] <= 32'd0;
        end
    end
    else begin
        case (state)
            IDLE: begin
                done <= 1'b0;
                if (start) begin
                    state <= LOAD_U;
                    i <= 0;
                    j <= 0;
                end
            end

            LOAD_U: begin
                // ? u_in_flat ????? u_reg
                if (i < TOTAL_SIZE) begin
                    u_reg[i] <= u_in_flat[(i*WIDTH) +: WIDTH];
                    i <= i + 1;
                end
                else begin
                    i <= 0;
                    j <= 0;
                    state <= CALC_NORM_B;
                end
            end

            CALC_NORM_B: begin
                // ?? norm_b = sum(|u[0][j]| + |u[M+1][j]| + |u[j][0]| + |u[j][M+1]|)
                if (j < M + 2) begin
                    // ????
                    // u[0][j] = j
                    // u[M+1][j] = (M+1)*(M+2) + j
                    // u[j][0] = j*(M+2)
                    // u[j][M+1] = j*(M+2) + (M+1)

                    // |u[0][j]|
                    if (u_reg[j][31])
                        abs_val = ~u_reg[j] + 1;
                    else
                        abs_val = u_reg[j];
                    norm_b <= norm_b + abs_val;

                    // |u[M+1][j]|
                    if (u_reg[(M+1)*(M+2) + j][31])
                        abs_val = ~u_reg[(M+1)*(M+2) + j] + 1;
                    else
                        abs_val = u_reg[(M+1)*(M+2) + j];
                    norm_b <= norm_b + abs_val;

                    // |u[j][0]|
                    if (u_reg[j*(M+2)][31])
                        abs_val = ~u_reg[j*(M+2)] + 1;
                    else
                        abs_val = u_reg[j*(M+2)];
                    norm_b <= norm_b + abs_val;

                    // |u[j][M+1]|
                    if (u_reg[j*(M+2) + (M+1)][31])
                        abs_val = ~u_reg[j*(M+2) + (M+1)] + 1;
                    else
                        abs_val = u_reg[j*(M+2) + (M+1)];
                    norm_b <= norm_b + abs_val;

                    j <= j + 1;
                end
                else begin
                    // ?? norm_b ????????
                    j <= 1;
                    i <= 1;
                    norm_r <= 64'd0;
                    state <= CALC_UNEW;
                end
            end

            CALC_UNEW: begin
                // ?? unew[i][j] = 0.25 * (u[i][j-1] + u[i-1][j] + u[i+1][j] + u[i][j+1])
                if (i <= M && j <= M) begin
                    // ??????
                    unew_reg[i*(M+2) + j] <= (u_reg[i*(M+2) + (j-1)] + 
                                              u_reg[(i-1)*(M+2) + j] + 
                                              u_reg[(i+1)*(M+2) + j] + 
                                              u_reg[i*(M+2) + (j+1)]) >>> 2;
                    if (j < M)
                        j <= j + 1;
                    else begin
                        j <= 1;
                        i <= i + 1;
                    end
                end
                else begin
                    // ?? unew ??????? norm_r
                    i <=1;
                    j <=1;
                    norm_r <= 64'd0;
                    state <= CALC_NORM_R;
                end
            end

            CALC_NORM_R: begin
                // ?? norm_r = sum(|-u[i][j-1] - u[i-1][j] + 4*u[i][j] - u[i+1][j] - u[i][j+1]|)
                if (i <= M && j <= M) begin
                    // ????
                    temp_residual = (-u_reg[i*(M+2) + (j-1)] - 
                                      u_reg[(i-1)*(M+2) + j] + 
                                      (u_reg[i*(M+2) + j] << 2) - 
                                      u_reg[(i+1)*(M+2) + j] - 
                                      u_reg[i*(M+2) + (j+1)]);
                    
                    // ????
                    if (temp_residual[31] == 1'b1)
                        norm_r <= norm_r + (~temp_residual + 1);
                    else
                        norm_r <= norm_r + temp_residual;
                    
                    // ????
                    if (j < M)
                        j <= j + 1;
                    else begin
                        j <=1;
                        i <= i +1;
                    end
                end
                else begin
                    // ?? norm_r ????? res
                    // res = (norm_r << FRAC) / norm_b
                    // ?????????????????????
                    res <= (norm_r >> (64 - WIDTH)) / norm_b;
                    state <= CHECK_CONV;
                end
            end

            CHECK_CONV: begin
                // ???? 1e-1 ???????
                // 1e-1 * (1 << FRAC) ? 6553
                // ?? scaled_norm_r > threshold * norm_b
                // ???? scaled_norm_r ? threshold * norm_b
                if ((norm_r << FRAC) > (32'd6553 * norm_b)) begin
                    // ??????
                    state <= COPY_UNEW;
                    i <=1;
                    j <=1;
                end
                else begin
                    // ????
                    state <= OUTPUT_RESULT;
                end
            end

            COPY_UNEW: begin
                // ? unew_reg ??? u_reg
                if (i <= M && j <= M) begin
                    u_reg[i*(M+2) + j] <= unew_reg[i*(M+2) + j];
                    if (j < M)
                        j <= j +1;
                    else begin
                        j <=1;
                        i <= i +1;
                    end
                end
                else begin
                    // ????????????
                    i <=1;
                    j <=1;
                    norm_r <=64'd0;
                    state <= CALC_UNEW;
                end
            end

            OUTPUT_RESULT: begin
                // ? u_reg ???? u_out_flat
                if (i < TOTAL_SIZE) begin
                    u_out_flat[(i*WIDTH) +: WIDTH] <= u_reg[i];
                    i <= i + 1;
                end
                else begin
                    done <= 1'b1;
                    state <= IDLE;
                end
            end

            default: begin
                state <= IDLE;
            end

        endcase
    end
end

endmodule