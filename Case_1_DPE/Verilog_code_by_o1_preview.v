`timescale 1ns / 1ps

module jacobi_5ptr #(
    parameter M = 4,          // ??????????????
    parameter WIDTH = 32,     // ?????
    parameter FRAC = 16       // ??????
)(
    input wire clk,
    input wire rst,
    input wire start,
    output reg done,
    input wire [(M+2)*(M+2)*WIDTH-1:0] u_in_flat,  // ???????
    output reg [(M+2)*(M+2)*WIDTH-1:0] u_out_flat  // ???????
);

    // ?????
    localparam ONE = 1 << FRAC;
    localparam THRESHOLD = (ONE / 10); // ?????0.1

    // ????
    reg [WIDTH-1:0] u [(M+2)*(M+2)-1:0];
    reg [WIDTH-1:0] unew [(M+2)*(M+2)-1:0];

    // ??????
    integer idx;
    reg [15:0] i;
    reg [WIDTH+31:0] norm_b; // ?????????
    reg [WIDTH+31:0] norm_r;
    reg [WIDTH+31:0] res;

    // ???????
    localparam IDLE = 0;
    localparam LOAD_INPUT = 1;
    localparam CALC_NORM_B = 2;
    localparam CALC_NORM_R = 3;
    localparam CHECK_RES = 4;
    localparam UPDATE_UNEW = 5;
    localparam COPY_UNEW = 6;
    localparam OUTPUT_RESULT = 7;

    reg [2:0] state;
    reg [2:0] next_state;

    // ???
    always @(posedge clk or posedge rst) begin
        if (rst)
            state <= IDLE;
        else
            state <= next_state;
    end

    // ??????
    always @(*) begin
        case (state)
            IDLE:
                if (start)
                    next_state = LOAD_INPUT;
                else
                    next_state = IDLE;
            LOAD_INPUT:
                next_state = CALC_NORM_B;
            CALC_NORM_B:
                if (i >= (M + 2) * 4)
                    next_state = CALC_NORM_R;
                else
                    next_state = CALC_NORM_B;
            CALC_NORM_R:
                if (i >= M * M)
                    next_state = CHECK_RES;
                else
                    next_state = CALC_NORM_R;
            CHECK_RES:
                if (res > THRESHOLD)
                    next_state = UPDATE_UNEW;
                else
                    next_state = OUTPUT_RESULT;
            UPDATE_UNEW:
                if (i >= M * M)
                    next_state = COPY_UNEW;
                else
                    next_state = UPDATE_UNEW;
            COPY_UNEW:
                if (i >= M * M)
                    next_state = CALC_NORM_R;
                else
                    next_state = COPY_UNEW;
            OUTPUT_RESULT:
                next_state = IDLE;
            default:
                next_state = IDLE;
        endcase
    end

    // ????
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            // ??????
            done <= 0;
            norm_b <= 0;
            norm_r <= 0;
            res <= 0;
            i <= 0;
            for (idx = 0; idx < (M+2)*(M+2); idx = idx + 1) begin
                u[idx] <= 0;
                unew[idx] <= 0;
            end
        end else begin
            case (state)
                LOAD_INPUT: begin
                    // ??????????? u
                    for (idx = 0; idx < (M+2)*(M+2); idx = idx + 1) begin
                        u[idx] <= u_in_flat[(idx*WIDTH) +: WIDTH];
                    end
                    done <= 0;
                    norm_b <= 0;
                    i <= 0;
                end
                CALC_NORM_B: begin
                    if (i < (M + 2) * 4) begin
                        idx = i % (M + 2);
                        case (i / (M + 2))
                            0: norm_b <= norm_b + abs_fixed(u[idx]); // u[0][j]
                            1: norm_b <= norm_b + abs_fixed(u[(M + 1)*(M + 2) + idx]); // u[M+1][j]
                            2: norm_b <= norm_b + abs_fixed(u[idx*(M + 2)]); // u[j][0]
                            3: norm_b <= norm_b + abs_fixed(u[idx*(M + 2) + (M + 1)]); // u[j][M+1]
                        endcase
                        i <= i + 1;
                    end else begin
                        // ??? i
                        i <= 0;
                        norm_r <= 0;
                    end
                end
                CALC_NORM_R: begin
                    if (i < M * M) begin
                        idx = ((i / M) + 1) * (M + 2) + (i % M) + 1;
                        norm_r <= norm_r + abs_fixed(
                            - u[idx - 1] // u[i][j - 1]
                            - u[idx - (M + 2)] // u[i - 1][j]
                            + (u[idx] << 2) // 4 * u[i][j]
                            - u[idx + (M + 2)] // u[i + 1][j]
                            - u[idx + 1] // u[i][j + 1]
                        );
                        i <= i + 1;
                    end else begin
                        // ?? res
                        if (norm_b != 0)
                            res <= (norm_r << FRAC) / norm_b; // ???????
                        else
                            res <= 0;
                        // ?????
                        i <= 0;
                    end
                end
                CHECK_RES: begin
                    if (res > THRESHOLD) begin
                        done <= 0;
                        // ???? unew
                        i <= 0;
                    end else begin
                        // ????? done
                        i <= 0;
                    end
                end
                UPDATE_UNEW: begin
                    if (i < M * M) begin
                        idx = ((i / M) + 1) * (M + 2) + (i % M) + 1;
                        unew[idx] <= (u[idx - 1] // u[i][j - 1]
                                     + u[idx - (M + 2)] // u[i - 1][j]
                                     + u[idx + (M + 2)] // u[i + 1][j]
                                     + u[idx + 1] // u[i][j + 1]
                                    ) >> 2; // ?? 4
                        i <= i + 1;
                    end else begin
                        // ?????
                        i <= 0;
                    end
                end
                COPY_UNEW: begin
                    if (i < M * M) begin
                        idx = ((i / M) + 1) * (M + 2) + (i % M) + 1;
                        u[idx] <= unew[idx];
                        i <= i + 1;
                    end else begin
                        // ?????
                        i <= 0;
                        norm_r <= 0;
                    end
                end
                OUTPUT_RESULT: begin
                    // ?????? done
                    for (idx = 0; idx < (M+2)*(M+2); idx = idx + 1) begin
                        u_out_flat[(idx*WIDTH) +: WIDTH] <= u[idx];
                    end
                    done <= 1;
                end
                default: ;
            endcase
        end
    end

    // ????????
    function [WIDTH-1:0] abs_fixed;
        input [WIDTH-1:0] value;
        begin
            if (value[WIDTH-1] == 1'b1) // ??
                abs_fixed = (~value) + 1;
            else
                abs_fixed = value;
        end
    endfunction

endmodule
