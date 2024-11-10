#include <math.h>
#define M 4

void jacobi_5ptr(double u[M + 2][M + 2], double unew[M + 2][M + 2], double h) {
    #pragma HLS INTERFACE m_axi port=u offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=unew offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=u bundle=control
    #pragma HLS INTERFACE s_axilite port=unew bundle=control
    #pragma HLS INTERFACE s_axilite port=h bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    double norm_b = 0.0;
    for (int j = 0; j < M + 2; j++) {
        #pragma HLS PIPELINE
        norm_b += fabs(u[0][j]) + fabs(u[M + 1][j]) + fabs(u[j][0]) + fabs(u[j][M + 1]);
    }

    double norm_r = 0.0;
    for (int i = 1; i < M + 1; i++) {
        for (int j = 1; j < M + 1; j++) {
            #pragma HLS PIPELINE
            norm_r += fabs(-u[i][j - 1] - u[i - 1][j] + 4 * u[i][j] - u[i + 1][j] - u[i][j + 1]);
        }
    }
    double res = norm_r / norm_b;

    while (res > 1e-1) {
        for (int i = 1; i < M + 1; i++) {
            for (int j = 1; j < M + 1; j++) {
                #pragma HLS PIPELINE
                unew[i][j] = 0.25 * (u[i][j - 1] + u[i - 1][j] + u[i + 1][j] + u[i][j + 1]);
            }
        }
        for (int i = 1; i < M + 1; i++) {
            for (int j = 1; j < M + 1; j++) {
                #pragma HLS PIPELINE
                u[i][j] = unew[i][j];
            }
        }

        norm_r = 0.0;
        for (int i = 1; i < M + 1; i++) {
            for (int j = 1; j < M + 1; j++) {
                #pragma HLS PIPELINE
                norm_r += fabs(-u[i][j - 1] - u[i - 1][j] + 4 * u[i][j] - u[i + 1][j] - u[i][j + 1]);
            }
        }
        res = norm_r / norm_b;
    }
}
