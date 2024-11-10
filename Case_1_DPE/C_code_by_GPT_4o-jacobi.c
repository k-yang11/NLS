#include <stdio.h>
#include <math.h>

#define M 4
#define MAX_ITER 10000

void jacobi_poisson(double matrix[M][M]) {
    double new_matrix[M][M];
    double res;
    double norm_b = 0.0;
    int iter = 0;

    // Partition arrays to improve parallel access
    #pragma HLS ARRAY_PARTITION variable=matrix complete dim=2
    #pragma HLS ARRAY_PARTITION variable=new_matrix complete dim=2

    // Calculate norm_b
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            #pragma HLS PIPELINE
            if (i == 0 || i == M - 1 || j == 0 || j == M - 1) {
                norm_b += matrix[i][j] * matrix[i][j];
            }
        }
    }
    norm_b = sqrt(norm_b);

    // Jacobi iteration
    do {
        res = 0.0;

        // Update inner elements
        for (int i = 1; i < M - 1; i++) {
            for (int j = 1; j < M - 1; j++) {
                #pragma HLS PIPELINE
                new_matrix[i][j] = 0.25 * (matrix[i - 1][j] + matrix[i + 1][j] + matrix[i][j - 1] + matrix[i][j + 1]);

                res += (new_matrix[i][j] - matrix[i][j]) * (new_matrix[i][j] - matrix[i][j]);
            }
        }

        // Update main matrix
        for (int k = 1; k < M - 1; k++) {
            for (int j = 1; j < M - 1; j++) {
                #pragma HLS PIPELINE
                matrix[k][j] = new_matrix[k][j];
            }
        }

        res = sqrt(res) / norm_b;
        iter++;

    } while (res > 1e-1 && iter < MAX_ITER);
}
