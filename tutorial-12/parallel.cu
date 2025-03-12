#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 10000

double *matrix1 = (double *)malloc(N * N * sizeof(double));
double *matrix2 = (double *)malloc(N * N * sizeof(double));
double *result = (double *)malloc(N * N * sizeof(double));

__global__ void multiply_matrices(double *d_matrix1, double *d_matrix2, double *d_result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N) {
        int idx = i * N + j;
        d_result[idx] = d_matrix1[idx] * d_matrix2[idx];
        printf("d_result[%d] = %f\n", idx, d_result[idx]);
    }
}

void read_matrix_from_file(const char *filename, double *matrix) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    int total_elements = 0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fscanf(file, "%lf", &matrix[i * N + j]) != 1) {
                perror("Error reading file");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            else{
                total_elements++;
            }
        }
    }

    if (total_elements != N * N) {
        printf("Warning: Expected %d elements but found %d in the file.\n", N * N, total_elements);
    }
    fclose(file);
}

/*void add_matrices(double *matrix1, double *matrix2, double *result) {
    
}*/

int main() {
    if (!matrix1 || !matrix2 || !result) {
        perror("Error allocating memory");
        exit(EXIT_FAILURE);
    }

    read_matrix_from_file("output.txt", matrix1);
    read_matrix_from_file("output2.txt", matrix2);

    double *d_matrix1, *d_matrix2, *d_result;
    cudaMalloc((void **)&d_matrix1, N * N * sizeof(double));
    cudaMalloc((void **)&d_matrix2, N * N * sizeof(double));
    cudaMalloc((void **)&d_result, N * N * sizeof(double));

    cudaMemcpy(d_matrix1, matrix1, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, matrix2, N * N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    multiply_matrices<<<blocksPerGrid, threadsPerBlock>>>(d_matrix1, d_matrix2, d_result);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    printf("Time: %f seconds\n", milliseconds);

    cudaMemcpy(result, d_result, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_result);

    //add_matrices(matrix1, matrix2, result);

    free(matrix1);
    free(matrix2);
    free(result);

    return 0;
}