#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void dot_product(double *d_arr1, double *d_arr2, double *d_dot, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        atomicAdd(d_dot, d_arr1[tid] * d_arr2[tid]);
    }
}

int main() {
    FILE *fp1 = fopen("output1.txt", "r");
    FILE *fp2 = fopen("output2.txt", "r");
    if (fp1 == NULL || fp2 == NULL) {
        printf("Error opening files\n");
        exit(1);
    }

    int count = 0;
    double temp, dot_prod = 0.0;
    while (fscanf(fp1, "%lf", &temp) != EOF) {
        count++;
    }

    double *h_arr1 = (double *)malloc(count * sizeof(double));
    double *h_arr2 = (double *)malloc(count * sizeof(double));
    if (h_arr1 == NULL || h_arr2 == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    rewind(fp1);
    rewind(fp2);

    for (int i = 0; i < count; i++) {
        fscanf(fp1, "%lf", &h_arr1[i]);
        fscanf(fp2, "%lf", &h_arr2[i]);
    }
    fclose(fp1);
    fclose(fp2);

    double *d_arr1, *d_arr2, *d_dot;
    cudaMalloc((void **)&d_arr1, count * sizeof(double));
    cudaMalloc((void **)&d_arr2, count * sizeof(double));
    cudaMalloc((void **)&d_dot, sizeof(double));

    cudaMemcpy(d_arr1, h_arr1, count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, h_arr2, count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_dot, 0, sizeof(double));

    int threadsPerBlock = 1024;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

    cudaDeviceSynchronize();
    clock_t start = clock();
    dot_product<<<blocksPerGrid, threadsPerBlock>>>(d_arr1, d_arr2, d_dot, count);
    cudaDeviceSynchronize();
    clock_t end = clock();

    cudaMemcpy(&dot_prod, d_dot, sizeof(double), cudaMemcpyDeviceToHost);

    double kernel_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Dot Product: %.2lf\n", dot_prod);
    printf("Kernel Execution Time: %.6f seconds\n", kernel_time);

    free(h_arr1);
    free(h_arr2);
    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_dot);

    return 0;
}
