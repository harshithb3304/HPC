#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void reduce_sum(double *d_arr, double *d_sum, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ double shared_data[1024];
    
    if (tid < n) {
        shared_data[threadIdx.x] = d_arr[tid];
    } else {
        shared_data[threadIdx.x] = 0.0;
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        double local_sum = 0.0;
        for (int i = 0; i < blockDim.x && tid + i < n; i++) {
            local_sum += shared_data[i];
        }
        atomicAdd(d_sum, local_sum);
    }
}

int main() {
    FILE *fp = fopen("output.txt", "r");
    if (fp == NULL) {
        printf("Error opening file\n");
        exit(1);
    }
    
    int count = 0;
    double temp, sum = 0.0;
    while (fscanf(fp, "%lf", &temp) != EOF) {
        count++;
    }
    
    double *h_arr = (double *)malloc(count * sizeof(double));
    if (h_arr == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    rewind(fp);
    
    for (int i = 0; i < count; i++) {
        fscanf(fp, "%lf", &h_arr[i]);
    }
    fclose(fp);
    
    double *d_arr, *d_sum;
    cudaMalloc((void **)&d_arr, count * sizeof(double));
    cudaMalloc((void **)&d_sum, sizeof(double));
    cudaMemcpy(d_arr, h_arr, count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(double));
    
    int threadsPerBlock = 1024;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaDeviceSynchronize();
    clock_t start = clock();
    reduce_sum<<<blocksPerGrid, threadsPerBlock>>>(d_arr, d_sum, count);
    cudaDeviceSynchronize();
    clock_t end = clock();
    
    cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_arr);
    cudaFree(d_sum);
    free(h_arr);
    
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Sum: %.2lf\n", sum);
    printf("Kernel execution time: %.6f seconds\n", time_taken);
    
    return 0;
}
