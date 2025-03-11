#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

__global__ void add(double *d_arr1, double *d_arr2, double *d_sum, int n){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n){
        d_sum[tid] = d_arr1[tid] + d_arr2[tid];
    }
}

int main(){
    FILE *fp1 = fopen("output1.txt","r");
    if(fp1 == NULL){
        printf("Error opening file 1\n");
        exit(1);
    }

    FILE *fp2 = fopen("output2.txt","r");
    if(fp2 == NULL){
        printf("Error opening file 2\n");
        exit(1);
    }

    int count = 0;
    double temp; 
    while(fscanf(fp1, "%lf", &temp) != EOF){
        count++;
    }

    double *h_arr1 = (double *)malloc(count * sizeof(double));
    double *h_arr2 = (double *)malloc(count * sizeof(double));
    double *sum = (double *)malloc(count * sizeof(double)); // Correct memory allocation
    if(h_arr1 == NULL || h_arr2 == NULL || sum == NULL){
        printf("Memory allocation failed\n");
        exit(1);
    }
    
    rewind(fp1);
    rewind(fp2);

    for(int i = 0; i < count; i++){
        fscanf(fp1, "%lf", &h_arr1[i]);
        fscanf(fp2, "%lf", &h_arr2[i]);
    }
    fclose(fp1);
    fclose(fp2);

    double *d_arr1, *d_arr2, *d_sum;
    cudaMalloc((void**)&d_arr1, count * sizeof(double));
    cudaMalloc((void**)&d_arr2, count * sizeof(double));
    cudaMalloc((void**)&d_sum, count * sizeof(double));

    cudaMemcpy(d_arr1, h_arr1, count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, h_arr2, count * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

    cudaDeviceSynchronize();
    clock_t start = clock();
    add<<<blocksPerGrid, threadsPerBlock>>>(d_arr1, d_arr2, d_sum, count);
    cudaDeviceSynchronize();
    clock_t end = clock();

    cudaMemcpy(sum, d_sum, count * sizeof(double), cudaMemcpyDeviceToHost); 

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %lf\n", time_taken);   

    free(h_arr1);
    free(h_arr2);
    free(sum);
    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_sum);

    return 0;
}
