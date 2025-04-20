#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <cuda_runtime.h>

// Define a 2D point structure
struct Point {
    double x, y;
};

// Gaussian kernel function
__device__ double gaussianKernel(double distance, double bandwidth) {
    return exp(-0.5 * pow(distance / bandwidth, 2));
}

// CUDA kernel to compute new centroids
__global__ void computeNewCentroids(const struct Point *data, struct Point *centroids, double *weightsSum, int numPoints, double bandwidth) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPoints) {
        double sumX = 0.0, sumY = 0.0;
        double totalWeight = 0.0;

        for (int j = 0; j < numPoints; j++) {
            double distance = sqrt(pow(centroids[i].x - data[j].x, 2) + pow(centroids[i].y - data[j].y, 2));
            double weight = gaussianKernel(distance, bandwidth);

            sumX += data[j].x * weight;
            sumY += data[j].y * weight;
            totalWeight += weight;
        }

        if (totalWeight > 0) {
            centroids[i].x = sumX / totalWeight;
            centroids[i].y = sumY / totalWeight;
        }
    }
}

// CUDA kernel to compute maximum shift
__global__ void computeMaxShift(const struct Point *centroids, const struct Point *newCentroids, double *maxShift, int numPoints) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPoints) {
        double shift = sqrt(pow(newCentroids[i].x - centroids[i].x, 2) + pow(newCentroids[i].y - centroids[i].y, 2));
        if (shift > *maxShift) {
            *maxShift = shift;
        }
    }
}

// Mean Shift clustering function
struct Point *meanShiftClustering(const struct Point *data, int numPoints, double bandwidth, double epsilon, int maxIterations, int *numCentroids) {
    struct Point *centroids = (struct Point *)malloc(numPoints * sizeof(struct Point));
    memcpy(centroids, data, numPoints * sizeof(struct Point));

    struct Point *d_data, *d_centroids;
    double *d_weightsSum, *d_maxShift;

    cudaMalloc((void **)&d_data, numPoints * sizeof(struct Point));
    cudaMalloc((void **)&d_centroids, numPoints * sizeof(struct Point));
    cudaMalloc((void **)&d_weightsSum, numPoints * sizeof(double));
    cudaMalloc((void **)&d_maxShift, sizeof(double));

    cudaMemcpy(d_data, data, numPoints * sizeof(struct Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, numPoints * sizeof(struct Point), cudaMemcpyHostToDevice);

    int iteration = 0;
    while (1) {
        iteration++;
        if (iteration > maxIterations) {
            printf("Reached maximum iterations.\n");
            break;
        }

        cudaMemset(d_weightsSum, 0, numPoints * sizeof(double));

        int blockSize = 256;
        int gridSize = (numPoints + blockSize - 1) / blockSize;

        computeNewCentroids<<<gridSize, blockSize>>>(d_data, d_centroids, d_weightsSum, numPoints, bandwidth);
        cudaDeviceSynchronize();

        double h_maxShift = 0.0;
        cudaMemcpy(d_maxShift, &h_maxShift, sizeof(double), cudaMemcpyHostToDevice);

        computeMaxShift<<<gridSize, blockSize>>>(d_centroids, d_centroids, d_maxShift, numPoints);
        cudaMemcpy(&h_maxShift, d_maxShift, sizeof(double), cudaMemcpyDeviceToHost);

        if (h_maxShift < epsilon) {
            break;
        }
    }

    cudaMemcpy(centroids, d_centroids, numPoints * sizeof(struct Point), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_weightsSum);
    cudaFree(d_maxShift);

    *numCentroids = 0;
    bool *isUnique = (bool *)malloc(numPoints * sizeof(bool));
    for (int i = 0; i < numPoints; i++) {
        isUnique[i] = true;
    }

    for (int i = 0; i < numPoints; i++) {
        if (isUnique[i]) {
            (*numCentroids)++;
            for (int j = i + 1; j < numPoints; j++) {
                double distance = sqrt(pow(centroids[i].x - centroids[j].x, 2) + pow(centroids[i].y - centroids[j].y, 2));
                if (distance < epsilon) {
                    isUnique[j] = false;
                }
            }
        }
    }

    struct Point *uniqueCentroids = (struct Point *)malloc((*numCentroids) * sizeof(struct Point));
    int index = 0;
    for (int i = 0; i < numPoints; i++) {
        if (isUnique[i]) {
            uniqueCentroids[index++] = centroids[i];
        }
    }

    free(centroids);
    free(isUnique);

    return uniqueCentroids;
}

// Function to read data points from a file
struct Point *readDataFromFile(const char *filename, int *numPoints) {
    struct Point *data = NULL;
    int capacity = 10;
    *numPoints = 0;
    data = (struct Point *)malloc(capacity * sizeof(struct Point));

    FILE *inputFile = fopen(filename, "r");
    if (!inputFile) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        exit(1);
    }

    double x, y;
    while (fscanf(inputFile, "%lf %lf", &x, &y) == 2) {
        if (*numPoints >= capacity) {
            capacity *= 2;
            data = (struct Point *)realloc(data, capacity * sizeof(struct Point));
        }
        data[*numPoints].x = x;
        data[*numPoints].y = y;
        (*numPoints)++;
    }

    fclose(inputFile);
    return data;
}

int main() {
    char filename[] = "test.txt";

    // Read data points from the file
    int numPoints = 0;
    struct Point *data = readDataFromFile(filename, &numPoints);

    // Mean Shift parameters
    double bandwidth = 4.0;  // Bandwidth for the Gaussian kernel
    double epsilon = 0.1;    // Convergence threshold
    int maxIterations = 100; // Maximum number of iterations

    int numCentroids = 0;

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    struct Point *centroids = meanShiftClustering(data, numPoints, bandwidth, epsilon, maxIterations, &numCentroids);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print results
    printf("Total time taken: %f milliseconds\n", milliseconds);
    printf("Number of Centroids: %d\n", numCentroids);
    for (int i = 0; i < numCentroids; i++) {
        printf("Centroid %d: (%f, %f)\n", i, centroids[i].x, centroids[i].y);
    }

    free(centroids);
    free(data);

    return 0;
}