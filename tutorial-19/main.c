#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <mpi.h> // Include MPI header

#define MASTER 0

// Define a 2D point structure
struct Point
{
    double x, y;
};

// Gaussian kernel function
double gaussianKernel(double distance, double bandwidth)
{
    return exp(-0.5 * pow(distance / bandwidth, 2));
}

// Function to calculate new centroid positions for a subset of centroids
void calculateNewCentroids(const struct Point *data, int numPoints,
                           const struct Point *centroids, int startIdx, int endIdx,
                           struct Point *newCentroids, double *weightsSum, double bandwidth)
{
    for (int i = startIdx; i < endIdx; i++)
    {
        newCentroids[i - startIdx].x = 0.0;
        newCentroids[i - startIdx].y = 0.0;
        weightsSum[i - startIdx] = 0.0;

        for (int j = 0; j < numPoints; j++)
        {
            double distance = sqrt(pow(centroids[i].x - data[j].x, 2) +
                                   pow(centroids[i].y - data[j].y, 2));
            double weight = gaussianKernel(distance, bandwidth);

            newCentroids[i - startIdx].x += data[j].x * weight;
            newCentroids[i - startIdx].y += data[j].y * weight;
            weightsSum[i - startIdx] += weight;
        }

        if (weightsSum[i - startIdx] > 0)
        {
            newCentroids[i - startIdx].x /= weightsSum[i - startIdx];
            newCentroids[i - startIdx].y /= weightsSum[i - startIdx];
        }
    }
}

// Function to calculate maximum shift for a subset of centroids
double calculateMaxShift(const struct Point *oldCentroids, const struct Point *newCentroids,
                         int startIdx, int count)
{
    double maxShift = 0.0;
    for (int i = 0; i < count; i++)
    {
        double shift = sqrt(pow(newCentroids[i].x - oldCentroids[startIdx + i].x, 2) +
                            pow(newCentroids[i].y - oldCentroids[startIdx + i].y, 2));
        if (shift > maxShift)
        {
            maxShift = shift;
        }
    }
    return maxShift;
}

// Mean Shift clustering function using MPI
struct Point *meanShiftClustering(const struct Point *data, int numPoints, double bandwidth,
                                  double epsilon, int maxIterations, int *numCentroids, int rank, int size)
{
    struct Point *centroids = NULL;
    struct Point *allCentroids = NULL;
    int iteration = 0;
    double globalMaxShift = 0.0;

    // Allocate memory for all centroids on all processes
    allCentroids = (struct Point *)malloc(numPoints * sizeof(struct Point));
    if (rank == MASTER)
    {
        for (int i = 0; i < numPoints; i++)
        {
            allCentroids[i] = data[i];
        }
    }

    // Calculate workload distribution
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    int *byte_counts = (int *)malloc(size * sizeof(int));
    int *byte_displs = (int *)malloc(size * sizeof(int));

    int base_count = numPoints / size;
    int remainder = numPoints % size;

    displs[0] = 0;
    byte_displs[0] = 0;
    for (int i = 0; i < size; i++)
    {
        sendcounts[i] = base_count + (i < remainder ? 1 : 0);
        byte_counts[i] = sendcounts[i] * sizeof(struct Point);
        if (i > 0)
        {
            displs[i] = displs[i - 1] + sendcounts[i - 1];
            byte_displs[i] = byte_displs[i - 1] + byte_counts[i - 1];
        }
    }

    // Allocate memory for local centroids
    int localCount = sendcounts[rank];
    centroids = (struct Point *)malloc(localCount * sizeof(struct Point));

    while (1)
    {
        iteration++;
        if (iteration > maxIterations)
        {
            if (rank == MASTER)
            {
                printf("Reached maximum iterations.\n");
            }
            break;
        }

        // Broadcast all centroids to all processes
        MPI_Bcast(allCentroids, numPoints * sizeof(struct Point), MPI_BYTE, MASTER, MPI_COMM_WORLD);

        // Allocate memory for new centroids and weights
        struct Point *newCentroids = (struct Point *)malloc(localCount * sizeof(struct Point));
        double *weightsSum = (double *)malloc(localCount * sizeof(double));

        // Calculate new centroids for local portion
        calculateNewCentroids(data, numPoints, allCentroids, displs[rank],
                              displs[rank] + localCount, newCentroids, weightsSum, bandwidth);

        // Calculate maximum shift for local portion
        double localMaxShift = calculateMaxShift(allCentroids, newCentroids, displs[rank], localCount);

        // Find global maximum shift
        MPI_Allreduce(&localMaxShift, &globalMaxShift, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // Gather all new centroids to master process
        MPI_Gatherv(newCentroids, localCount * sizeof(struct Point), MPI_BYTE,
                    allCentroids, byte_counts, byte_displs, MPI_BYTE,
                    MASTER, MPI_COMM_WORLD);

        free(newCentroids);
        free(weightsSum);

        if (globalMaxShift < epsilon)
        {
            break;
        }
    }

    // Only master process performs final clustering
    struct Point *uniqueCentroids = NULL;
    if (rank == MASTER)
    {
        *numCentroids = 0;
        bool *isUnique = (bool *)malloc(numPoints * sizeof(bool));
        for (int i = 0; i < numPoints; i++)
        {
            isUnique[i] = true;
        }

        for (int i = 0; i < numPoints; i++)
        {
            if (isUnique[i])
            {
                (*numCentroids)++;
                for (int j = i + 1; j < numPoints; j++)
                {
                    double distance = sqrt(pow(allCentroids[i].x - allCentroids[j].x, 2) +
                                           pow(allCentroids[i].y - allCentroids[j].y, 2));
                    if (distance < epsilon)
                    {
                        isUnique[j] = false;
                    }
                }
            }
        }

        uniqueCentroids = (struct Point *)malloc((*numCentroids) * sizeof(struct Point));
        int index = 0;
        for (int i = 0; i < numPoints; i++)
        {
            if (isUnique[i])
            {
                uniqueCentroids[index++] = allCentroids[i];
            }
        }

        free(isUnique);
        free(allCentroids);
    }

    free(sendcounts);
    free(displs);
    free(byte_counts);
    free(byte_displs);
    free(centroids);

    // Broadcast number of centroids to all processes
    MPI_Bcast(numCentroids, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    // For non-master processes, allocate memory for unique centroids
    if (rank != MASTER)
    {
        uniqueCentroids = (struct Point *)malloc((*numCentroids) * sizeof(struct Point));
    }

    // Broadcast unique centroids to all processes
    MPI_Bcast(uniqueCentroids, (*numCentroids) * sizeof(struct Point), MPI_BYTE, MASTER, MPI_COMM_WORLD);

    return uniqueCentroids;
}

// Function to read data points from a file
struct Point *readDataFromFile(const char *filename, int *numPoints)
{
    struct Point *data = NULL;
    int capacity = 10;
    *numPoints = 0;
    data = (struct Point *)malloc(capacity * sizeof(struct Point));

    FILE *inputFile = fopen(filename, "r");
    if (!inputFile)
    {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        exit(1);
    }

    double x, y;
    while (fscanf(inputFile, "%lf %lf", &x, &y) == 2)
    {
        if (*numPoints >= capacity)
        {
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

// Debug function to print array contents
void printArray(const char *name, int *array, int size, int rank)
{
    printf("[Rank %d] %s: ", rank, name);
    for (int i = 0; i < size; i++)
    {
        printf("%d ", array[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    int rank, size, numPoints = 0;
    double start_time, end_time;
    struct Point *data = NULL;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(processor_name, &name_len);

    char filename[] = "test.txt";

    // Only master process reads data from file
    if (rank == MASTER)
    {
        data = readDataFromFile(filename, &numPoints);
        printf("Read %d points from file.\n", numPoints);
    }

    // Broadcast number of points to all processes
    MPI_Bcast(&numPoints, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    // For non-master processes, allocate memory for data
    if (rank != MASTER)
    {
        data = (struct Point *)malloc(numPoints * sizeof(struct Point));
    }

    // Broadcast data to all processes
    MPI_Bcast(data, numPoints * sizeof(struct Point), MPI_BYTE, MASTER, MPI_COMM_WORLD);

    // Mean Shift parameters
    double bandwidth = 4.0;  // Bandwidth for the Gaussian kernel
    double epsilon = 0.1;    // Convergence threshold
    int maxIterations = 100; // Maximum number of iterations

    // Open file to save results (only master process)
    FILE *fp = NULL;
    if (rank == MASTER)
    {
        fp = fopen("clustering_results_mpi.txt", "w");
        if (!fp)
        {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // Write header to file
        fprintf(fp, "# MPI_Processes Time_Seconds Num_Centroids\n");
    }

    // Perform Mean Shift clustering
    int numCentroids = 0;

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes before starting timing
    start_time = MPI_Wtime();

    struct Point *centroids = meanShiftClustering(data, numPoints, bandwidth, epsilon,
                                                  maxIterations, &numCentroids, rank, size);

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes after clustering
    end_time = MPI_Wtime();

    // Print results (only master process)
    if (rank == MASTER)
    {
        double time_spent = end_time - start_time;

        printf("MPI Processes: %d, Time: %f seconds, Number of Centroids: %d\n",
               size, time_spent, numCentroids);

        // Print all centroids
        for (int i = 0; i < numCentroids; i++)
        {
            printf("Centroid %d: (%f, %f)\n", i, centroids[i].x, centroids[i].y);
        }

        // Save results to file
        fprintf(fp, "%d %f %d\n", size, time_spent, numCentroids);
        printf("Time spent: %f seconds\n", time_spent);
        fclose(fp);
    }

    // Clean up
    free(data);
    free(centroids);

    MPI_Finalize();
    return 0;
}
