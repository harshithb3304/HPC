#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

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

// Mean Shift clustering function
struct Point *meanShiftClustering(const struct Point *data, int numPoints, double bandwidth, double epsilon, int maxIterations, int *numCentroids)
{
    struct Point *centroids = (struct Point *)malloc(numPoints * sizeof(struct Point));
    for (int i = 0; i < numPoints; i++)
    {
        centroids[i] = data[i];
    }

    int iteration = 0;
    while (1)
    {
        iteration++;
        if (iteration > maxIterations)
        {
            printf("Reached maximum iterations.\n");
            break;
        }

        struct Point *newCentroids = (struct Point *)malloc(numPoints * sizeof(struct Point));
        double *weightsSum = (double *)malloc(numPoints * sizeof(double));
        for (int i = 0; i < numPoints; i++)
        {
            newCentroids[i].x = 0.0;
            newCentroids[i].y = 0.0;
            weightsSum[i] = 0.0;
        }

        for (int i = 0; i < numPoints; i++)
        {
            for (int j = 0; j < numPoints; j++)
            {
                double distance = sqrt(pow(centroids[i].x - data[j].x, 2) + pow(centroids[i].y - data[j].y, 2));
                double weight = gaussianKernel(distance, bandwidth);

                newCentroids[i].x += data[j].x * weight;
                newCentroids[i].y += data[j].y * weight;
                weightsSum[i] += weight;
            }

            if (weightsSum[i] > 0)
            {
                newCentroids[i].x /= weightsSum[i];
                newCentroids[i].y /= weightsSum[i];
            }
        }

        double maxShift = 0.0;
        for (int i = 0; i < numPoints; i++)
        {
            double shift = sqrt(pow(newCentroids[i].x - centroids[i].x, 2) + pow(newCentroids[i].y - centroids[i].y, 2));
            if (shift > maxShift)
            {
                maxShift = shift;
            }
        }

        free(centroids);
        centroids = newCentroids;

        if (maxShift < epsilon)
        {
            break;
        }

        free(weightsSum);
    }

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
                double distance = sqrt(pow(centroids[i].x - centroids[j].x, 2) + pow(centroids[i].y - centroids[j].y, 2));
                if (distance < epsilon)
                {
                    isUnique[j] = false;
                }
            }
        }
    }

    struct Point *uniqueCentroids = (struct Point *)malloc((*numCentroids) * sizeof(struct Point));
    int index = 0;
    for (int i = 0; i < numPoints; i++)
    {
        if (isUnique[i])
        {
            uniqueCentroids[index++] = centroids[i];
        }
    }

    free(centroids);
    free(isUnique);

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

int main()
{
    char filename[] = "test.txt";

    // Read data points from the file
    int vecSize = 0;
    struct Point *dataVec = readDataFromFile(filename, &vecSize);
    int numPoints = vecSize;
    struct Point *data = (struct Point *)malloc(numPoints * sizeof(struct Point));
    for (int i = 0; i < numPoints; i++)
    {
        data[i] = dataVec[i];
    }
    free(dataVec);

    // Mean Shift parameters
    double bandwidth = 4;  // Bandwidth for the Gaussian kernel
    double epsilon = 0.1; // Convergence threshold 1e-5
    int maxIterations = 2; // Maximum number of iterations

    // Perform Mean Shift clustering
    int numCentroids = 0;
    struct Point *centroids = meanShiftClustering(data, numPoints, bandwidth, epsilon, maxIterations, &numCentroids);

    // Display the resulting centroids
    printf("Cluster Centroids:\n");
    for (int i = 0; i < numCentroids; i++)
    {
        printf("(%f, %f)\n", centroids[i].x, centroids[i].y);
    }

    printf("Number of Clusters: %d\n", numCentroids);

    // Clean up
    free(data);
    free(centroids);

    return 0;
}
