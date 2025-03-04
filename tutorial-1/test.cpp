#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <likwid.h>  // Include LIKWID header

using namespace std;

// Define a 2D point structure
struct Point {
    double x, y;
};

// Gaussian kernel function
double gaussianKernel(double distance, double bandwidth) {
    return exp(-0.5 * pow(distance / bandwidth, 2));
}

// Mean Shift clustering function
Point* meanShiftClustering(const Point* data, int numPoints, double bandwidth, double epsilon, int maxIterations, int& numCentroids) {
    // Initialize centroids with the data points
    Point* centroids = new Point[numPoints];
    for (int i = 0; i < numPoints; i++) {
        centroids[i] = data[i];
    }

    int iteration = 0;
    LIKWID_MARKER_START("MeanShift"); // Start tracking Mean Shift clustering
    while (true) {
        iteration++;
        if (iteration > maxIterations) {
            cout << "Reached maximum iterations." << endl;
            break;
        }

        Point* newCentroids = new Point[numPoints];
        double* weightsSum = new double[numPoints];
        for (int i = 0; i < numPoints; i++) {
            newCentroids[i] = {0.0, 0.0};
            weightsSum[i] = 0.0;
        }

        // Weighted mean calculation
        for (int i = 0; i < numPoints; i++) {
            for (int j = 0; j < numPoints; j++) {
                double distance = sqrt(pow(centroids[i].x - data[j].x, 2) + pow(centroids[i].y - data[j].y, 2));
                double weight = gaussianKernel(distance, bandwidth);

                newCentroids[i].x += data[j].x * weight;
                newCentroids[i].y += data[j].y * weight;
                weightsSum[i] += weight;
            }

            if (weightsSum[i] > 0) {
                newCentroids[i].x /= weightsSum[i];
                newCentroids[i].y /= weightsSum[i];
            }
        }

        double maxShift = 0.0;
        for (int i = 0; i < numPoints; i++) {
            double shift = sqrt(pow(newCentroids[i].x - centroids[i].x, 2) + pow(newCentroids[i].y - centroids[i].y, 2));
            if (shift > maxShift) {
                maxShift = shift;
            }
        }

        delete[] centroids;
        centroids = newCentroids;

        if (maxShift < epsilon) {
            break;
        }

        delete[] weightsSum;
    }
    LIKWID_MARKER_STOP("MeanShift"); // Stop tracking Mean Shift clustering

    // Count unique centroids
    numCentroids = 0;
    bool* isUnique = new bool[numPoints];
    for (int i = 0; i < numPoints; i++) {
        isUnique[i] = true;
    }

    for (int i = 0; i < numPoints; i++) {
        if (isUnique[i]) {
            numCentroids++;
            for (int j = i + 1; j < numPoints; j++) {
                double distance = sqrt(pow(centroids[i].x - centroids[j].x, 2) + pow(centroids[i].y - centroids[j].y, 2));
                if (distance < epsilon) {
                    isUnique[j] = false;
                }
            }
        }
    }

    Point* uniqueCentroids = new Point[numCentroids];
    int index = 0;
    for (int i = 0; i < numPoints; i++) {
        if (isUnique[i]) {
            uniqueCentroids[index++] = centroids[i];
        }
    }

    delete[] centroids;
    delete[] isUnique;

    return uniqueCentroids;
}

// Function to generate a moderately complex dataset
Point* generateModeratelyComplexDataset(int numPoints, int numClusters) {
    Point* data = new Point[numPoints];
    srand(time(0));

    Point clusterCenters[] = {
        {2.0, 2.0}, {5.0, 5.0}, {8.0, 2.0}, {2.0, 8.0},
        {5.0, 5.0}, {7.0, 7.0}, {1.0, 5.0}
    };

    LIKWID_MARKER_START("DatasetGeneration"); // Start tracking dataset generation
    for (int i = 0; i < numPoints; i++) {
        int clusterIdx = rand() % numClusters;
        double x = clusterCenters[clusterIdx].x + (rand() % 1000 - 500) / 500.0;
        double y = clusterCenters[clusterIdx].y + (rand() % 1000 - 500) / 500.0;
        data[i] = {x, y};
    }
    LIKWID_MARKER_STOP("DatasetGeneration"); // Stop tracking dataset generation

    return data;
}

int main() {
    LIKWID_MARKER_INIT; // Initialize LIKWID markers

    int numPoints = 2000;
    int numClusters = 7;
    Point* data = generateModeratelyComplexDataset(numPoints, numClusters);

    double bandwidth = 1.5;
    double epsilon = 1e-5;
    int maxIterations = 100;

    int numCentroids = 0;
    Point* centroids = meanShiftClustering(data, numPoints, bandwidth, epsilon, maxIterations, numCentroids);

    cout << "Cluster Centroids:" << endl;
    for (int i = 0; i < numCentroids; i++) {
        cout << "(" << centroids[i].x << ", " << centroids[i].y << ")" << endl;
    }
    cout << "Number of Clusters: " << numCentroids << endl;

    delete[] data;
    delete[] centroids;

    LIKWID_MARKER_CLOSE; // Finalize LIKWID markers

    return 0;
}
