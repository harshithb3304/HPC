#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <ctime>

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
vector<Point> meanShiftClustering(const vector<Point>& data, double bandwidth, double epsilon = 1e-5, int maxIterations = 100) {
    vector<Point> centroids = data; // Initialize centroids with the data points
    int iteration = 0;

    while (true) {
        iteration++;
        if (iteration > maxIterations) {
            cout << "Reached maximum iterations." << endl;
            break;
        }

        vector<Point> newCentroids(data.size(), {0.0, 0.0});
        vector<double> weightsSum(data.size(), 0.0);

        // For each centroid, compute the weighted mean
        for (size_t i = 0; i < centroids.size(); i++) {
            for (size_t j = 0; j < data.size(); j++) {
                double distance = sqrt(pow(centroids[i].x - data[j].x, 2) + pow(centroids[i].y - data[j].y, 2));
                double weight = gaussianKernel(distance, bandwidth);

                newCentroids[i].x += data[j].x * weight;
                newCentroids[i].y += data[j].y * weight;
                weightsSum[i] += weight;
            }

            // Normalize the new centroid
            if (weightsSum[i] > 0) {
                newCentroids[i].x /= weightsSum[i];
                newCentroids[i].y /= weightsSum[i];
            }
        }

        // Check for convergence
        double maxShift = 0.0;
        for (size_t i = 0; i < centroids.size(); i++) {
            double shift = sqrt(pow(newCentroids[i].x - centroids[i].x, 2) + pow(newCentroids[i].y - centroids[i].y, 2));
            if (shift > maxShift) {
                maxShift = shift;
            }
        }

        if (maxShift < epsilon) {
            break; // Convergence reached
        }

        centroids = newCentroids;
    }

    return centroids;
}

// Function to generate a moderately complex dataset
vector<Point> generateModeratelyComplexDataset(int numPoints, int numClusters) {
    vector<Point> data;
    srand(time(0));

    // Define cluster centers
    vector<Point> clusterCenters = {
        {2.0, 2.0},
        {5.0, 5.0},
        {8.0, 2.0},
        {2.0, 8.0},
        {5.0, 5.0}, // Overlapping cluster
        {7.0, 7.0},
        {1.0, 5.0}
    };

    // Generate points around each cluster center
    for (int i = 0; i < numPoints; i++) {
        int clusterIdx = rand() % numClusters;
        double x = clusterCenters[clusterIdx].x + (rand() % 1000 - 500) / 500.0; // Add moderate noise
        double y = clusterCenters[clusterIdx].y + (rand() % 1000 - 500) / 500.0; // Add moderate noise
        data.push_back({x, y});
    }

    return data;
}

int main() {
    // Generate a moderately complex dataset
    int numPoints = 2000;    // Number of data points
    int numClusters = 7;     // Number of clusters
    vector<Point> data = generateModeratelyComplexDataset(numPoints, numClusters);

    // Mean Shift parameters
    double bandwidth = 3; // Bandwidth for the Gaussian kernel
    double epsilon = 1e-5;  // Convergence threshold
    int maxIterations = 100; // Maximum number of iterations

    // Perform Mean Shift clustering
    vector<Point> centroids = meanShiftClustering(data, bandwidth, epsilon, maxIterations);

    // Display the resulting centroids
    cout << "Cluster Centroids:" << endl;
    for (const auto& centroid : centroids) {
        cout << "(" << centroid.x << ", " << centroid.y << ")" << endl;
    }

    cout << "Number of Clusters: " << centroids.size() << endl;

    return 0;
}