#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>

using namespace std;

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
Point *meanShiftClustering(const Point *data, int numPoints, double bandwidth, double epsilon, int maxIterations, int &numCentroids)
{
    Point *centroids = new Point[numPoints];
    for (int i = 0; i < numPoints; i++)
    {
        centroids[i] = data[i];
    }

    int iteration = 0;
    while (true)
    {
        iteration++;
        if (iteration > maxIterations)
        {
            cout << "Reached maximum iterations." << endl;
            break;
        }

        Point *newCentroids = new Point[numPoints];
        double *weightsSum = new double[numPoints];
        for (int i = 0; i < numPoints; i++)
        {
            newCentroids[i] = {0.0, 0.0};
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

        delete[] centroids;
        centroids = newCentroids;

        if (maxShift < epsilon)
        {
            break;
        }

        delete[] weightsSum;
    }

    numCentroids = 0;
    bool *isUnique = new bool[numPoints];
    for (int i = 0; i < numPoints; i++)
    {
        isUnique[i] = true;
    }

    for (int i = 0; i < numPoints; i++)
    {
        if (isUnique[i])
        {
            numCentroids++;
            // for (int j = i + 1; j < numPoints; j++) {
            //     double distance = sqrt(pow(centroids[i].x - centroids[j].x, 2) + pow(centroids[i].y - centroids[j].y, 2));
            //     if (distance < epsilon) {
            //         isUnique[j] = false;
            //     }
            // }
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

    Point *uniqueCentroids = new Point[numCentroids];
    int index = 0;
    for (int i = 0; i < numPoints; i++)
    {
        if (isUnique[i])
        {
            uniqueCentroids[index++] = centroids[i];
        }
    }

    delete[] centroids;
    delete[] isUnique;

    return uniqueCentroids;
}

// Function to read data points from a file
vector<Point> readDataFromFile(const string &filename)
{
    vector<Point> data;
    ifstream inputFile(filename);

    if (!inputFile)
    {
        cerr << "Error: Could not open file " << filename << endl;
        exit(1);
    }

    double x, y;
    while (inputFile >> x >> y)
    {
        data.push_back({x, y});
    }

    inputFile.close();
    return data;
}

int main()
{
    string filename = "test.txt";

    // Read data points from the file
    vector<Point> dataVec = readDataFromFile(filename);
    int numPoints = dataVec.size();
    Point *data = new Point[numPoints];
    for (int i = 0; i < numPoints; i++)
    {
        data[i] = dataVec[i];
    }

    // Mean Shift parameters
    double bandwidth = 4;  // Bandwidth for the Gaussian kernel
    double epsilon = 0.1; // Convergence threshold 1e-5
    int maxIterations = 2; // Maximum number of iterations

    // Perform Mean Shift clustering
    int numCentroids = 0;
    Point *centroids = meanShiftClustering(data, numPoints, bandwidth, epsilon, maxIterations, numCentroids);

    // Display the resulting centroids
    cout << "Cluster Centroids:" << endl;
    for (int i = 0; i < numCentroids; i++)
    {
        cout << "(" << centroids[i].x << ", " << centroids[i].y << ")" << endl;
    }

    cout << "Number of Clusters: " << numCentroids << endl;

    // Clean up
    delete[] data;
    delete[] centroids;

    return 0;
}
