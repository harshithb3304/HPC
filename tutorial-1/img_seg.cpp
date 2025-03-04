#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Define a 5D point structure (x, y, R, G, B)
struct Point5D {
    double x, y, r, g, b;
};

// Gaussian kernel function
double gaussianKernel(double distance, double bandwidth) {
    return exp(-0.5 * pow(distance / bandwidth, 2));
}

// Mean Shift clustering function
vector<Point5D> meanShiftClustering(const vector<Point5D>& data, double bandwidth, double epsilon = 1e-5, int maxIterations = 100) {
    vector<Point5D> centroids = data; // Initialize centroids with the data points
    int iteration = 0;

    while (true) {
        iteration++;
        if (iteration > maxIterations) {
            cout << "Reached maximum iterations." << endl;
            break;
        }

        double maxShift = 0.0;

        // Compute new centroids
        for (size_t i = 0; i < centroids.size(); i++) {
            Point5D newCentroid = {0.0, 0.0, 0.0, 0.0, 0.0};
            double weightSum = 0.0;

            for (size_t j = 0; j < data.size(); j++) {
                double distance = sqrt(pow(centroids[i].x - data[j].x, 2) +
                                       pow(centroids[i].y - data[j].y, 2) +
                                       pow(centroids[i].r - data[j].r, 2) +
                                       pow(centroids[i].g - data[j].g, 2) +
                                       pow(centroids[i].b - data[j].b, 2));
                double weight = gaussianKernel(distance, bandwidth);

                newCentroid.x += data[j].x * weight;
                newCentroid.y += data[j].y * weight;
                newCentroid.r += data[j].r * weight;
                newCentroid.g += data[j].g * weight;
                newCentroid.b += data[j].b * weight;
                weightSum += weight;
            }

            if (weightSum > 0) {
                newCentroid.x /= weightSum;
                newCentroid.y /= weightSum;
                newCentroid.r /= weightSum;
                newCentroid.g /= weightSum;
                newCentroid.b /= weightSum;
            }

            // Check for convergence
            double shift = sqrt(pow(newCentroid.x - centroids[i].x, 2) +
                                pow(newCentroid.y - centroids[i].y, 2) +
                                pow(newCentroid.r - centroids[i].r, 2) +
                                pow(newCentroid.g - centroids[i].g, 2) +
                                pow(newCentroid.b - centroids[i].b, 2));
            maxShift = max(maxShift, shift);
            centroids[i] = newCentroid;
        }

        cout << "Iteration: " << iteration << ", Max Shift: " << maxShift << endl;

        if (maxShift < epsilon) {
            cout << "Converged after " << iteration << " iterations." << endl;
            break;
        }
    }

    return centroids;
}

// Convert image to 5D points
vector<Point5D> imageToPoints(const Mat& image, int step = 5) {
    vector<Point5D> points;
    for (int y = 0; y < image.rows; y += step) {
        for (int x = 0; x < image.cols; x += step) {
            Vec3b pixel = image.at<Vec3b>(y, x);
            points.push_back({static_cast<double>(x), static_cast<double>(y), static_cast<double>(pixel[2]), static_cast<double>(pixel[1]), static_cast<double>(pixel[0])});
        }
    }
    return points;
}

void segmentImage(const Mat& image, const vector<Point5D>& centroids, Mat& foreground, Mat& background) {
    // Initialize foreground and background images
    foreground = Mat::zeros(image.size(), image.type()); // Black image for foreground
    background = Mat::zeros(image.size(), image.type()); // Black image for background

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            Vec3b pixel = image.at<Vec3b>(y, x);
            Point5D point = {static_cast<double>(x), static_cast<double>(y), static_cast<double>(pixel[2]), static_cast<double>(pixel[1]), static_cast<double>(pixel[0])};

            // Find the closest centroid
            double minDistance = numeric_limits<double>::max();
            int closestCentroidIdx = 0;
            for (size_t i = 0; i < centroids.size(); i++) {
                double distance = sqrt(pow(point.x - centroids[i].x, 2) + pow(point.y - centroids[i].y, 2) +
                                       pow(point.r - centroids[i].r, 2) + pow(point.g - centroids[i].g, 2) + pow(point.b - centroids[i].b, 2));
                if (distance < minDistance) {
                    minDistance = distance;
                    closestCentroidIdx = i;
                }
            }

            // Assign to foreground or background based on the closest centroid
            if (closestCentroidIdx == 0) { // Assume centroid 0 is foreground
                foreground.at<Vec3b>(y, x) = pixel; // Copy pixel to foreground
            } else { // Other centroids are background
                background.at<Vec3b>(y, x) = pixel; // Copy pixel to background
            }
        }
    }
}

// int main() {
//     // Load the image
//     Mat image = imread("image.png");
//     if (image.empty()) {
//         cout << "Could not open or find the image" << endl;
//         return -1;
//     }

//     // Resize image for efficiency
//     resize(image, image, Size(image.cols , image.rows));

//     // Convert image to 5D points
//     vector<Point5D> data = imageToPoints(image);

//     // Mean Shift parameters
//     double bandwidth = 30.0; // Bandwidth for the Gaussian kernel
//     double epsilon = 1e-2;   // Convergence threshold
//     int maxIterations = 50; // Maximum number of iterations

//     // Perform Mean Shift clustering
//     vector<Point5D> centroids = meanShiftClustering(data, bandwidth, epsilon, maxIterations);

//     cout << "Clustering complete. Found " << centroids.size() << " centroids." << endl;

//     return 0;
// }

int main() {
    // Load the image
    Mat image = imread("image.png");
    if (image.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // Convert image to 5D points
    vector<Point5D> data = imageToPoints(image);

    // Mean Shift parameters
    double bandwidth = 30.0; // Bandwidth for the Gaussian kernel
    double epsilon = 1e-5;   // Convergence threshold
    int maxIterations = 100; // Maximum number of iterations

    // Perform Mean Shift clustering
    vector<Point5D> centroids = meanShiftClustering(data, bandwidth, epsilon, maxIterations);

    // Segment the image into foreground and background
    Mat foreground, background;
    segmentImage(image, centroids, foreground, background);

    // Save the foreground and background images to files
    string fgOutputPath = "fg.png";
    string bgOutputPath = "bg.png";
    bool isFgSaved = imwrite(fgOutputPath, foreground);
    bool isBgSaved = imwrite(bgOutputPath, background);

    if (isFgSaved && isBgSaved) {
        cout << "Foreground and background images saved successfully." << endl;
        cout << "Foreground: " << fgOutputPath << endl;
        cout << "Background: " << bgOutputPath << endl;
    } else {
        cout << "Failed to save the images." << endl;
    }

    return 0;
}