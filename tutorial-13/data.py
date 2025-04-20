import numpy as np
import random

# Set parameters
num_points = 1000000  # 1 million points
num_clusters = 6  # Number of clusters

# Random cluster centers within a 100x100 grid
cluster_centers = np.random.rand(num_clusters, 2) * 100

# Generate points around each cluster
points = []
for center in cluster_centers:
    x, y = center
    cluster_points = np.random.randn(num_points // num_clusters, 2) * 2 + [x, y]
    points.append(cluster_points)

# Combine all cluster points
data = np.vstack(points)

# Save to test.txt in the format required by main.c
filename = "test.txt"
with open(filename, "w") as f:
    for point in data:
        f.write(f"{point[0]:.6f} {point[1]:.6f}\n")

print(f"Generated {len(data)} points across {num_clusters} clusters and saved to {filename}.")