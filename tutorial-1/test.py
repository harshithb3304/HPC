# def analyze_edgelist():
#     max_vertex = 0
#     edge_count = 0
    
#     with open('edgelist.txt', 'r') as f:
#         for line in f:
#             v1, v2, cap = map(int, line.strip().split())
#             max_vertex = max(max_vertex, v1, v2)
#             edge_count += 1
    
#     print(f"Total edges: {edge_count}")
#     print(f"Max vertex number: {max_vertex}")
#     print(f"Suggested parameters:")
#     print(f"V = {max_vertex + 1}")  # +1 because vertices are 0-based
#     print(f"E = {edge_count}")
#     print(f"Suggested source = 1")   # typically lowest numbered vertex
#     print(f"Suggested sink = {max_vertex}") # typically highest numbered vertex
# analyze_edgelist()

# import random

# # Function to generate test cases and save to a text file
# def generate_test_cases(filename, num_points, num_clusters):
#     # Define cluster centers for moderate complexity
#     cluster_centers = [
#         (2.0, 2.0),
#         (5.0, 5.0),
#         (8.0, 2.0),
#         (2.0, 8.0),
#         (5.0, 5.0),  # Overlapping cluster
#         (7.0, 7.0),
#         (1.0, 5.0)
#     ]
    
#     with open(filename, "w") as f:
#         f.write(f"{num_points} {num_clusters}\n")  # First line: num_points and num_clusters
        
#         # Generate random points around cluster centers
#         for _ in range(num_points):
#             cluster_idx = random.randint(0, num_clusters - 1)
#             center_x, center_y = cluster_centers[cluster_idx]
#             x = center_x + (random.uniform(-0.5, 0.5))  # Add moderate noise
#             y = center_y + (random.uniform(-0.5, 0.5))
#             f.write(f"{x} {y}\n")

# # Parameters
# filename = "test_cases.txt"  # Output file
# num_points = 10000           # Total number of points
# num_clusters = 7             # Number of clusters

# # Generate test cases
# generate_test_cases(filename, num_points, num_clusters)
# print(f"Test cases saved to {filename}")


import numpy as np

# Set parameters
num_clusters = 6
points_per_cluster = 1666  # ~10,000 points
clusters = np.random.rand(num_clusters, 2) * 10  # Random cluster centers

# Generate data points
data = np.vstack([
    np.random.randn(points_per_cluster, 2) * 0.5 + cluster
    for cluster in clusters
])

# Save to test.txt
filename = "test.txt"
np.savetxt(filename, data, fmt="%.5f", delimiter=" ")

print(f"Generated {len(data)} points across {num_clusters} clusters and saved to {filename}.")
