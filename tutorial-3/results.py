import numpy as np
import matplotlib.pyplot as plt

# Load data from addition.txt and multiplication.txt
addition_data = np.loadtxt("addition.txt", skiprows=0)
multiplication_data = np.loadtxt("multiplication.txt", skiprows=0)

# Columns: 1st = Threads, 2nd = Time
threads_add = addition_data[:, 0]
times_add = addition_data[:, 1]

threads_mul = multiplication_data[:, 0]
times_mul = multiplication_data[:, 1]

# # Plot: Threads vs Time for Vector Addition with annotations
# plt.figure(figsize=(8, 6))
# plt.plot(threads_add, times_add, marker='o', linestyle='-', color='b', label="Vector Addition")
# for x, y in zip(threads_add, times_add):
#     plt.annotate(f"({int(x)},{y:.4f})", (x, y), textcoords="offset points", xytext=(5, 5))
# plt.xlabel("Number of Threads")
# plt.ylabel("Time (seconds)")
# plt.title("Vector Addition: Threads vs Time")
# plt.grid(True)
# plt.legend()
# plt.savefig("addition_threads_vs_time_annotated.png")
# plt.show()

# # Plot: Threads vs Time for Vector Multiplication with annotations
# plt.figure(figsize=(8, 6))
# plt.plot(threads_mul, times_mul, marker='o', linestyle='-', color='r', label="Vector Multiplication")
# for x, y in zip(threads_mul, times_mul):
#     plt.annotate(f"({int(x)},{y:.4f})", (x, y), textcoords="offset points", xytext=(5, 5))
# plt.xlabel("Number of Threads")
# plt.ylabel("Time (seconds)")
# plt.title("Vector Multiplication: Threads vs Time")
# plt.grid(True)
# plt.legend()
# plt.savefig("multiplication_threads_vs_time_annotated.png")
# plt.show()

# Compute speedup: speedup = T(1 thread) / T(n threads)
T1_add = times_add[threads_add == 1][0]
speedup_add = T1_add / times_add[1:]
print(speedup_add)

T1_mul = times_mul[threads_mul == 1][0]
speedup_mul = T1_mul / times_mul

# Plot: Speedup vs Threads for both methods, with annotations on each point
plt.figure(figsize=(8, 6))
plt.plot(threads_add, speedup_add, marker='o', linestyle='-', color='g', label="Addition Speedup")
for x, y in zip(threads_add, speedup_add):
    plt.annotate(f"({int(x)},{y:.3f})", (x, y), textcoords="offset points", xytext=(5, 5))
plt.plot(threads_mul, speedup_mul, marker='o', linestyle='-', color='m', label="Multiplication Speedup")
for x, y in zip(threads_mul, speedup_mul):
    plt.annotate(f"({int(x)},{y:.3f})", (x, y), textcoords="offset points", xytext=(5, -15))
plt.xlabel("Number of Threads")
plt.ylabel("Speedup (T(1)/T(n))")
plt.title("Speedup vs Number of Threads")
plt.legend()
plt.grid(True)
plt.savefig("speedup_vs_threads_annotated.png")
plt.show()

# Calculate parallelization factor for all thread counts
parallelization_factor_add = [
    (1 / speedup_add[i] - 1) / (1 / threads_add[i] - 1) if threads_add[i] != 1 else 0.0
    for i in range(len(threads_add))
]
parallelization_factor_mul = [
    (1 / speedup_mul[i] - 1) / (1 / threads_mul[i] - 1) if threads_mul[i] != 1 else 0.0
    for i in range(len(threads_mul))
]

# Plot: Parallelization Factor vs Threads
plt.figure(figsize=(8, 6))
plt.plot(threads_add, parallelization_factor_add, marker='o', linestyle='-', color='g', label="Addition Parallelization Factor")
for x, y in zip(threads_add, parallelization_factor_add):
    plt.annotate(f"({int(x)},{y:.3f})", (x, y), textcoords="offset points", xytext=(5, 5))
plt.plot(threads_mul, parallelization_factor_mul, marker='o', linestyle='-', color='m', label="Multiplication Parallelization Factor")
for x, y in zip(threads_mul, parallelization_factor_mul):
    plt.annotate(f"({int(x)},{y:.3f})", (x, y), textcoords="offset points", xytext=(5, -15))
plt.xlabel("Number of Threads")
plt.ylabel("Parallelization Factor")
plt.title("Parallelization Factor vs Number of Threads")
plt.legend()
plt.grid(True)
plt.savefig("parallelization_factor_vs_threads_annotated.png")
plt.show()