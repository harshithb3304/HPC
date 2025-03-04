import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from addition.txt and multiplication.txt
addition_data = np.loadtxt("dot_product.txt", skiprows=0)
#multiplication_data = np.loadtxt("multiplication.txt", skiprows=0)

# Columns: 1st = Threads, 2nd = Time
threads_add = addition_data[:, 0]
times_add = addition_data[:, 1]


# Save data into dictionaries: thread -> time
dict1 = dict(zip(threads_add, times_add))


threads = list(dict1.keys())
values1 = list(dict1.values())

print(threads)
# Plot 1: Execution Time vs Threads
plt.figure(figsize=(8, 6))
plt.plot(threads, values1, color='y', marker='o', label='Vector Dot Product')

plt.title("Execution Time vs Threads")
plt.xlabel("Number of Threads")
plt.ylabel("Execution Time (s)")
plt.legend()
plt.grid(True)
plt.show()  # Show Plot 1

# Plot 2: SpeedUp vs Threads
speedup_addition = [values1[0] / values1[i] for i in range(1, len(threads))]

plt.figure(figsize=(8, 6))
plt.plot(threads[1:], speedup_addition, color='g', marker='o', label='Vector Dot Product')

plt.title("SpeedUp vs Threads")
plt.xlabel("Number of Threads")
plt.ylabel("SpeedUp")
plt.legend()
plt.grid(True)
plt.show()  # Show Plot 2

# Plot 3: Parallelisation Factor vs Threads
parallelisation_factor_addition = [
    ((1 / speedup_addition[i - 1]) - 1) / ((1 / threads[i]) - 1) for i in range(1, len(threads))
]


plt.figure(figsize=(8, 6))
plt.plot(threads[1:], parallelisation_factor_addition, color='g', marker='o', label='Vector Dot Product')

plt.title("Parallelisation Factor vs Threads")
plt.xlabel("Number of Threads")
plt.ylabel("Parallelisation Factor")
plt.legend()
plt.grid(True)
plt.show()  # Show Plot 3