import numpy as np
import matplotlib.pyplot as plt

# Load data from matrix_add_parallel.txt
matrix_add_data = np.loadtxt("matrix_add_parallel.txt", skiprows=0)

# Extract threads and execution times
threads = matrix_add_data[:, 0]
times = matrix_add_data[:, 1]

# Save data into a dictionary
time_dict = dict(zip(threads, times))

# Extract thread counts and corresponding times
threads_list = list(time_dict.keys())
execution_times = list(time_dict.values())

print("Threads:", threads_list)

# Plot 1: Execution Time vs Threads
plt.figure(figsize=(8, 6))
plt.plot(threads_list, execution_times, color='y', marker='o', label='Matrix Addition')

plt.title("Execution Time vs Threads")
plt.xlabel("Number of Threads")
plt.ylabel("Execution Time (s)")
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: SpeedUp vs Threads
speedup = [execution_times[0] / execution_times[i] for i in range(1, len(threads_list))]
print(speedup)

plt.figure(figsize=(8, 6))
plt.plot(threads_list[1:], speedup, color='g', marker='o', label='Matrix Addition SpeedUp')

plt.title("SpeedUp vs Threads")
plt.xlabel("Number of Threads")
plt.ylabel("SpeedUp")
plt.legend()
plt.grid(True)
plt.show()

# Plot 3: Parallelization Factor vs Threads
parallelization_factor = [
    ((1 /speedup[i-1])-1) / ((1 / threads_list[i]) - 1) for i in range(1, len(threads_list))
]

plt.figure(figsize=(8, 6))
plt.plot(threads_list[1:], parallelization_factor, color='b', marker='o', label='Matrix Addition Parallelization Factor')

plt.title("Parallelization Factor vs Threads")
plt.xlabel("Number of Threads")
plt.ylabel("Parallelization Factor")
plt.legend()
plt.grid(True)
plt.show()
