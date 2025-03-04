import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from addition.txt and multiplication.txt
addition_data = np.loadtxt("addition.txt", skiprows=0)
multiplication_data = np.loadtxt("multiplication.txt", skiprows=0)

# Columns: 1st = Threads, 2nd = Time
threads_add = addition_data[:, 0]
times_add = addition_data[:, 1]

threads_mul = multiplication_data[:, 0]
times_mul = multiplication_data[:, 1]

# Save data into dictionaries: thread -> time
dict1 = dict(zip(threads_add, times_add))
dict2 = dict(zip(threads_mul, times_mul))
# print(dict1[1])
# Dictionary 1
# dict1 = {
#     1 :0.113448,
# 2 :0.040261,
# 4 :0.024823,
# 6 :0.024243,
# 8 :0.024850,
# 10 :0.024643,
# 12 :0.034257,
# 16 :0.021738,
# 20: 0.024972,
# 32 :0.026484,
# 64 :0.023871,
# }

# # Dictionary 2
# dict2 = {
#     1: 0.105551,
# 2 :0.033286,
# 4 :0.023083,
# 6 :0.027164,
# 8 :0.028865,
# 10: 0.024689,
# 12: 0.029481,
# 16 :0.029761,
# 20 :0.025925,
# 32 :0.022153,
# 64 :0.026152,

# }

threads = list(dict1.keys())
values1 = list(dict1.values())
values2 = list(dict2.values())
print(threads)
# Plot 1: Execution Time vs Threads
plt.figure(figsize=(8, 6))
plt.plot(threads, values1, color='y', marker='o', label='Vector Addition')
plt.plot(threads, values2, color='b', marker='s', label='Vector Multiplication')

plt.title("Execution Time vs Threads")
plt.xlabel("Number of Threads")
plt.ylabel("Execution Time (s)")
plt.legend()
plt.grid(True)
plt.show()  # Show Plot 1

# Plot 2: SpeedUp vs Threads
speedup_addition = [values1[0] / values1[i] for i in range(1, len(threads))]
speedup_multiplication = [values2[0] / values2[i] for i in range(1, len(threads))]

plt.figure(figsize=(8, 6))
plt.plot(threads[1:], speedup_addition, color='g', marker='o', label='Vector Addition')
plt.plot(threads[1:], speedup_multiplication, color='orange', marker='s', label='Vector Multiplication')

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
parallelisation_factor_multiplication = [
    (1 / speedup_multiplication[i - 1] - 1) / (1 / threads[i] - 1) for i in range(1, len(threads))
]

plt.figure(figsize=(8, 6))
plt.plot(threads[1:], parallelisation_factor_addition, color='g', marker='o', label='Vector Addition')
plt.plot(threads[1:], parallelisation_factor_multiplication, color='orange', marker='s', label='Vector Multiplication')

plt.title("Parallelisation Factor vs Threads")
plt.xlabel("Number of Threads")
plt.ylabel("Parallelisation Factor")
plt.legend()
plt.grid(True)
plt.show()  # Show Plot 3