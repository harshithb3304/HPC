import numpy as np
import matplotlib.pyplot as plt


# Load data from reduction.txt and critical.txt

reduction_data = np.loadtxt("reduction.txt", skiprows=0)
critical_data = np.loadtxt("critical.txt", skiprows=0)

# Columns: 1st = Threads, 2nd = Sum, 3rd = Time
threads_red = reduction_data[:, 0]
times_red   = reduction_data[:, 2]

threads_crit = critical_data[:, 0]
times_crit   = critical_data[:, 2]


# Plot: Threads vs Time for Reduction and Critical methods (separately)

# Plot for Reduction with annotations
plt.figure(figsize=(8, 6))
plt.plot(threads_red, times_red, marker='o', linestyle='-', color='b', label="Reduction")
for x, y in zip(threads_red, times_red):
    plt.annotate(f"({int(x)},{y:.4f})", (x, y), textcoords="offset points", xytext=(5,5))
plt.xlabel("Number of Threads")
plt.ylabel("Time (seconds)")
plt.title("Reduction: Threads vs Time")
plt.grid(True)
plt.legend()
plt.savefig("reduction_threads_vs_time_annotated.png")
plt.show()

# Plot for Critical Section with annotations
plt.figure(figsize=(8, 6))
plt.plot(threads_crit, times_crit, marker='o', linestyle='-', color='r', label="Critical")
for x, y in zip(threads_crit, times_crit):
    plt.annotate(f"({int(x)},{y:.4f})", (x, y), textcoords="offset points", xytext=(5,5))
plt.xlabel("Number of Threads")
plt.ylabel("Time (seconds)")
plt.title("Critical: Threads vs Time")
plt.grid(True)
plt.legend()
plt.savefig("critical_threads_vs_time_annotated.png")
plt.show()


# Plot: Speedup vs Threads for both methods, with annotations on each point

# Compute speedup: speedup = T(1 thread)/T(n threads)
T1_red = times_red[threads_red == 1][0]
speedup_red = T1_red / times_red

T1_crit = times_crit[threads_crit == 1][0]
speedup_crit = T1_crit / times_crit

plt.figure(figsize=(8, 6))
plt.plot(threads_red, speedup_red, marker='o', linestyle='-', color='g', label="Reduction Speedup")
for x, y in zip(threads_red, speedup_red):
    plt.annotate(f"({int(x)},{y:.3f})", (x, y), textcoords="offset points", xytext=(5,5))
plt.plot(threads_crit, speedup_crit, marker='o', linestyle='-', color='m', label="Critical Speedup")
for x, y in zip(threads_crit, speedup_crit):
    plt.annotate(f"({int(x)},{y:.3f})", (x, y), textcoords="offset points", xytext=(5,-15))
plt.xlabel("Number of Threads")
plt.ylabel("Speedup (T(1)/T(n))")
plt.title("Speedup vs Number of Threads")
plt.legend()
plt.grid(True)
plt.savefig("speedup_vs_threads_annotated.png")
plt.show()


# Estimate the parallelization fraction using Amdahl's Law:
# Amdahl's Law: Speedup(n) = 1 / [ (1 - p) + p/n ]
# Rearranged: p = n*(S - 1) / [S*(n - 1)]
# We'll use the speedup at the maximum thread count for each method.


# For Reduction data, find the index of the minimum time (best performance)
min_index_red = np.argmin(times_red)
T_opt_red = threads_red[min_index_red]
S_opt_red = times_red[threads_red==1][0] / times_red[min_index_red]  # speedup: T1_time / T_opt_time
p_red = (T_opt_red * (S_opt_red - 1)) / (S_opt_red * (T_opt_red - 1))
print("Best Reduction performance at %d threads (Speedup = %.3f)" % (T_opt_red, S_opt_red))
print("Estimated Parallelization Fraction (Reduction): p = {:.3f}".format(p_red))

# For Critical data, find the index of the minimum time
min_index_crit = np.argmin(times_crit)
T_opt_crit = threads_crit[min_index_crit]
# If the best performance is at 1 thread, speedup is 1 and p will be 0
if T_opt_crit == 1:
    S_opt_crit = 1.0
    p_crit = 0.0
else:
    S_opt_crit = times_crit[threads_crit==1][0] / times_crit[min_index_crit]
    p_crit = (T_opt_crit * (S_opt_crit - 1)) / (S_opt_crit * (T_opt_crit - 1))
print("Best Critical performance at %d threads (Speedup = %.3f)" % (T_opt_crit, S_opt_crit))
print("Estimated Parallelization Fraction (Critical): p = {:.3f}".format(p_crit))

# Explanation:
# The parallelization fraction 'p' is the portion of the computation that can be 
# executed in parallel. For example, if p = 0.95, 95% of the code can run in parallel
# while 5% of the code remains serial, which limits the maximum achievable speedup.
# A higher p means better scalability. Here we estimate p separately for reduction
# and critical implementations based on Amdahl's law.
