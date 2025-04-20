import matplotlib.pyplot as plt
import numpy as np

# Read benchmark data
data = []
with open('benchmark_results.txt', 'r') as f:
    # Skip header line
    next(f)
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            processes = int(parts[0])
            time = float(parts[1])
            centroids = int(parts[2])
            data.append((processes, time, centroids))

# Extract data for plotting
processes = [item[0] for item in data]
times = [item[1] for item in data]

# Calculate speedup (relative to single process)
base_time = times[0]
speedups = [base_time / time for time in times]

# Plot Time vs Processes
plt.figure(figsize=(10, 6))
plt.plot(processes, times, 'o-', linewidth=2, markersize=8)
plt.xlabel('Number of MPI Processes', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)
plt.title('Execution Time vs Number of MPI Processes', fontsize=14)
plt.grid(True)
plt.xticks(processes)
plt.savefig('time_vs_processes.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot Speedup vs Processes
plt.figure(figsize=(10, 6))
plt.plot(processes, speedups, 'o-', linewidth=2, markersize=8, color='green')
plt.xlabel('Number of MPI Processes', fontsize=12)
plt.ylabel('Speedup', fontsize=12)
plt.title('Speedup vs Number of MPI Processes', fontsize=14)
plt.grid(True)
plt.xticks(processes)
plt.savefig('speedup_vs_processes.png', dpi=300, bbox_inches='tight')
plt.close()

# Print some statistics for the report
print(f"Single Process Execution Time: {times[0]:.6f} seconds")
print(f"Maximum Speedup (with {processes[-1]} processes): {speedups[-1]:.2f}")
