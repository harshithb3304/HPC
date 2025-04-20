#!/bin/bash

# Compile the program
mpicc main.c -o msc -lm

# Create results file
echo "# MPI_Processes Time_Seconds Num_Centroids" > benchmark_results.txt

# Run with different process counts
for procs in 1 2 3 4 5 6
do
    echo "Running with $procs processes..."
    mpirun -np $procs ./msc
    
    # Extract the last line from clustering_results_mpi.txt and append to benchmark_results.txt
    tail -n 1 clustering_results_mpi.txt >> benchmark_results.txt
done

echo "Benchmark completed. Results saved to benchmark_results.txt"
