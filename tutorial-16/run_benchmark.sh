#!/bin/bash

# Compile the program
mpicc main.c -o vdp

# Create results file
echo "# MPI_Processes Time_Seconds" > benchmark_results.txt

# Run with different process counts
for procs in 1 2 3 4 5 6
do
    echo "Running with $procs processes..."
    mpirun -np $procs ./vdp
    
    # Extract the last line from vdp_results.txt and append to benchmark_results.txt
    tail -n 1 vdp_results.txt >> benchmark_results.txt
done

echo "Benchmark completed. Results saved to benchmark_results.txt"
