#!/bin/bash

# Compile the code
mpicc main.c -o vector_ops -lm

# Clear previous results
echo "# MPI_Processes Time_Seconds Sum" > benchmark_add_result.txt
echo "# MPI_Processes Time_Seconds Product_Sum" > benchmark_mult_result.txt

# Run for 1 to 6 processes
for procs in {1..6}
do
    echo "Running with $procs process(es)..."
    mpirun -np $procs ./vector_ops
done

echo "Benchmarking complete ✅"
