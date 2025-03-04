import numpy as np

# Number of double precision numbers to generate (e.g., 1.5 million)
num_numbers = 1500000

# Generate numbers in a specified range (e.g., between 1e3 and 1e6) so that each number is of decent magnitude.
numbers = np.random.uniform(low=1e3, high=1e6, size=num_numbers)

# Write the numbers to a text file with high precision formatting
with open("output.txt", "w") as f:
    for num in numbers:
        f.write("{:.10f}\n".format(num))