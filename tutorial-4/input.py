import numpy as np

# Number of double precision numbers to generate (e.g., 1.5 million)
num_numbers = 15000000

# Generate two arrays of numbers in a specified range (e.g., between 1e3 and 1e6)
numbers1 = np.random.uniform(low=1e3, high=1e6, size=num_numbers)
numbers2 = np.random.uniform(low=1e3, high=1e6, size=num_numbers)

# Write numbers1 to output1.txt with high precision formatting
with open("output1.txt", "w") as f1:
    for num in numbers1:
        f1.write("{:.10f}\n".format(num))

# Write numbers2 to output2.txt with high precision formatting
with open("output2.txt", "w") as f2:
    for num in numbers2:
        f2.write("{:.10f}\n".format(num))