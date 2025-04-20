import numpy as np
# random number of length x * x and x is user defiend and write it in a file given the file name all are store in a sequential order
def write_random_number_in_file(x, filename):
    random_number = np.random.rand(x, x)
    np.savetxt(filename, random_number, fmt='%f')
    return


def write_random_number_in_file2(x, filename):
    random_number = np.random.rand(x, x)
    np.savetxt(filename, random_number, fmt='%f')
    return

write_random_number_in_file(2000, 'attention_weights.txt')
write_random_number_in_file(2000, 'weights.txt')
write_random_number_in_file2(2000, 'attention_bias.txt')
write_random_number_in_file2(2000, 'node_bias.txt')

