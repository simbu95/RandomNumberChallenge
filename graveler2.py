""" References 
https://numpy.org/doc/stable/reference/random/index.html
https://stackoverflow.com/questions/42916330/efficiently-count-zero-elements-in-numpy-array

Top answer in stack overflow also mentions using JAX to accelerate with GPU, but that requires a lot of setup. 
Potentially 1000x faster. 

"""
import numpy as np
import time


LOOPS = int(1e3)
BATCH = int(1e6)  # More then this uses too much memory, and slows things down by loading out of RAM, or causes errors. 

rng = np.random.default_rng()
max_zeros = 0
s = time.time()

for i in range(LOOPS):
    randoms = rng.integers(0, 4, size=(231, BATCH))  # Generate 231 random numbers, in groups of Batch.
    non_zeros = np.count_nonzero(randoms, axis=0)  # For each row, count all the non-zero values (this is fast for numpy to do)

    # zeros = np.count_nonzero(randoms==0, axis=0)  # Alternative that counts number that meet condition of == 0. This seems to be the same speed.

    most_zeros = 231 - np.min(non_zeros)  # Find the minimum number of non zeros (or most zeros), and subtract from 231 to find number of zeros.

    if most_zeros > max_zeros:
        max_zeros = most_zeros

    print(f'Iteration: {i}, Total Time: {time.time() - s} seconds, Most this iteration: {most_zeros}, Current Max: {max_zeros}')

print(f'{max_zeros} found after {LOOPS * BATCH} iterations')