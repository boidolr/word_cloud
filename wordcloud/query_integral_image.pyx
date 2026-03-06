# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
import array
import numpy as np


def query_integral_image(unsigned int[:,:] integral_image, int size_x, int
                         size_y, random_state):
    cdef int x = integral_image.shape[0]
    cdef int y = integral_image.shape[1]
    cdef int area, i, j
    cdef int count = 0
    cdef int result_i = 0, result_j = 0

    # Use reservoir sampling to avoid double scan
    # This selects a random valid position in a single pass
    for i in xrange(x - size_x):
        for j in xrange(y - size_y):
            area = integral_image[i, j] + integral_image[i + size_x, j + size_y]
            area -= integral_image[i + size_x, j] + integral_image[i, j + size_y]
            if not area:
                count += 1
                # Reservoir sampling: replace with probability 1/count
                if random_state.randint(0, count) == 0:
                    result_i = i
                    result_j = j

    if not count:
        return None
    return result_i, result_j
