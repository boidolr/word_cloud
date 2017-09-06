# cython: boundscheck=False
# cython: wraparound=False
import array
import sys
import numpy as np


if sys.version < '3':
    range = xrange


def query_integral_image(unsigned int[:,:] integral_image, int size_x, int
                         size_y, position_strategy):
    cdef int x = integral_image.shape[0]
    cdef int y = integral_image.shape[1]
    cdef int area, i, j

    # gather possible locations
    candidates = []
    for i in range(x - size_x):
        for j in range(y - size_y):
            area = integral_image[i, j] + integral_image[i + size_x, j + size_y]
            area -= integral_image[i + size_x, j] + integral_image[i, j + size_y]
            if not area:
                candidates.append((i,j))
    if not len(candidates):
        # no room left
        return None
    # pick a location
    return position_strategy(candidates)
