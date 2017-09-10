# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as np

def query_integral_image(unsigned int[:,:] integral_image, int size_x, int
                         size_y, position_strategy):
    cdef int x = integral_image.shape[0]
    cdef int y = integral_image.shape[1]
    cdef unsigned int area, i, j, counter
    cdef unsigned int[:,:] candidates = np.zeros((max((x - size_x) * (y - size_y), 0), 2), dtype=np.int64)

    # gather possible locations
    #candidates = []
    counter = -1
    for i in range(x - size_x):
        for j in range(y - size_y):
            area = integral_image[i, j] + integral_image[i + size_x, j + size_y]
            area -= integral_image[i + size_x, j] + integral_image[i, j + size_y]
            if not area:
                counter +=1
                candidates[counter,0] = i
                candidates[counter,1] = j
    if counter == 0:
        # no room left
        return None
    # pick a location
    return position_strategy(candidates[0:counter+1,:])
