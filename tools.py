# Author: Zhuokai

import numpy as np

# lerp
def lerp(pos, in_min, in_max, out_min, out_max):

    # an array of numbers of be lerped
    if type(pos) is np.ndarray:
        alpha = np.true_divide(np.subtract(pos, in_min), float(in_max - in_min))
        new_pos = np.multiply(np.subtract(1.0, alpha), float(out_min)) + np.multiply(alpha, float(out_max))
    # a float number
    else:
        alpha = float(pos - in_min) / float(in_max - in_min)
        new_pos = float(1.0 - alpha) * float(out_min) + alpha * float(out_max)

    return new_pos

# convert a n*n*2 array, which is n*n array of 2D vectors into polar coordinates (r, theta)
def cart2pol(array_cart):
    dim = array_cart.shape
    if len(dim) != 3:
        raise Exception(f'Unsupporting matrix dimension {dim}')
    elif dim[2] != 2:
        raise Exception(f'Only 2D vector supported')

    # output array
    array_pol = np.zeros(dim)
    array_pol[:, :, 0] = np.sqrt(array_cart[:, :, 0]**2 + array_cart[:, :, 1]**2)
    array_pol[:, :, 1] = np.arctan2(array_cart[:, :, 1], array_cart[:, :, 0])

    return array_pol



