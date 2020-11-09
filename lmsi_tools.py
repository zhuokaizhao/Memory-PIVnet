# Author: Zhuokai

import os
import json
import math
import torch
import numpy as np


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
