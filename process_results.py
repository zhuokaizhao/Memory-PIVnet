# the script serves to process and visualize the results from various methods

import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

import plot


# helper function that computes energy from the velocity field
def get_energy(udata):
    """
    Returns energy(\vec{x}, t) of udata
    ... Assumes udata is equally spaced data.

    Parameters
    ----------
    udata: nd array

    Returns
    -------
    energy: nd array
        energy
    """
    shape = udata.shape  # shape=(dim, nrows, ncols, nstacks) if nstacks=0, shape=(dim, nrows, ncols)
    dim = udata.shape[0]
    energy = np.zeros(shape[1:])
    for d in range(dim):
        energy += udata[d, ...] ** 2
    energy /= 2.
    return energy

# list of methods
methods = ['memory-piv-net', 'pyramid', 'cross_correlation']

# corresponding data path or directory
result_paths = ['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/amnesia_memory/50000_seeds/no_pe/time_span_5/blend_vel_field/',
                '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/pyramid/isotropic1024coarse_x_1_1024_y_1_1024_z_662_762_t_1_5024_tstride_20_pyramid.h5',
                '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/cross_correlation/isotropic1024coarse_x_1_1024_y_1_1024_z_662_762_t_1_5024_tstride_20_standard.h5']

# start and end time (both inclusive)
time_range = [0, 248]

# loaded velocity fields
true_velocity = []
memory_velocity = []
pyramid_velocity = []
cc_velocity = []

# load results from each method
for i, cur_method in enumerate(methods):
    if cur_method == 'memory-piv-net':
        # load the velocity fields of the specified time range
        for t in range(time_range[0], time_range[1]+1):
            cur_path = os.path.join(result_paths[i], f'test_velocity_blend_{t}.npz')
            memory_velocity.append(np.load(cur_path)['velocity'])

        memory_velocity = np.array(memory_velocity)

    # for pyramid and standar methods
    elif cur_method == 'pyramid' or cur_method == 'cross_correlation':
        cur_path = result_paths[i]
        with h5py.File(cur_path, mode='r') as f:
            # print('The h5 file contains ', list(f.keys()))
            xx, yy = f['x'][...], f['y'][...]
            ux, uy = f['ux'][...], f['uy'][...]

            # if reverse_y:
            #     yy = yy[::-1, ...]
            #     uy *= -1
            # print('Resolution of the output', xx.shape)
            # print('Shape of ux or uy (nrows, ncols, duration)', ux.shape)
        if cur_method == 'pyramid':
            pyramid_velocity = np.stack((ux, uy))
            pyramid_velocity = np.moveaxis(pyramid_velocity, [0, 1, 2, 3], [3, 1, 2, 0])
        else:
            cc_velocity = np.stack((ux, uy))
            cc_velocity = np.moveaxis(cc_velocity, [0, 1, 2, 3], [3, 1, 2, 0])


# print all the shapes
print(f'Loaded memory-piv-net output has shape {memory_velocity.shape}')
print(f'Loaded pyramid output has shape {pyramid_velocity.shape}')
print(f'Loaded cross-correlation output has shape {cc_velocity.shape}')
