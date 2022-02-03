# The script provides a test function comparing vorticity computed from both CPU and GPU
import time
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

import vorticity
import vorticity_gpu

if __name__=='__main__':

    # if we want to actually plot
    plot = False

    # compute vorticity from 2D velocity field
    # cpu
    # Create a sample v-field
    nrows, ncols = 64, 64
    x, y = list(range(ncols)), list(range(nrows))
    xx, yy = np.meshgrid(x, y)
    udata = vorticity.rankine_vortex_2d(xx, yy, x0=32, y0=32, a=10)
    print('\nCompute curl on CPU')
    cpu_start_time = time.time()
    print('Velocity shape', udata.shape)

    # Compute vorticity
    ## For a 2D v-field, it returns an array with a shape (nrows, ncols, duration)
    ## In this example, udata has a shape (2, nrows, ncols) but one can pass a more general v-field with a shape (2, nrows, ncols, duration)
    ## curl() also works with a 3D v-field input. udata.shape = (3, nrows, ncols, ndepth) or (3, nrows, ncols, ndepth, duration)
    ## In this case, curl returns an array (3, nrows, ncols, ndepth, duration).
    omega = vorticity.curl(udata, xx=xx, yy=yy)
    cpu_end_time = time.time()
    print('Vorticity shape', omega.shape)
    print(f'Curl on CPU took {cpu_end_time-cpu_start_time} seconds')

    # PLOTTING
    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.pcolormesh(xx, yy, omega[..., 0], cmap='bwr', vmin=-3e-3, vmax=3e-3)
        plt.colorbar(label='$\omega_z$')
        plt.gca().set_aspect('equal')
        inc = 4
        plt.gca().quiver(xx[::inc, ::inc], yy[::inc, ::inc], udata[0, ::inc, ::inc], udata[1, ::inc, ::inc], )
        plt.gca().set_xlabel('$x$')
        plt.gca().set_ylabel('$y$')
        plt.show()

    # gpu
    # Create a sample v-field
    nrows, ncols = 64, 64
    x, y = cp.array(range(ncols)), cp.array(range(nrows))
    xx, yy = cp.meshgrid(x, y)
    udata = vorticity_gpu.rankine_vortex_2d(xx, yy, x0=32, y0=32, a=10)
    print('\nCompute curl on GPU')
    gpu_start_time = time.time()
    print('Velocity shape', udata.shape)

    # Compute vorticity
    ## For a 2D v-field, it returns an array with a shape (nrows, ncols, duration)
    ## In this example, udata has a shape (2, nrows, ncols) but one can pass a more general v-field with a shape (2, nrows, ncols, duration)
    ## curl() also works with a 3D v-field icput. udata.shape = (3, nrows, ncols, ndepth) or (3, nrows, ncols, ndepth, duration)
    ## In this case, curl returns an array (3, nrows, ncols, ndepth, duration).
    omega_gpu = vorticity_gpu.curl(udata, xx=xx, yy=yy)
    gpu_end_time = time.time()
    print('Vorticity shape', omega_gpu.shape)
    print(f'Curl on GPU took {gpu_end_time-gpu_start_time} seconds\n')

    # confirm if both arrays are the same
    # if not np.array_equiv(omega, cp.asnumpy(omega_gpu)):
    #     print(omega.shape)
    #     print(omega)
    #     print(cp.asnumpy(omega_gpu).shape)
    #     print(cp.asnumpy(omega_gpu))
    #     print(omega-cp.asnumpy(omega_gpu))
    #     raise Exception(f'Results from CPU and GPU are not the same')

    # PLOTTING
    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.pcolormesh(cp.asnumpy(xx), cp.asnumpy(yy), cp.asnumpy(omega[..., 0]), cmap='bwr', vmin=-3e-3, vmax=3e-3)
        plt.colorbar(label='$\omega_z$')
        plt.gca().set_aspect('equal')
        inc = 4
        plt.gca().quiver(cp.asnumpy(xx[::inc, ::inc]),
                            cp.asnumpy(yy[::inc, ::inc]),
                            cp.asnumpy(udata[0, ::inc, ::inc]),
                            cp.asnumpy(udata[1, ::inc, ::inc]))
        plt.gca().set_xlabel('$x$')
        plt.gca().set_ylabel('$y$')
        plt.show()