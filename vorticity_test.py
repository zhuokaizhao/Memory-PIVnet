# The script provides a test function comparing vorticity computed from both CPU and GPU
import time
import torch
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

import vorticity_numpy
import vorticity_cupy
import vorticity_torch

if __name__=='__main__':

    # if we want to actually plot
    use_numpy = True
    use_cupy = False
    use_torch = True
    plot = False

    # compute vorticity from 2D velocity field
    # Use numpy on CPU
    if use_numpy:
        # Create a sample v-field
        nrows, ncols = 64, 64
        x, y = list(range(ncols)), list(range(nrows))
        xx, yy = np.meshgrid(x, y)
        udata = vorticity_numpy.rankine_vortex_2d(xx, yy, x0=32, y0=32, a=10)
        print('\nCompute curl using Numpy on CPU')
        np_start_time = time.time()
        print('Velocity shape', udata.shape)

        # Compute vorticity
        omega = vorticity_numpy.curl(udata, xx=xx, yy=yy)
        np_end_time = time.time()
        print('Vorticity shape', omega.shape)
        print(f'Numpy on CPU took {np_end_time-np_start_time} seconds\n')

        # plot
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

    # cupy on GPU
    if use_cupy:
        cp.cuda.Device(0).use()
        # Create a sample v-field
        nrows, ncols = 64, 64
        x, y = cp.array(range(ncols)), cp.array(range(nrows))
        xx, yy = cp.meshgrid(x, y)
        udata = vorticity_cupy.rankine_vortex_2d(xx, yy, x0=32, y0=32, a=10)
        print(f'Compute curl using cupy on GPU {udata.device} out of {cp.cuda.runtime.getDeviceCount()}')
        cp_start_time = time.time()
        print('Velocity shape', udata.shape)

        # Compute vorticity
        omega_cupy = vorticity_cupy.curl(udata, xx=xx, yy=yy)
        cp_end_time = time.time()
        print('Vorticity shape', omega_cupy.shape)
        print(f'Cupy on GPU took {cp_end_time-cp_start_time} seconds\n')

        # confirm if both cupy and torch results are the same as numpy's
        if not np.allclose(omega, cp.asnumpy(omega_cupy)):
            print(omega.shape)
            print(omega)
            print(cp.asnumpy(omega_cupy).shape)
            print(cp.asnumpy(omega_cupy))
            print(omega-cp.asnumpy(omega_cupy))
            raise Exception(f'Result from cupy is not the same as numpy')

        # plot
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

    # Torch on GPU
    if use_torch:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Create a sample v-field
        nrows, ncols = 64, 64
        x, y = torch.tensor(range(ncols)), torch.tensor(range(nrows))
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        xx = xx.double().to(device)
        yy = yy.double().to(device)
        udata = vorticity_torch.rankine_vortex_2d(xx, yy, x0=32, y0=32, a=10)
        print(f'Compute curl using torch on GPU {udata.device} out of {torch.cuda.device_count()}')
        torch_start_time = time.time()
        print('Velocity shape', udata.shape)

        # Compute vorticity
        omega_torch = vorticity_torch.curl(udata, xx=xx, yy=yy)
        torch_end_time = time.time()
        print('Vorticity shape', omega_cupy.shape)
        print(f'Cupy on GPU took {torch_end_time-torch_start_time} seconds\n')

        if not np.allclose(omega, omega_torch.cpu().numpy()):
            print(omega.shape)
            print(omega)
            print(omega_torch.shape)
            print(omega_torch)
            print(omega-omega_torch)
            raise Exception(f'Result from torch is not the same as numpy')

        # plot
        if plot:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.pcolormesh(xx.cpu().numpy(), yy.cpu().numpy(), omega[..., 0].cpu().numpy(), cmap='bwr', vmin=-3e-3, vmax=3e-3)
            plt.colorbar(label='$\omega_z$')
            plt.gca().set_aspect('equal')
            inc = 4
            plt.gca().quiver(xx[::inc, ::inc].cpu().numpy(),
                                yy[::inc, ::inc].cpu().numpy(),
                                udata[0, ::inc, ::inc].cpu().numpy(),
                                udata[1, ::inc, ::inc].cpu().numpy())
            plt.gca().set_xlabel('$x$')
            plt.gca().set_ylabel('$y$')
            plt.show()