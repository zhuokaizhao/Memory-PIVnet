# The script provides a test function comparing vorticity computed from both CPU and GPU
import time
import torch
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

import rate_of_strain_numpy
import rate_of_strain_cupy
import rate_of_strain_torch

if __name__=='__main__':

    # if we want to actually plot
    use_numpy = True
    use_cupy = True
    use_torch = True
    plot = False

    # compute vorticity from 2D velocity field
    # Use numpy on CPU
    if use_numpy:
        # Create a sample v-field
        nrows, ncols = 64, 64
        x, y = list(range(ncols)), list(range(nrows))
        xx, yy = np.meshgrid(x, y)
        udata = rate_of_strain_numpy.rankine_vortex_2d(xx, yy, x0=32, y0=32, a=10)
        print('\nComputing vorticity using Numpy on CPU')
        np_start_time = time.time()
        print('Sample data (velocity) shape', udata.shape)

        # Compute vorticity using curl
        omega_1 = rate_of_strain_numpy.curl(udata, xx=xx, yy=yy)
        np_end_time = time.time()
        print('Curl vorticity shape', omega_1.shape, f'took {np_end_time-np_start_time} seconds')

        # Compute vorticity using rate of strain tensor decomposition
        np_start_time = time.time()
        duidxj = rate_of_strain_numpy.get_duidxj_tensor(udata, xx=xx, yy=yy)
        omega_2 = duidxj[..., 1, 0] - duidxj[..., 0, 1]
        np_end_time = time.time()
        print('Decomposed rate-of-strain vorticity shape', omega_2.shape, f'took {np_end_time-np_start_time} seconds')

        # compute divergence
        np_start_time = time.time()
        duidxj = rate_of_strain_numpy.get_duidxj_tensor(udata, xx=xx, yy=yy)
        div = duidxj[..., 0, 0] + duidxj[..., 1, 1]
        np_end_time = time.time()
        print('Divergence shape', div.shape, f'took {np_end_time-np_start_time} seconds')

        # compute shear 1 and 2
        np_start_time = time.time()
        duidxj = rate_of_strain_numpy.get_duidxj_tensor(udata, xx=xx, yy=yy)
        # Shear 1
        shear1 = duidxj[..., 0, 0] - duidxj[..., 1, 1]
        # Shear 2
        shear2 = duidxj[..., 1, 0] + duidxj[..., 0, 1]
        np_end_time = time.time()
        print('Shear 1 and 2 shape', shear1.shape, shear2.shape, f'took {np_end_time-np_start_time} seconds\n')

        # confirm both approaches on vorticity are the same
        if not np.allclose(omega_1, omega_2):
            print(omega_1.shape)
            print(omega_1)
            print(omega_2.shape)
            print(omega_2)
            print(omega_1-omega_2)
            raise Exception(f'Result from both approaches on computing vorticity are not the same')

        # plot
        if plot:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.pcolormesh(xx, yy, omega_1[..., 0], cmap='bwr', vmin=-3e-3, vmax=3e-3)
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
        udata = rate_of_strain_cupy.rankine_vortex_2d(xx, yy, x0=32, y0=32, a=10)
        print(f'Computing curl using cupy on GPU {udata.device} out of {cp.cuda.runtime.getDeviceCount()}')
        cp_start_time = time.time()
        print('Sample data (velocity) shape', udata.shape)

        # Compute vorticity using curl
        omega_cupy_1 = rate_of_strain_cupy.curl(udata, xx=xx, yy=yy)
        cp_end_time = time.time()
        print('Curl vorticity shape', omega_cupy_1.shape, f'took {cp_end_time-cp_start_time} seconds')

        # Compute vorticity using rate of strain tensor decomposition
        cp_start_time = time.time()
        duidxj = rate_of_strain_cupy.get_duidxj_tensor(udata, xx=xx, yy=yy)
        omega_cupy_2 = duidxj[..., 1, 0] - duidxj[..., 0, 1]
        cp_end_time = time.time()
        print('Decomposed rate-of-strain vorticity shape', omega_cupy_2.shape, f'took {cp_end_time-cp_start_time} seconds')

        # compute divergence
        cp_start_time = time.time()
        duidxj = rate_of_strain_cupy.get_duidxj_tensor(udata, xx=xx, yy=yy)
        div_cp = duidxj[..., 0, 0] + duidxj[..., 1, 1]
        cp_end_time = time.time()
        print('Divergence shape', div_cp.shape, f'took {cp_end_time-cp_start_time} seconds')

        # compute shear 1 and 2
        cp_start_time = time.time()
        duidxj = rate_of_strain_cupy.get_duidxj_tensor(udata, xx=xx, yy=yy)
        # Shear 1
        shear1_cp = duidxj[..., 0, 0] - duidxj[..., 1, 1]
        # Shear 2
        shear2_cp = duidxj[..., 1, 0] + duidxj[..., 0, 1]
        cp_end_time = time.time()
        print('Shear 1 and 2 shape', shear1.shape, shear2.shape, f'took {cp_end_time-cp_start_time} seconds\n')


        # confirm both approaches are the same
        if not np.allclose(omega_cupy_1, omega_cupy_2):
            print(omega_cupy_1.shape)
            print(omega_cupy_1)
            print(omega_cupy_2.shape)
            print(omega_cupy_2)
            print(omega_cupy_1-omega_cupy_2)
            raise Exception(f'Results from both approaches on computing cupy vorticity are not the same')

        # confirm if cupy vorticity is the same as numpy's
        if not np.allclose(omega_1, cp.asnumpy(omega_cupy_2)):
            print(omega_1.shape)
            print(omega_1)
            print(cp.asnumpy(omega_cupy_1).shape)
            print(cp.asnumpy(omega_cupy_1))
            print(omega_1-cp.asnumpy(omega_cupy_1))
            raise Exception(f'Result from cupy is not the same as numpy')

        # confirm if cupy divergence is the same as numpy's
        if not np.allclose(div, cp.asnumpy(div_cp)):
            print(div.shape)
            print(div)
            print(cp.asnumpy(div_cp).shape)
            print(cp.asnumpy(div_cp))
            print(div-cp.asnumpy(div_cp))
            raise Exception(f'Divergence result from cupy is not the same as numpy')

        # confirm if cupy shear 1 and 2 are the same as numpy's
        if not np.allclose(shear1, cp.asnumpy(shear1_cp)):
            print(shear1.shape)
            print(shear1)
            print(cp.asnumpy(shear1_cp).shape)
            print(cp.asnumpy(shear1_cp))
            print(shear1-cp.asnumpy(shear1_cp))
            raise Exception(f'Shear 1 result from cupy is not the same as numpy')
        if not np.allclose(shear2, cp.asnumpy(shear2_cp)):
            print(shear2.shape)
            print(shear2)
            print(cp.asnumpy(shear2_cp).shape)
            print(cp.asnumpy(shear2_cp))
            print(shear2-cp.asnumpy(shear2_cp))
            raise Exception(f'Shear 2 result from cupy is not the same as numpy')

        # plot
        if plot:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.pcolormesh(cp.asnumpy(xx), cp.asnumpy(yy), cp.asnumpy(omega_cupy_1[..., 0]), cmap='bwr', vmin=-3e-3, vmax=3e-3)
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
        udata = rate_of_strain_torch.rankine_vortex_2d(xx, yy, x0=32, y0=32, a=10)
        print(f'Compute curl using torch on GPU {udata.device} out of {torch.cuda.device_count()}')
        torch_start_time = time.time()
        print('Sample data (velocity) shape', udata.shape)

        # Compute vorticity using curl
        omega_torch_1 = rate_of_strain_torch.curl(udata, xx=xx, yy=yy)
        torch_end_time = time.time()
        print('Curl vorticity shape', omega_torch_1.shape, f'took {torch_end_time-torch_start_time} seconds')

        # Compute vorticity using rate of strain tensor decomposition
        torch_start_time = time.time()
        duidxj = rate_of_strain_torch.get_duidxj_tensor(udata, xx=xx, yy=yy)
        omega_torch_2 = duidxj[..., 1, 0] - duidxj[..., 0, 1]
        torch_end_time = time.time()
        print('Decomposed rate-of-strain vorticity shape', omega_torch_2.shape, f'took {torch_end_time-torch_start_time} seconds')

        # compute divergence
        torch_start_time = time.time()
        duidxj = rate_of_strain_torch.get_duidxj_tensor(udata, xx=xx, yy=yy)
        div_torch = duidxj[..., 0, 0] + duidxj[..., 1, 1]
        torch_end_time = time.time()
        print('Divergence shape', div_torch.shape, f'took {torch_end_time-torch_start_time} seconds')

        # compute shear 1 and 2
        torch_start_time = time.time()
        duidxj = rate_of_strain_cupy.get_duidxj_tensor(udata, xx=xx, yy=yy)
        # Shear 1
        shear1_torch = duidxj[..., 0, 0] - duidxj[..., 1, 1]
        # Shear 2
        shear2_torch = duidxj[..., 1, 0] + duidxj[..., 0, 1]
        torch_end_time = time.time()
        print('Shear 1 and 2 shape', div_torch.shape, div_torch.shape, f'took {torch_end_time-torch_start_time} seconds\n')

        # confirm both approaches are the same
        if not np.allclose(omega_torch_1, omega_torch_2):
            print(omega_torch_1.shape)
            print(omega_torch_1)
            print(omega_torch_2.shape)
            print(omega_torch_2)
            print(omega_torch_1-omega_torch_2)
            raise Exception(f'Result from both approaches on computing torch vorticity are not the same')

        if not np.allclose(omega_1, omega_torch_2.cpu().numpy()):
            print(omega_1.shape)
            print(omega_1)
            print(omega_torch_2.shape)
            print(omega_torch_2)
            print(omega_1-omega_torch_2)
            raise Exception(f'Result from torch is not the same as numpy')

        # confirm if Torch divergence is the same as numpy's
        if not np.allclose(div, cp.asnumpy(div_torch)):
            print(div.shape)
            print(div)
            print(cp.asnumpy(div_torch).shape)
            print(cp.asnumpy(div_torch))
            print(div-cp.asnumpy(div_torch))
            raise Exception(f'Divergence result from torch is not the same as numpy')

        # confirm if Torch shear 1 and 2 are the same as numpy's
        if not np.allclose(shear1, cp.asnumpy(shear1_torch)):
            print(shear1.shape)
            print(shear1)
            print(cp.asnumpy(shear1_torch).shape)
            print(cp.asnumpy(shear1_torch))
            print(shear1-cp.asnumpy(shear1_torch))
            raise Exception(f'Shear 1 result from torch is not the same as numpy')
        if not np.allclose(shear2, cp.asnumpy(shear2_torch)):
            print(shear2.shape)
            print(shear2)
            print(cp.asnumpy(shear2_torch).shape)
            print(cp.asnumpy(shear2_torch))
            print(shear2-cp.asnumpy(shear2_torch))
            raise Exception(f'Shear 2 result from torch is not the same as numpy')

        # plot
        if plot:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.pcolormesh(xx.cpu().numpy(), yy.cpu().numpy(), omega_torch_2[..., 0].cpu().numpy(), cmap='bwr', vmin=-3e-3, vmax=3e-3)
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