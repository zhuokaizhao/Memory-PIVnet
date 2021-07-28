# the script serves to process and visualize the results from various methods

import os
import glob
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import plot


# helper function that computes energy from the velocity field
def get_energy(velocity):

    dim = velocity.shape[-1]
    # energy = np.zeros(shape[1:])
    energy = np.zeros(velocity.shape[:2])
    for d in range(dim):
        energy += velocity[:, :, d] ** 2
    energy /= 2.
    return energy

# list of methods
methods = ['ground_truth', 'memory-piv-net', 'pyramid', 'cross_correlation']

# corresponding data path or directory
result_paths = ['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/amnesia_memory/50000_seeds/no_pe/time_span_5/true_vel_field/',
                '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/amnesia_memory/50000_seeds/no_pe/time_span_5/blend_vel_field/',
                '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/pyramid/isotropic1024coarse_x_1_1024_y_1_1024_z_662_762_t_1_5024_tstride_20_pyramid.h5',
                '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/cross_correlation/isotropic1024coarse_x_1_1024_y_1_1024_z_662_762_t_1_5024_tstride_20_standard.h5']

# start and end time (both inclusive)
load_time_range = [0, 248]
vis_frame = [41]
img_size = 256

# directory to save the results
figs_dir = '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/figs/Isotropic_1024/velocity/'

# different types of visualizations
plot_image_quiver = False
plot_color_encoded = False
plot_energy = True

# different losses
loss = 'RMSE'

# loaded velocity fields
true_velocity = []
memory_velocity = []
pyramid_velocity = []
cc_velocity = []

# load results from each method
for i, cur_method in enumerate(methods):
    if cur_method == 'ground_truth':
        for t in range(load_time_range[0], load_time_range[1]+1):
            cur_path = os.path.join(result_paths[i], f'true_velocity_{t}.npz')
            true_velocity.append(np.load(cur_path)['velocity'])

        true_velocity = np.array(true_velocity)

    if cur_method == 'memory-piv-net':
        # load the velocity fields of the specified time range
        for t in range(load_time_range[0], load_time_range[1]+1):
            cur_path = os.path.join(result_paths[i], f'test_velocity_blend_{t}.npz')
            memory_velocity.append(np.load(cur_path)['velocity'])

        memory_velocity = np.array(memory_velocity)

    # for pyramid and standar methods
    elif cur_method == 'pyramid' or cur_method == 'cross_correlation':
        cur_path = result_paths[i]
        with h5py.File(cur_path, mode='r') as f:
            # print('The h5 file contains ', list(f.keys()))
            xx, yy = f['x'][...], f['y'][...]
            ux, uy = f['ux'][...]*30, f['uy'][...]*30

        if cur_method == 'pyramid':
            pyramid_velocity = np.stack((ux, uy))
            pyramid_velocity = np.moveaxis(pyramid_velocity, [0, 1, 2, 3], [3, 1, 2, 0])

        else:
            cc_velocity = np.stack((ux, uy))
            cc_velocity = np.moveaxis(cc_velocity, [0, 1, 2, 3], [3, 1, 2, 0])



# print all the shapes
print(f'\nLoaded ground truth velocity has shape {true_velocity.shape}')
print(f'Loaded memory-piv-net velocity has shape {memory_velocity.shape}')
print(f'Loaded pyramid velocity has shape {pyramid_velocity.shape}')
print(f'Loaded cross-correlation velocity has shape {cc_velocity.shape}')

# print(true_velocity[10, :, :, 0].max())
# print(true_velocity[10, :, :, 0].min())
# print(pyramid_velocity[10, :, :, 0].max())
# print(pyramid_velocity[10, :, :, 0].min())

# upsampling pyramid and cc results to full image resolution by duplicating
pyramid_velocity_full_res = []
cc_velocity_full_res = []
for t in range(len(pyramid_velocity)):
    ratio = img_size // pyramid_velocity[t].shape[1]
    pyramid_velocity_full_res.append(pyramid_velocity[t].repeat(ratio, axis=0).repeat(ratio, axis=1))
    cc_velocity_full_res.append(cc_velocity[t].repeat(ratio, axis=0).repeat(ratio, axis=1))

pyramid_velocity_full_res = np.array(pyramid_velocity_full_res)
cc_velocity_full_res = np.array(cc_velocity_full_res)
print(f'\nFull-resolution pyramid velocity has shape {pyramid_velocity_full_res.shape}')
print(f'Full-resolution cross-correlation velocity has shape {cc_velocity_full_res.shape}')


# visualizing the results
for i in vis_frame:
    cur_true_velocity = true_velocity[i]
    cur_memory_velocity = memory_velocity[i]
    cur_pyramid_velocity = pyramid_velocity_full_res[i]
    cur_cc_velocity = cc_velocity_full_res[i]

    # test image superimposed quiver plot
    if plot_image_quiver:
        test_image_path = f'/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/amnesia_memory/50000_seeds/no_pe/time_span_5/test_images/test_velocity_{i}.png'
        test_image = Image.open(test_image_path)

        # plot includes four subplots
        fig, axes = plt.subplots(nrows=1, ncols=len(methods), figsize=(5*len(methods), 5))
        skip = 7

        # superimpose quiver plot on color-coded images
        max_vel = np.max(cur_true_velocity)
        # ground truth
        x = np.linspace(0, img_size-1, img_size)
        y = np.linspace(0, img_size-1, img_size)
        y_pos, x_pos = np.meshgrid(x, y)
        axes[0].imshow(test_image, 'gray')
        Q = axes[0].quiver(y_pos[::skip, ::skip],
                            x_pos[::skip, ::skip],
                            cur_true_velocity[::skip, ::skip, 0]/max_vel,
                            -cur_true_velocity[::skip, ::skip, 1]/max_vel,
                            # scale=4.0,
                            scale_units='inches',
                            color='green')
        Q._init()
        assert isinstance(Q.scale, float)
        axes[0].set_title('Ground truth')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')

        # predictions
        # memory-piv-net
        axes[1].imshow(test_image, 'gray')
        axes[1].quiver(y_pos[::skip, ::skip],
                        x_pos[::skip, ::skip],
                        cur_memory_velocity[::skip, ::skip, 0]/max_vel,
                        -cur_memory_velocity[::skip, ::skip, 1]/max_vel,
                        scale=Q.scale,
                        scale_units='inches',
                        color='green')
        axes[1].set_title('Memory-PIVnet')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        # compute and annotate loss
        if loss == 'MSE':
            cur_loss = np.square(cur_true_velocity - cur_memory_velocity).mean(axis=None)
        elif loss == 'RMSE':
            cur_loss = np.sqrt(np.square(cur_true_velocity - cur_memory_velocity)).mean(axis=None)
        axes[1].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')

        # pyramid
        axes[2].imshow(test_image, 'gray')
        axes[2].quiver(y_pos[::skip, ::skip],
                        x_pos[::skip, ::skip],
                        cur_pyramid_velocity[::skip, ::skip, 0]/max_vel,
                        -cur_pyramid_velocity[::skip, ::skip, 1]/max_vel,
                        scale=Q.scale,
                        scale_units='inches',
                        color='green')
        axes[2].set_title('Pyramid (original results scaled 30x)')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        # compute and annotate loss
        if loss == 'MSE':
            cur_loss = np.square(cur_true_velocity - cur_pyramid_velocity).mean(axis=None)
        elif loss == 'RMSE':
            cur_loss = np.sqrt(np.square(cur_true_velocity - cur_pyramid_velocity)).mean(axis=None)
        axes[2].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')


        # cross-correlation
        axes[3].imshow(test_image, 'gray')
        axes[3].quiver(y_pos[::skip, ::skip],
                        x_pos[::skip, ::skip],
                        cur_cc_velocity[::skip, ::skip, 0]/max_vel,
                        -cur_cc_velocity[::skip, ::skip, 1]/max_vel,
                        scale=Q.scale,
                        scale_units='inches',
                        color='green')
        axes[3].set_title('Cross-correlation (original results scaled 30x)')
        axes[3].set_xlabel('x')
        axes[3].set_ylabel('y')
        # compute and annotate loss
        if loss == 'MSE':
            cur_loss = np.square(cur_true_velocity - cur_cc_velocity).mean(axis=None)
        elif loss == 'RMSE':
            cur_loss = np.sqrt(np.square(cur_true_velocity - cur_cc_velocity)).mean(axis=None)
        axes[3].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')


        # save the image
        test_quiver_path = os.path.join(figs_dir, f'test_quiver_{i}.png')
        plt.savefig(test_quiver_path, bbox_inches='tight', dpi=500)
        print(f'\nSuperimposed test quiver plot has been saved to {test_quiver_path}')


    # color encoding plots
    if plot_color_encoded:
        # ground truth
        cur_flow_true, max_vel = plot.visualize_flow(cur_true_velocity)
        # predictions
        # memory-piv-net
        memory_flow_pred, _ = plot.visualize_flow(cur_memory_velocity, max_vel=max_vel)
        # pyramid
        pyramid_flow_pred, _ = plot.visualize_flow(cur_pyramid_velocity, max_vel=max_vel)
        # cross-correlation
        cc_flow_pred, _ = plot.visualize_flow(cur_cc_velocity, max_vel=max_vel)

        # convert to Image
        cur_flow_true = Image.fromarray(cur_flow_true)
        memory_flow_pred = Image.fromarray(memory_flow_pred)
        pyramid_flow_pred = Image.fromarray(pyramid_flow_pred)
        cc_flow_pred = Image.fromarray(cc_flow_pred)

        # plot includes four subplots
        fig, axes = plt.subplots(nrows=1, ncols=len(methods), figsize=(5*len(methods), 5))
        skip = 7

        # superimpose quiver plot on color-coded images
        # ground truth
        x = np.linspace(0, img_size-1, img_size)
        y = np.linspace(0, img_size-1, img_size)
        y_pos, x_pos = np.meshgrid(x, y)
        axes[0].imshow(cur_flow_true)
        Q = axes[0].quiver(y_pos[::skip, ::skip],
                            x_pos[::skip, ::skip],
                            cur_true_velocity[::skip, ::skip, 0]/max_vel,
                            -cur_true_velocity[::skip, ::skip, 1]/max_vel,
                            # scale=4.0,
                            scale_units='inches')
        Q._init()
        assert isinstance(Q.scale, float)
        axes[0].set_title('Ground truth')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')

        # predictions
        # memory-piv-net
        axes[1].imshow(memory_flow_pred)
        axes[1].quiver(y_pos[::skip, ::skip],
                        x_pos[::skip, ::skip],
                        cur_memory_velocity[::skip, ::skip, 0]/max_vel,
                        -cur_memory_velocity[::skip, ::skip, 1]/max_vel,
                        scale=Q.scale,
                        scale_units='inches')
        axes[1].set_title('Memory-PIVnet')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        # compute and annotate loss
        if loss == 'MSE':
            cur_loss = np.square(cur_true_velocity - cur_memory_velocity).mean(axis=None)
        elif loss == 'RMSE':
            cur_loss = np.sqrt(np.square(cur_true_velocity - cur_memory_velocity)).mean(axis=None)
        axes[1].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')

        # pyramid
        axes[2].imshow(pyramid_flow_pred)
        axes[2].quiver(y_pos[::skip, ::skip],
                        x_pos[::skip, ::skip],
                        cur_pyramid_velocity[::skip, ::skip, 0]/max_vel,
                        -cur_pyramid_velocity[::skip, ::skip, 1]/max_vel,
                        scale=Q.scale,
                        scale_units='inches')
        axes[2].set_title('Pyramid (original results scaled 30x)')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        # compute and annotate loss
        if loss == 'MSE':
            cur_loss = np.square(cur_true_velocity - cur_pyramid_velocity).mean(axis=None)
        elif loss == 'RMSE':
            cur_loss = np.sqrt(np.square(cur_true_velocity - cur_pyramid_velocity)).mean(axis=None)
        axes[2].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')

        # cross-correlation
        axes[3].imshow(cc_flow_pred)
        axes[3].quiver(y_pos[::skip, ::skip],
                        x_pos[::skip, ::skip],
                        cur_cc_velocity[::skip, ::skip, 0]/max_vel,
                        -cur_cc_velocity[::skip, ::skip, 1]/max_vel,
                        scale=Q.scale,
                        scale_units='inches')
        axes[3].set_title('Cross-correlation (original results scaled 30x)')
        axes[3].set_xlabel('x')
        axes[3].set_ylabel('y')
        # compute and annotate loss
        if loss == 'MSE':
            cur_loss = np.square(cur_true_velocity - cur_cc_velocity).mean(axis=None)
        elif loss == 'RMSE':
            cur_loss = np.sqrt(np.square(cur_true_velocity - cur_cc_velocity)).mean(axis=None)
        axes[3].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')

        # save the image
        color_encoded_path = os.path.join(figs_dir, f'color_encoded_{i}.png')
        plt.savefig(color_encoded_path, bbox_inches='tight', dpi=500)
        print(f'\nColor-encoded plot has been saved to {color_encoded_path}')


    # energy plot
    if plot_energy:
        # compute energy
        cur_true_energy = get_energy(cur_true_velocity)
        cur_memory_energy = get_energy(cur_memory_velocity)
        cur_pyramid_energy = get_energy(cur_pyramid_velocity)
        cur_cc_energy = get_energy(cur_cc_velocity)

        # plot includes four subplots
        fig, axes = plt.subplots(nrows=1, ncols=len(methods), figsize=(5*len(methods), 5))
        skip = 7

        # superimpose quiver plot on color-coded images
        max_vel = np.max(cur_true_velocity)
        # ground truth
        x = np.linspace(0, img_size-1, img_size)
        y = np.linspace(0, img_size-1, img_size)
        y_pos, x_pos = np.meshgrid(x, y)
        axes[0].pcolor(cur_true_energy)
        Q = axes[0].quiver(y_pos[::skip, ::skip],
                            x_pos[::skip, ::skip],
                            cur_true_velocity[::skip, ::skip, 0]/max_vel,
                            -cur_true_velocity[::skip, ::skip, 1]/max_vel,
                            # scale=4.0,
                            scale_units='inches',
                            color='green')
        Q._init()
        assert isinstance(Q.scale, float)
        axes[0].set_title('Ground truth')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].invert_yaxis()

        # predictions
        # memory-piv-net
        axes[1].pcolor(cur_memory_energy)
        axes[1].quiver(y_pos[::skip, ::skip],
                        x_pos[::skip, ::skip],
                        cur_memory_velocity[::skip, ::skip, 0]/max_vel,
                        -cur_memory_velocity[::skip, ::skip, 1]/max_vel,
                        scale=Q.scale,
                        scale_units='inches',
                        color='green')
        axes[1].set_title('Memory-PIVnet')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].invert_yaxis()
        # compute and annotate loss
        if loss == 'MSE':
            cur_loss = np.square(cur_true_energy - cur_memory_energy).mean(axis=None)
        elif loss == 'RMSE':
            cur_loss = np.sqrt(np.square(cur_true_energy - cur_memory_energy)).mean(axis=None)
        axes[1].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')

        # pyramid
        axes[2].pcolor(cur_pyramid_energy)
        axes[2].quiver(y_pos[::skip, ::skip],
                        x_pos[::skip, ::skip],
                        cur_pyramid_velocity[::skip, ::skip, 0]/max_vel,
                        -cur_pyramid_velocity[::skip, ::skip, 1]/max_vel,
                        scale=Q.scale,
                        scale_units='inches',
                        color='green')
        axes[2].set_title('Pyramid (original results scaled 30x)')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        axes[2].invert_yaxis()
        # compute and annotate loss
        if loss == 'MSE':
            cur_loss = np.square(cur_true_energy - cur_pyramid_energy).mean(axis=None)
        elif loss == 'RMSE':
            cur_loss = np.sqrt(np.square(cur_true_energy - cur_pyramid_energy)).mean(axis=None)
        axes[2].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')


        # cross-correlation
        axes[3].pcolor(cur_cc_energy)
        axes[3].quiver(y_pos[::skip, ::skip],
                        x_pos[::skip, ::skip],
                        cur_cc_velocity[::skip, ::skip, 0]/max_vel,
                        -cur_cc_velocity[::skip, ::skip, 1]/max_vel,
                        scale=Q.scale,
                        scale_units='inches',
                        color='green')
        axes[3].set_title('Cross-correlation (original results scaled 30x)')
        axes[3].set_xlabel('x')
        axes[3].set_ylabel('y')
        axes[3].invert_yaxis()
        # compute and annotate loss
        if loss == 'MSE':
            cur_loss = np.square(cur_true_energy - cur_cc_energy).mean(axis=None)
        elif loss == 'RMSE':
            cur_loss = np.sqrt(np.square(cur_true_energy - cur_cc_energy)).mean(axis=None)
        axes[3].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')


        # save the image
        energy_path = os.path.join(figs_dir, f'energy_{i}.png')
        plt.savefig(energy_path, bbox_inches='tight', dpi=500)
        print(f'\nEnergy plot has been saved to {energy_path}')


