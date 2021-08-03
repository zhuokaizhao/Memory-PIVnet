# the script serves to process and visualize the results from various methods

import os
import glob
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
methods = ['ground_truth', 'memory-piv-net', 'LiteFlowNet-en', 'pyramid', 'cross_correlation']

# corresponding data path or directory
result_paths = ['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/amnesia_memory/50000_seeds/no_pe/time_span_5/true_vel_field/',
                '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/amnesia_memory/50000_seeds/no_pe/time_span_5/blend_vel_field/',
                '/home/zhuokai/Desktop/UChicago/Research/PIV-LiteFlowNet-en-Pytorch/output/Isotropic_1024/50000_seeds/lfn_vel_field/',
                '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/pyramid/isotropic1024coarse_x_1_1024_y_1_1024_z_662_762_t_1_5024_tstride_20_pyramid.h5',
                '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/cross_correlation/isotropic1024coarse_x_1_1024_y_1_1024_z_662_762_t_1_5024_tstride_20_standard.h5']

# start and end time (both inclusive)
load_time_range = [0, 248]
# frame 81, 153, 154 have broken ground truth
vis_frame = [41, 141, 241]
vis_frame = list(range(0, 249, 20))
# vis_frame = list(range(12, 81)) + list(range(82, 153)) + list(range(155, 249))
img_size = 256
my_dpi = 100

# directory to save the results
figs_dir = '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/new_figs/Isotropic_1024/velocity/50000_seeds/'

# different types of visualizations
plot_image_quiver = True
plot_color_encoded = True
plot_aee_heatmap = True
plot_energy = True
plot_error_line_plot = False

# different losses
loss = 'RMSE'

# loaded velocity fields
true_velocity = []
memory_velocity = []
lfn_velocity = []
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

    if cur_method == 'LiteFlowNet-en':
        # load the velocity fields of the specified time range
        for t in range(load_time_range[0], load_time_range[1]+1):
            cur_path = os.path.join(result_paths[i], f'lfn_velocity_{t}.npz')
            lfn_velocity.append(np.load(cur_path)['velocity'])

        lfn_velocity = np.array(lfn_velocity)

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
print(f'Loaded LiteFlowNet-en velocity has shape {lfn_velocity.shape}')
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

if plot_error_line_plot:
    memory_errors = []
    lfn_errors = []
    pyramid_errors = []
    cc_errors = []

# visualizing the results
for i in tqdm(vis_frame):
    cur_true_velocity = true_velocity[i]
    cur_memory_velocity = memory_velocity[i]
    cur_lfn_velocity = lfn_velocity[i]
    cur_pyramid_velocity = pyramid_velocity_full_res[i]
    cur_cc_velocity = cc_velocity_full_res[i]

    # test image superimposed quiver plot
    if plot_image_quiver:
        test_image_path = f'/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/amnesia_memory/50000_seeds/no_pe/time_span_5/test_images/test_velocity_{i}.png'
        test_image = Image.open(test_image_path)

        # plot includes four subplots
        fig, axes = plt.subplots(nrows=1, ncols=len(methods), figsize=(5*len(methods), 5))
        plt.suptitle(f't = {i}')
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

        # predictions
        # lite-flow-net-en
        axes[2].imshow(test_image, 'gray')
        axes[2].quiver(y_pos[::skip, ::skip],
                        x_pos[::skip, ::skip],
                        cur_lfn_velocity[::skip, ::skip, 0]/max_vel,
                        -cur_lfn_velocity[::skip, ::skip, 1]/max_vel,
                        scale=Q.scale,
                        scale_units='inches',
                        color='green')
        axes[2].set_title('LiteFlowNet-en')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        # compute and annotate loss
        if loss == 'MSE':
            cur_loss = np.square(cur_true_velocity - cur_lfn_velocity).mean(axis=None)
        elif loss == 'RMSE':
            cur_loss = np.sqrt(np.square(cur_true_velocity - cur_lfn_velocity)).mean(axis=None)
        axes[2].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')

        # pyramid
        axes[3].imshow(test_image, 'gray')
        axes[3].quiver(y_pos[::skip, ::skip],
                        x_pos[::skip, ::skip],
                        cur_pyramid_velocity[::skip, ::skip, 0]/max_vel,
                        -cur_pyramid_velocity[::skip, ::skip, 1]/max_vel,
                        scale=Q.scale,
                        scale_units='inches',
                        color='green')
        axes[3].set_title('Pyramid (output scaled 30x)')
        axes[3].set_xlabel('x')
        axes[3].set_ylabel('y')
        # compute and annotate loss
        if loss == 'MSE':
            cur_loss = np.square(cur_true_velocity - cur_pyramid_velocity).mean(axis=None)
        elif loss == 'RMSE':
            cur_loss = np.sqrt(np.square(cur_true_velocity - cur_pyramid_velocity)).mean(axis=None)
        axes[3].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')


        # cross-correlation
        axes[4].imshow(test_image, 'gray')
        axes[4].quiver(y_pos[::skip, ::skip],
                        x_pos[::skip, ::skip],
                        cur_cc_velocity[::skip, ::skip, 0]/max_vel,
                        -cur_cc_velocity[::skip, ::skip, 1]/max_vel,
                        scale=Q.scale,
                        scale_units='inches',
                        color='green')
        axes[4].set_title('Cross-correlation (output scaled 30x)')
        axes[4].set_xlabel('x')
        axes[4].set_ylabel('y')
        # compute and annotate loss
        if loss == 'MSE':
            cur_loss = np.square(cur_true_velocity - cur_cc_velocity).mean(axis=None)
        elif loss == 'RMSE':
            cur_loss = np.sqrt(np.square(cur_true_velocity - cur_cc_velocity)).mean(axis=None)
        axes[4].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')


        # save the image
        test_quiver_dir = os.path.join(figs_dir, 'test_quiver')
        os.makedirs(test_quiver_dir, exist_ok=True)
        test_quiver_path = os.path.join(test_quiver_dir, f'test_quiver_{str(i).zfill(4)}.png')
        plt.savefig(test_quiver_path, bbox_inches='tight', dpi=my_dpi)
        fig.clf()
        plt.close(fig)
        # print(f'\nSuperimposed test quiver plot has been saved to {test_quiver_path}')

    # color encoding plots
    if plot_color_encoded:
        # ground truth
        cur_flow_true, max_vel = plot.visualize_flow(cur_true_velocity)
        # predictions
        # memory-piv-net
        memory_flow_pred, _ = plot.visualize_flow(cur_memory_velocity, max_vel=max_vel)
        # lfn
        lfn_flow_pred, _ = plot.visualize_flow(cur_lfn_velocity, max_vel=max_vel)
        # pyramid
        pyramid_flow_pred, _ = plot.visualize_flow(cur_pyramid_velocity, max_vel=max_vel)
        # cross-correlation
        cc_flow_pred, _ = plot.visualize_flow(cur_cc_velocity, max_vel=max_vel)

        # convert to Image
        cur_flow_true = Image.fromarray(cur_flow_true)
        memory_flow_pred = Image.fromarray(memory_flow_pred)
        lfn_flow_pred = Image.fromarray(lfn_flow_pred)
        pyramid_flow_pred = Image.fromarray(pyramid_flow_pred)
        cc_flow_pred = Image.fromarray(cc_flow_pred)

        # plot includes four subplots
        fig, axes = plt.subplots(nrows=1, ncols=len(methods), figsize=(5*len(methods), 5))
        plt.suptitle(f't = {i}')
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

        # lite-flow-net-en
        axes[2].imshow(lfn_flow_pred)
        axes[2].quiver(y_pos[::skip, ::skip],
                        x_pos[::skip, ::skip],
                        cur_lfn_velocity[::skip, ::skip, 0]/max_vel,
                        -cur_lfn_velocity[::skip, ::skip, 1]/max_vel,
                        scale=Q.scale,
                        scale_units='inches',
                        color='green')
        axes[2].set_title('LiteFlowNet-en')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        # compute and annotate loss
        if loss == 'MSE':
            cur_loss = np.square(cur_true_velocity - cur_lfn_velocity).mean(axis=None)
        elif loss == 'RMSE':
            cur_loss = np.sqrt(np.square(cur_true_velocity - cur_lfn_velocity)).mean(axis=None)
        axes[2].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')

        # pyramid
        axes[3].imshow(pyramid_flow_pred)
        axes[3].quiver(y_pos[::skip, ::skip],
                        x_pos[::skip, ::skip],
                        cur_pyramid_velocity[::skip, ::skip, 0]/max_vel,
                        -cur_pyramid_velocity[::skip, ::skip, 1]/max_vel,
                        scale=Q.scale,
                        scale_units='inches')
        axes[3].set_title('Pyramid (output scaled 30x)')
        axes[3].set_xlabel('x')
        axes[3].set_ylabel('y')
        # compute and annotate loss
        if loss == 'MSE':
            cur_loss = np.square(cur_true_velocity - cur_pyramid_velocity).mean(axis=None)
        elif loss == 'RMSE':
            cur_loss = np.sqrt(np.square(cur_true_velocity - cur_pyramid_velocity)).mean(axis=None)
        axes[3].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')

        # cross-correlation
        axes[4].imshow(cc_flow_pred)
        axes[4].quiver(y_pos[::skip, ::skip],
                        x_pos[::skip, ::skip],
                        cur_cc_velocity[::skip, ::skip, 0]/max_vel,
                        -cur_cc_velocity[::skip, ::skip, 1]/max_vel,
                        scale=Q.scale,
                        scale_units='inches')
        axes[4].set_title('Cross-correlation (output scaled 30x)')
        axes[4].set_xlabel('x')
        axes[4].set_ylabel('y')
        # compute and annotate loss
        if loss == 'MSE':
            cur_loss = np.square(cur_true_velocity - cur_cc_velocity).mean(axis=None)
        elif loss == 'RMSE':
            cur_loss = np.sqrt(np.square(cur_true_velocity - cur_cc_velocity)).mean(axis=None)
        axes[4].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')

        # save the image
        color_encoded_dir = os.path.join(figs_dir, 'color_encoded')
        os.makedirs(color_encoded_dir, exist_ok=True)
        color_encoded_path = os.path.join(color_encoded_dir, f'color_encoded_{str(i).zfill(4)}.png')
        plt.savefig(color_encoded_path, bbox_inches='tight', dpi=my_dpi)
        fig.clf()
        plt.close(fig)
        # print(f'\nColor-encoded plot has been saved to {color_encoded_path}')

    # aee heatmap plot
    if plot_aee_heatmap:

        # average end point error for all the outputs
        cur_memory_aee = np.sqrt((cur_memory_velocity[:,:,0]-cur_true_velocity[:,:,0])**2 + (cur_memory_velocity[:,:,1]-cur_true_velocity[:,:,1])**2)
        cur_lfn_aee = np.sqrt((cur_lfn_velocity[:,:,0]-cur_true_velocity[:,:,0])**2 + (cur_lfn_velocity[:,:,1]-cur_true_velocity[:,:,1])**2)
        cur_pyramid_aee = np.sqrt((cur_pyramid_velocity[:,:,0]-cur_true_velocity[:,:,0])**2 + (cur_pyramid_velocity[:,:,1]-cur_true_velocity[:,:,1])**2)
        cur_cc_aee = np.sqrt((cur_cc_velocity[:,:,0]-cur_true_velocity[:,:,0])**2 + (cur_cc_velocity[:,:,1]-cur_true_velocity[:,:,1])**2)

        # plot includes four subplots
        fig, axes = plt.subplots(nrows=1, ncols=len(methods)-1, figsize=(5*(len(methods)-1), 5))
        plt.suptitle(f't = {i}')
        cmap_range = [0, 40]

        axes[0].imshow(cur_memory_aee, vmin=cmap_range[0], vmax=cmap_range[1], cmap=plt.get_cmap('viridis'))
        axes[0].set_title('Memory-PIVnet')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].annotate(f'AEE: ' + '{:.3f}'.format(cur_memory_aee.mean()), (5, 10), color='white', fontsize='medium')

        axes[1].imshow(cur_lfn_aee, vmin=cmap_range[0], vmax=cmap_range[1], cmap=plt.get_cmap('viridis'))
        axes[1].set_title('LiteFlowNet-en')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].annotate(f'AEE: ' + '{:.3f}'.format(cur_lfn_aee.mean()), (5, 10), color='white', fontsize='medium')

        axes[2].imshow(cur_pyramid_aee, vmin=cmap_range[0], vmax=cmap_range[1], cmap=plt.get_cmap('viridis'))
        axes[2].set_title('Pyramid (output scaled 30x)')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        axes[2].annotate(f'AEE: ' + '{:.3f}'.format(cur_pyramid_aee.mean()), (5, 10), color='white', fontsize='medium')

        im=axes[3].imshow(cur_cc_aee, vmin=cmap_range[0], vmax=cmap_range[1], cmap=plt.get_cmap('viridis'))
        axes[3].set_title('Cross-correlation (output scaled 30x)')
        axes[3].set_xlabel('x')
        axes[3].set_ylabel('y')
        axes[3].annotate(f'AEE: ' + '{:.3f}'.format(cur_cc_aee.mean()), (5, 10), color='white', fontsize='medium')

        cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
        plt.colorbar(im, cax=cax, **kw)

        # save the image
        aee_dir = os.path.join(figs_dir, 'aee_plot')
        os.makedirs(aee_dir, exist_ok=True)
        aee_path = os.path.join(aee_dir, f'aee_error_{str(i).zfill(4)}.png')
        plt.savefig(aee_path, bbox_inches='tight', dpi=my_dpi)
        fig.clf()
        plt.close(fig)
        # print(f'\nAEE plot has been saved to {aee_path}')

    # energy plot
    if plot_energy:
        # compute energy
        cur_true_energy = get_energy(cur_true_velocity)
        cur_memory_energy = get_energy(cur_memory_velocity)
        cur_lfn_energy = get_energy(cur_lfn_velocity)
        cur_pyramid_energy = get_energy(cur_pyramid_velocity)
        cur_cc_energy = get_energy(cur_cc_velocity)

        # plot includes four subplots
        fig, axes = plt.subplots(nrows=1, ncols=len(methods), figsize=(5*len(methods), 5))
        plt.suptitle(f't = {i}')
        skip = 7
        cmap_range = [np.min(cur_true_energy), np.max(cur_true_energy)]

        # superimpose quiver plot on color-coded images
        max_vel = np.max(cur_true_velocity)
        # ground truth
        x = np.linspace(0, img_size-1, img_size)
        y = np.linspace(0, img_size-1, img_size)
        y_pos, x_pos = np.meshgrid(x, y)
        axes[0].imshow(cur_true_energy, vmin=cmap_range[0], vmax=cmap_range[1], cmap=plt.get_cmap('viridis'))
        Q = axes[0].quiver(y_pos[::skip, ::skip],
                            x_pos[::skip, ::skip],
                            cur_true_velocity[::skip, ::skip, 0]/max_vel,
                            -cur_true_velocity[::skip, ::skip, 1]/max_vel,
                            # scale=4.0,
                            scale_units='inches',
                            color='black')
        Q._init()
        assert isinstance(Q.scale, float)
        axes[0].set_title('Ground truth')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')

        # predictions
        # memory-piv-net
        axes[1].imshow(cur_memory_energy, vmin=cmap_range[0], vmax=cmap_range[1], cmap=plt.get_cmap('viridis'))
        axes[1].quiver(y_pos[::skip, ::skip],
                        x_pos[::skip, ::skip],
                        cur_memory_velocity[::skip, ::skip, 0]/max_vel,
                        -cur_memory_velocity[::skip, ::skip, 1]/max_vel,
                        scale=Q.scale,
                        scale_units='inches',
                        color='black')
        axes[1].set_title('Memory-PIVnet')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        # compute and annotate loss
        if loss == 'MSE':
            cur_loss = np.square(cur_true_energy - cur_memory_energy).mean(axis=None)
        elif loss == 'RMSE':
            cur_loss = np.sqrt(np.square(cur_true_energy - cur_memory_energy)).mean(axis=None)
        axes[1].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')

        # lite-flow-net-en
        axes[2].imshow(cur_lfn_energy, vmin=cmap_range[0], vmax=cmap_range[1], cmap=plt.get_cmap('viridis'))
        axes[2].quiver(y_pos[::skip, ::skip],
                        x_pos[::skip, ::skip],
                        cur_lfn_velocity[::skip, ::skip, 0]/max_vel,
                        -cur_lfn_velocity[::skip, ::skip, 1]/max_vel,
                        scale=Q.scale,
                        scale_units='inches',
                        color='black')
        axes[2].set_title('LiteFlowNet-en')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        # compute and annotate loss
        if loss == 'MSE':
            cur_loss = np.square(cur_true_energy - cur_lfn_energy).mean(axis=None)
        elif loss == 'RMSE':
            cur_loss = np.sqrt(np.square(cur_true_energy - cur_lfn_energy)).mean(axis=None)
        axes[2].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')

        # pyramid
        axes[3].imshow(cur_pyramid_energy, vmin=cmap_range[0], vmax=cmap_range[1], cmap=plt.get_cmap('viridis'))
        axes[3].quiver(y_pos[::skip, ::skip],
                        x_pos[::skip, ::skip],
                        cur_pyramid_velocity[::skip, ::skip, 0]/max_vel,
                        -cur_pyramid_velocity[::skip, ::skip, 1]/max_vel,
                        scale=Q.scale,
                        scale_units='inches',
                        color='black')
        axes[3].set_title('Pyramid (output scaled 30x)')
        axes[3].set_xlabel('x')
        axes[3].set_ylabel('y')
        # compute and annotate loss
        if loss == 'MSE':
            cur_loss = np.square(cur_true_energy - cur_pyramid_energy).mean(axis=None)
        elif loss == 'RMSE':
            cur_loss = np.sqrt(np.square(cur_true_energy - cur_pyramid_energy)).mean(axis=None)
        axes[3].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')


        # cross-correlation
        axes[4].imshow(cur_cc_energy, vmin=cmap_range[0], vmax=cmap_range[1], cmap=plt.get_cmap('viridis'))
        axes[4].quiver(y_pos[::skip, ::skip],
                        x_pos[::skip, ::skip],
                        cur_cc_velocity[::skip, ::skip, 0]/max_vel,
                        -cur_cc_velocity[::skip, ::skip, 1]/max_vel,
                        scale=Q.scale,
                        scale_units='inches',
                        color='green')
        axes[4].set_title('Cross-correlation (output scaled 30x)')
        axes[4].set_xlabel('x')
        axes[4].set_ylabel('y')
        # compute and annotate loss
        if loss == 'MSE(Energy)':
            cur_loss = np.square(cur_true_energy - cur_cc_energy).mean(axis=None)
        elif loss == 'RMSE':
            cur_loss = np.sqrt(np.square(cur_true_energy - cur_cc_energy)).mean(axis=None)
        axes[4].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')


        # save the image
        energy_dir = os.path.join(figs_dir, 'energy_plot')
        os.makedirs(energy_dir, exist_ok=True)
        energy_path = os.path.join(energy_dir, f'energy_{str(i).zfill(4)}.png')
        plt.savefig(energy_path, bbox_inches='tight', dpi=my_dpi)
        fig.clf()
        plt.close(fig)
        # print(f'\nEnergy plot has been saved to {energy_path}')

    # error line plot
    if plot_error_line_plot:
        if loss == 'MSE':
            memory_errors.append(np.square(cur_true_velocity - cur_memory_velocity).mean(axis=None))
            lfn_errors.append(np.square(cur_true_velocity - cur_lfn_velocity).mean(axis=None))
            pyramid_errors.append(np.square(cur_true_velocity - cur_pyramid_velocity).mean(axis=None))
            cc_errors.append(np.square(cur_true_velocity - cur_cc_velocity).mean(axis=None))
        elif loss == 'RMSE':
            memory_errors.append(np.sqrt(np.square(cur_true_velocity - cur_memory_velocity)).mean(axis=None))
            lfn_errors.append(np.sqrt(np.square(cur_true_velocity - cur_lfn_velocity)).mean(axis=None))
            pyramid_errors.append(np.sqrt(np.square(cur_true_velocity - cur_pyramid_velocity)).mean(axis=None))
            cc_errors.append(np.sqrt(np.square(cur_true_velocity - cur_cc_velocity)).mean(axis=None))


if plot_error_line_plot:
    fig, ax = plt.subplots()
    ax.plot(vis_frame, memory_errors, label='Memory-PIVnet')
    ax.plot(vis_frame, lfn_errors, label='LiteFlowNet-en')
    ax.plot(vis_frame, pyramid_errors, label='Pyramid')
    ax.plot(vis_frame, cc_errors, label='CC')
    ax.set(xlabel='timestamp', ylabel=f'{loss}')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    loss_curve_path = os.path.join(figs_dir, f'all_frame_losses.png')
    fig.savefig(loss_curve_path, bbox_inches='tight', dpi=my_dpi)
    print(f'\nLosses of all frames plot has been saved to {loss_curve_path}')
