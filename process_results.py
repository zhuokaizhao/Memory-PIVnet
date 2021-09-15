# the script serves to process and visualize the results from various methods

import os
import glob
import h5py
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import plot
import vorticity


# helper function that computes energy from the velocity field
def get_energy(velocity):

    dim = velocity.shape[-1]
    # energy = np.zeros(shape[1:])
    energy = np.zeros(velocity.shape[:2])
    for d in range(dim):
        energy += velocity[:, :, d] ** 2

    energy /= 2.

    return energy


# return all_vorticity_fields
def compute_vorticity(cur_velocity_field):
    # all_velocity_fields has shape (height, width, 2)
    num_rows = cur_velocity_field.shape[0]
    num_cols = cur_velocity_field.shape[1]
    x, y = list(range(num_cols)), list(range(num_rows))
    xx, yy = np.meshgrid(x, y)

    # curl function takes (dim, num_rows, num_cols)
    udata = np.moveaxis(cur_velocity_field, [0, 1, 2], [1, 2, 0])
    cur_vorticity = vorticity.curl(udata, xx=xx, yy=yy)

    return cur_vorticity


def main():

    parser = argparse.ArgumentParser()
    # mode (velocity or vorticity)
    parser.add_argument('--mode', required=True, action='store', nargs=1, dest='mode')
    # start and end t used when testing (both inclusive)
    parser.add_argument('--start-t', action='store', nargs=1, dest='start_t')
    parser.add_argument('--end-t', action='store', nargs=1, dest='end_t')
    # loss function
    parser.add_argument('-l', '--loss', action='store', nargs=1, dest='loss')

    args = parser.parse_args()
    mode = args.mode[0]
    start_t = int(args.start_t[0])
    end_t = int(args.end_t[0])
    loss = args.loss[0]

    # corresponding data path or directory
    if mode == 'velocity':
        # list of methods
        methods = ['ground_truth', 'memory-piv-net', 'LiteFlowNet-en', 'pyramid', 'cross_correlation']
        result_paths = ['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/amnesia_memory/50000_seeds/no_pe/time_span_5/true_vel_field/',
                        '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/amnesia_memory/50000_seeds/no_pe/time_span_5/blend_vel_field/',
                        '/home/zhuokai/Desktop/UChicago/Research/PIV-LiteFlowNet-en-Pytorch/output/Isotropic_1024/50000_seeds/lfn_vel_field/',
                        '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/pyramid/TR_Pyramid(2,5)_MPd(1x8x8_50ov)_2x32x32.h5',
                        '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/cross_correlation/TR_PIV_MPd(1x8x8_50ov)_2x32x32.h5']
    # when vorticity, still velocity results is loaded except ground truth and memory-piv-net
    elif mode == 'vorticity':
        # list of methods
        methods = ['ground_truth', 'memory-piv-net', 'pyramid', 'cross_correlation']
        result_paths = ['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/vorticity/amnesia_memory/50000_seeds/no_pe/time_span_5/true_vor_field/',
                        '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/vorticity/amnesia_memory/50000_seeds/no_pe/time_span_5/blend_vor_field/',
                        '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/pyramid/TR_Pyramid(2,5)_MPd(1x8x8_50ov)_2x32x32.h5',
                        '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/cross_correlation/TR_PIV_MPd(1x8x8_50ov)_2x32x32.h5']
    else:
        raise Exception(f'Unknown mode {mode}')


    # sanity check
    if len(methods) != len(result_paths):
        raise Exception(f'Number of methods should equal to number of result paths')

    # start and end time (both inclusive)
    time_range = [start_t, end_t]
    # frame 81, 153, 154 have broken ground truth
    non_vis_frame = [81, 153, 154]
    img_size = 256
    my_dpi = 100

    # directory to save the results
    figs_dir = '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/new_figs/Isotropic_1024/velocity/50000_seeds/'

    # load model testing output
    # different types of visualizations
    plot_image_quiver = True
    plot_color_encoded = True
    plot_aee_heatmap = True
    plot_energy = True
    plot_velocity_error_line_plot = True
    plot_vorticity_error_line_plot = True

    # loaded velocity fields
    ground_truths = []
    memory_results = []
    lfn_results = []
    pyramid_results = []
    cc_results = []

    # load results from each method
    for i, cur_method in enumerate(methods):
        if cur_method == 'ground_truth':
            for t in range(time_range[0], time_range[1]+1):
                cur_path = os.path.join(result_paths[i], f'true_{mode}_{t}.npz')
                ground_truths.append(np.load(cur_path)[f'{mode}'])

            ground_truths = np.array(ground_truths)

        if cur_method == 'memory-piv-net':
            # load the velocity fields of the specified time range
            for t in range(time_range[0], time_range[1]+1):
                cur_path = os.path.join(result_paths[i], f'test_{mode}_blend_{t}.npz')
                memory_results.append(np.load(cur_path)[f'{mode}'])

            memory_results = np.array(memory_results)

        if cur_method == 'LiteFlowNet-en':
            # load the velocity fields of the specified time range
            for t in range(time_range[0], time_range[1]+1):
                cur_path = os.path.join(result_paths[i], f'lfn_{mode}_{t}.npz')
                lfn_results.append(np.load(cur_path)[f'{mode}'])

            lfn_results = np.array(lfn_results)

        # for pyramid and standar methods
        elif cur_method == 'pyramid' or cur_method == 'cross_correlation':
            cur_path = result_paths[i]
            with h5py.File(cur_path, mode='r') as f:
                # print('The h5 file contains ', list(f.keys()))
                xx, yy = f['x'][...], f['y'][...]
                ux, uy = f['ux'][...], f['uy'][...]

            if cur_method == 'pyramid':
                pyramid_results = np.stack((ux, uy))
                pyramid_results = np.moveaxis(pyramid_results, [0, 1, 2, 3], [3, 1, 2, 0])
            else:
                cc_results = np.stack((ux, uy))
                cc_results = np.moveaxis(cc_results, [0, 1, 2, 3], [3, 1, 2, 0])

    # print all the shapes
    print(f'\nLoaded ground truth velocity has shape {ground_truths.shape}')
    print(f'Loaded memory-piv-net velocity has shape {memory_results.shape}')
    print(f'Loaded LiteFlowNet-en velocity has shape {lfn_results.shape}')
    print(f'Loaded pyramid velocity has shape {pyramid_results.shape}')
    print(f'Loaded cross-correlation velocity has shape {cc_results.shape}')

    exit()

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

    if plot_velocity_error_line_plot:
        memory_vel_errors = []
        lfn_vel_errors = []
        pyramid_vel_errors = []
        cc_vel_errors = []

    if plot_vorticity_error_line_plot:
        memory_vor_errors = []
        lfn_vor_errors = []
        pyramid_vor_errors = []
        cc_vor_errors = []

    # visualizing the results
    for i in tqdm(range(time_range[0], time_range[1])):
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
                cur_loss = np.sqrt(np.square(cur_true_velocity - cur_memory_velocity).mean(axis=None))
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
                cur_loss = np.sqrt(np.square(cur_true_velocity - cur_lfn_velocity).mean(axis=None))
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
                cur_loss = np.sqrt(np.square(cur_true_velocity - cur_pyramid_velocity).mean(axis=None))
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
                cur_loss = np.sqrt(np.square(cur_true_velocity - cur_cc_velocity).mean(axis=None))
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
                cur_loss = np.sqrt(np.square(cur_true_velocity - cur_memory_velocity).mean(axis=None))
            axes[1].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')

            # lite-flow-net-en
            axes[2].imshow(lfn_flow_pred)
            axes[2].quiver(y_pos[::skip, ::skip],
                            x_pos[::skip, ::skip],
                            cur_lfn_velocity[::skip, ::skip, 0]/max_vel,
                            -cur_lfn_velocity[::skip, ::skip, 1]/max_vel,
                            scale=Q.scale,
                            scale_units='inches')
            axes[2].set_title('LiteFlowNet-en')
            axes[2].set_xlabel('x')
            axes[2].set_ylabel('y')
            # compute and annotate loss
            if loss == 'MSE':
                cur_loss = np.square(cur_true_velocity - cur_lfn_velocity).mean(axis=None)
            elif loss == 'RMSE':
                cur_loss = np.sqrt(np.square(cur_true_velocity - cur_lfn_velocity).mean(axis=None))
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
                cur_loss = np.sqrt(np.square(cur_true_velocity - cur_pyramid_velocity).mean(axis=None))
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
                cur_loss = np.sqrt(np.square(cur_true_velocity - cur_cc_velocity).mean(axis=None))
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
                cur_loss = np.sqrt(np.square(cur_true_energy - cur_memory_energy).mean(axis=None))
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
                cur_loss = np.sqrt(np.square(cur_true_energy - cur_lfn_energy).mean(axis=None))
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
                cur_loss = np.sqrt(np.square(cur_true_energy - cur_pyramid_energy).mean(axis=None))
            axes[3].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')


            # cross-correlation
            axes[4].imshow(cur_cc_energy, vmin=cmap_range[0], vmax=cmap_range[1], cmap=plt.get_cmap('viridis'))
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
            if loss == 'MSE(Energy)':
                cur_loss = np.square(cur_true_energy - cur_cc_energy).mean(axis=None)
            elif loss == 'RMSE':
                cur_loss = np.sqrt(np.square(cur_true_energy - cur_cc_energy).mean(axis=None))
            axes[4].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')


            # save the image
            energy_dir = os.path.join(figs_dir, 'energy_plot')
            os.makedirs(energy_dir, exist_ok=True)
            energy_path = os.path.join(energy_dir, f'energy_{str(i).zfill(4)}.png')
            plt.savefig(energy_path, bbox_inches='tight', dpi=my_dpi)
            fig.clf()
            plt.close(fig)
            # print(f'\nEnergy plot has been saved to {energy_path}')

        # vorticity plot
        if plot_vorticity:
            # compute energy
            cur_true_vorticity = compute_vorticity(cur_true_velocity/0.04)
            cur_memory_vorticity = compute_vorticity(cur_memory_velocity/0.04)
            cur_lfn_vorticity = compute_vorticity(cur_lfn_velocity/0.04)
            cur_pyramid_vorticity = compute_vorticity(cur_pyramid_velocity/0.04)
            cur_cc_vorticity = compute_vorticity(cur_cc_velocity/0.04)

            # plot includes four subplots
            fig, axes = plt.subplots(nrows=1, ncols=len(methods), figsize=(5*len(methods), 5))
            plt.suptitle(f't = {i}')
            skip = 7
            cmap_range = [-3e-3, 3e-3]

            # superimpose quiver plot on color-coded images
            max_vel = np.max(cur_true_velocity)
            # ground truth
            x = np.linspace(0, img_size-1, img_size)
            y = np.linspace(0, img_size-1, img_size)
            y_pos, x_pos = np.meshgrid(x, y)
            axes[0].imshow(cur_true_vorticity[..., 0], vmin=cmap_range[0], vmax=cmap_range[1], cmap=plt.get_cmap('bwr'))
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
            axes[1].imshow(cur_memory_vorticity, vmin=cmap_range[0], vmax=cmap_range[1], cmap=plt.get_cmap('bwr'))
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
                cur_loss = np.square(cur_true_vorticity - cur_memory_vorticity).mean(axis=None)
            elif loss == 'RMSE':
                cur_loss = np.sqrt(np.square(cur_true_vorticity - cur_memory_vorticity).mean(axis=None))
            axes[1].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')

            # lite-flow-net-en
            axes[2].imshow(cur_lfn_vorticity, vmin=cmap_range[0], vmax=cmap_range[1], cmap=plt.get_cmap('bwr'))
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
                cur_loss = np.square(cur_true_vorticity - cur_lfn_vorticity).mean(axis=None)
            elif loss == 'RMSE':
                cur_loss = np.sqrt(np.square(cur_true_vorticity - cur_lfn_vorticity).mean(axis=None))
            axes[2].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')

            # pyramid
            axes[3].imshow(cur_pyramid_vorticity, vmin=cmap_range[0], vmax=cmap_range[1], cmap=plt.get_cmap('bwr'))
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
                cur_loss = np.square(cur_true_vorticity - cur_pyramid_vorticity).mean(axis=None)
            elif loss == 'RMSE':
                cur_loss = np.sqrt(np.square(cur_true_vorticity - cur_pyramid_vorticity).mean(axis=None))
            axes[3].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')


            # cross-correlation
            axes[4].imshow(cur_cc_vorticity, vmin=cmap_range[0], vmax=cmap_range[1], cmap=plt.get_cmap('bwr'))
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
            if loss == 'MSE(Energy)':
                cur_loss = np.square(cur_true_vorticity - cur_cc_vorticity).mean(axis=None)
            elif loss == 'RMSE':
                cur_loss = np.sqrt(np.square(cur_true_vorticity - cur_cc_vorticity).mean(axis=None))
            axes[4].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')


            # save the image
            vorticity_dir = os.path.join(figs_dir, 'vorticity_plot')
            os.makedirs(vorticity_dir, exist_ok=True)
            vorticity_path = os.path.join(vorticity_dir, f'vorticity_{str(i).zfill(4)}.png')
            plt.savefig(vorticity_path, bbox_inches='tight', dpi=my_dpi)
            fig.clf()
            plt.close(fig)
            # print(f'\nEnergy plot has been saved to {energy_path}')

        # velocity error line plot
        if plot_velocity_error_line_plot:
            if loss == 'MSE':
                memory_vel_errors.append(np.square(cur_true_velocity - cur_memory_velocity).mean(axis=None))
                lfn_vel_errors.append(np.square(cur_true_velocity - cur_lfn_velocity).mean(axis=None))
                pyramid_vel_errors.append(np.square(cur_true_velocity - cur_pyramid_velocity).mean(axis=None))
                cc_vel_errors.append(np.square(cur_true_velocity - cur_cc_velocity).mean(axis=None))
            elif loss == 'RMSE':
                memory_vel_errors.append(np.sqrt(np.square(cur_true_velocity - cur_memory_velocity).mean(axis=None)))
                lfn_vel_errors.append(np.sqrt(np.square(cur_true_velocity - cur_lfn_velocity).mean(axis=None)))
                pyramid_vel_errors.append(np.sqrt(np.square(cur_true_velocity - cur_pyramid_velocity).mean(axis=None)))
                cc_vel_errors.append(np.sqrt(np.square(cur_true_velocity - cur_cc_velocity).mean(axis=None)))

        # vorticity error line plot
        if plot_vorticity_error_line_plot:
            if loss == 'MSE':
                memory_vor_errors.append(np.square(cur_true_vorticity - cur_memory_vorticity).mean(axis=None))
                lfn_vor_errors.append(np.square(cur_true_vorticity - cur_lfn_vorticity).mean(axis=None))
                pyramid_vor_errors.append(np.square(cur_true_vorticity - cur_pyramid_vorticity).mean(axis=None))
                cc_vor_errors.append(np.square(cur_true_vorticity - cur_cc_vorticity).mean(axis=None))
            elif loss == 'RMSE':
                memory_vor_errors.append(np.sqrt(np.square(cur_true_vorticity - cur_memory_vorticity).mean(axis=None)))
                lfn_vor_errors.append(np.sqrt(np.square(cur_true_vorticity - cur_lfn_vorticity).mean(axis=None)))
                pyramid_vor_errors.append(np.sqrt(np.square(cur_true_vorticity - cur_pyramid_vorticity).mean(axis=None)))
                cc_vor_errors.append(np.sqrt(np.square(cur_true_vorticity - cur_cc_vorticity).mean(axis=None)))


    if plot_velocity_error_line_plot:
        fig, ax = plt.subplots()
        ax.plot(vis_frame, memory_vel_errors, label='Memory-PIVnet')
        ax.plot(vis_frame, lfn_vel_errors, label='LiteFlowNet-en')
        ax.plot(vis_frame, pyramid_vel_errors, label='Pyramid')
        ax.plot(vis_frame, cc_vel_errors, label='CC')
        ax.set(xlabel='timestamp', ylabel=f'{loss}')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        vel_loss_curve_path = os.path.join(figs_dir, f'all_frame_velocity_losses.png')
        fig.savefig(vel_loss_curve_path, bbox_inches='tight', dpi=my_dpi)
        print(f'\nVelocity losses of all frames plot has been saved to {vel_loss_curve_path}')


    if plot_vorticity_error_line_plot:
        fig, ax = plt.subplots()
        ax.plot(vis_frame, memory_vor_errors, label='Memory-PIVnet')
        ax.plot(vis_frame, lfn_vor_errors, label='LiteFlowNet-en')
        ax.plot(vis_frame, pyramid_vor_errors, label='Pyramid')
        ax.plot(vis_frame, cc_vor_errors, label='CC')
        ax.set(xlabel='timestamp', ylabel=f'{loss}')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        vor_loss_curve_path = os.path.join(figs_dir, f'all_frame_vorticity_losses.png')
        fig.savefig(vor_loss_curve_path, bbox_inches='tight', dpi=my_dpi)
        print(f'\nVorticity losses of all frames plot has been saved to {vor_loss_curve_path}')


if __name__ == "__main__":
    main()
