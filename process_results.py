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
def compute_vorticity(cur_velocity_field, xx, yy):
    # all_velocity_fields has shape (height, width, 2)
    # num_rows = cur_velocity_field.shape[0]
    # num_cols = cur_velocity_field.shape[1]
    # x, y = list(range(num_cols)), list(range(num_rows))
    # xx, yy = np.meshgrid(x, y)

    # curl function takes (dim, num_rows, num_cols)
    udata = np.moveaxis(cur_velocity_field, [0, 1, 2], [1, 2, 0])
    cur_vorticity = vorticity.curl(udata, xx=xx, yy=yy)

    return np.array(cur_vorticity)


def main():

    parser = argparse.ArgumentParser()
    # mode (velocity or vorticity)
    parser.add_argument('--mode', required=True, action='store', nargs=1, dest='mode')
    # start and end t used when testing (both inclusive)
    parser.add_argument('--start-t', action='store', nargs=1, dest='start_t')
    parser.add_argument('--end-t', action='store', nargs=1, dest='end_t')
    # loss function
    parser.add_argument('-l', '--loss', action='store', nargs=1, dest='loss')
    # output figure directory
    parser.add_argument('-o', '--output-dir', action='store', nargs=1, dest='output_dir')

    args = parser.parse_args()
    mode = args.mode[0]
    start_t = int(args.start_t[0])
    end_t = int(args.end_t[0])
    loss = args.loss[0]
    output_dir = args.output_dir[0]

    # corresponding data path or directory
    if mode == 'velocity':
        ground_truth_path = '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/amnesia_memory/50000_seeds/no_pe/time_span_5/true_vel_field/'
        # list of methods
        methods = ['memory-piv-net', 'LiteFlowNet-en', 'pyramid', 'cross_correlation']
        result_dirs = ['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/amnesia_memory/50000_seeds/no_pe/time_span_5/blend_vel_field/',
                        '/home/zhuokai/Desktop/UChicago/Research/PIV-LiteFlowNet-en-Pytorch/output/Isotropic_1024/50000_seeds/lfn_vel_field/',
                        '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/pyramid/TR_Pyramid(2,5)_MPd(1x8x8_50ov)_2x32x32.h5',
                        '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/cross_correlation/TR_PIV_MPd(1x8x8_50ov)_2x32x32.h5']
    # when vorticity, still velocity results is loaded except ground truth and memory-piv-net
    elif mode == 'vorticity':
        ground_truth_path = '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/vorticity/amnesia_memory/50000_seeds/no_pe/time_span_5/true_vor_field/'
        # list of methods
        methods = ['memory-piv-net', 'memory-piv-net-velocity', 'pyramid', 'cross_correlation']
        result_dirs = ['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/vorticity/amnesia_memory/50000_seeds/no_pe/time_span_5/blend_vor_field/',
                        '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/amnesia_memory/50000_seeds/no_pe/time_span_5/blend_vel_field/',
                        '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/pyramid/TR_Pyramid(2,5)_MPd(1x8x8_50ov)_2x32x32.h5',
                        '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/cross_correlation/TR_PIV_MPd(1x8x8_50ov)_2x32x32.h5']
    else:
        raise Exception(f'Unknown mode {mode}')


    # sanity check
    if len(methods) != len(result_dirs):
        raise Exception(f'Number of methods should equal to number of result paths')

    # start and end time (both inclusive)
    time_range = [start_t, end_t]
    # frame 81, 153, 154 have broken ground truth
    non_vis_frames = [81, 153, 154]
    img_size = 256
    my_dpi = 100

    # frames that we are visualizing
    vis_frames = list(range(time_range[0], time_range[1]+1))
    for i in non_vis_frames:
        if i in vis_frames:
            vis_frames.remove(i)

    # load model testing output
    # different types of visualizations
    if mode == 'velocity':
        plot_image_quiver = True
        plot_color_encoded = True
        plot_aee_heatmap = True
        plot_energy = True
        plot_error_line_plot = True
    elif mode == 'vorticity':
        # plot_image_quiver = False
        # plot_color_encoded = True
        # plot_aee_heatmap = True
        # plot_energy = False
        # plot_error_line_plot = True
        plot_image_quiver = False
        plot_color_encoded = False
        plot_aee_heatmap = False
        plot_energy = False
        plot_error_line_plot = False
        plot_probability_density = True

    # loaded velocity fields
    ground_truth = {}
    results_all_methods = {}
    if plot_error_line_plot:
        errors_all_methods = {}

    # load ground truth
    for t in range(time_range[0], time_range[1]+1):
        cur_path = os.path.join(ground_truth_path, f'true_{mode}_{t}.npz')
        ground_truth[str(t)] = np.load(cur_path)[f'{mode}']

    # when computing vorticity, xx and yy grids are required
    if mode == 'vorticity':
        with h5py.File(result_dirs[2], mode='r') as f:
            xx, yy = f['x'][...], f['y'][...]

    print(f'Loaded ground truth {mode} has shape ({len(ground_truth)}, {ground_truth[str(time_range[0])].shape})')


    # load results from each method
    for i, cur_method in enumerate(methods):

        results_all_methods[cur_method] = {}
        if plot_error_line_plot:
            errors_all_methods[cur_method] = []

        if cur_method == 'memory-piv-net':
            # load the velocity fields of the specified time range
            for t in range(time_range[0], time_range[1]+1):
                cur_path = os.path.join(result_dirs[i], f'test_{mode}_blend_{t}.npz')
                results_all_methods[cur_method][str(t)] = np.load(cur_path)[f'{mode}']

        if cur_method == 'memory-piv-net-velocity':

            # load the velocity fields of the specified time range
            for t in range(time_range[0], time_range[1]+1):
                cur_path = os.path.join(result_dirs[i], f'test_velocity_blend_{t}.npz')
                cur_velocity = np.load(cur_path)['velocity']

                # compute vorticity from it
                results_all_methods[cur_method][str(t)] = compute_vorticity(cur_velocity, xx, yy)

        if cur_method == 'LiteFlowNet-en':
            # load the velocity fields of the specified time range
            for t in range(time_range[0], time_range[1]+1):
                cur_path = os.path.join(result_dirs[i], f'lfn_{mode}_{t}.npz')
                results_all_methods[cur_method][str(t)] = np.load(cur_path)[f'{mode}']

        # for pyramid and standar methods
        elif cur_method == 'pyramid' or cur_method == 'cross_correlation':
            cur_path = result_dirs[i]
            with h5py.File(cur_path, mode='r') as f:
                # print('The h5 file contains ', list(f.keys()))
                ux, uy = f['ux'][...], f['uy'][...]

            velocity = np.stack((ux, uy))
            velocity = np.moveaxis(velocity, [0, 1, 2, 3], [3, 1, 2, 0])

            # upsampling pyramid or cc results to full image resolution by duplicating
            ratio = img_size // velocity.shape[1]
            for t in range(time_range[0], time_range[1]+1):
                if mode == 'velocity':
                    cur_velocity = velocity[t].repeat(ratio, axis=0).repeat(ratio, axis=1)
                    results_all_methods[cur_method][str(t)] = cur_velocity
                elif mode == 'vorticity':
                    # compute vorticity from velocity
                    cur_vorticity = compute_vorticity(velocity[t], xx, yy)
                    results_all_methods[cur_method][str(t)] = cur_vorticity.repeat(ratio, axis=0).repeat(ratio, axis=1)

    # print all the shapes
    for i, cur_method in enumerate(methods):
        print(f'Loaded {cur_method} {mode} has shape ({len(results_all_methods[cur_method])}, {results_all_methods[cur_method][str(time_range[0])].shape})')

    # max velocity from ground truth is useful for normalization
    if mode == 'velocity':
        max_truth = np.max(ground_truth)*0.3
        min_truth = np.min(ground_truth)*0.3
    elif mode == 'vorticity':
        max_truth = 1
        min_truth = -1

    # visualizing the results
    for i in tqdm(vis_frames):

        # test image superimposed quiver plot
        if plot_image_quiver:
            test_image_path = f'/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/amnesia_memory/50000_seeds/no_pe/time_span_5/test_images/test_velocity_{i}.png'
            test_image = Image.open(test_image_path)

            # each method is a subplot
            fig, axes = plt.subplots(nrows=1, ncols=len(methods), figsize=(5*len(methods), 5))
            plt.suptitle(f'Particle image quiver plot at t = {i}')
            skip = 7

            # superimpose quiver plot on color-coded images
            for j, cur_method in enumerate(methods):
                # current method
                x = np.linspace(0, img_size-1, img_size)
                y = np.linspace(0, img_size-1, img_size)
                y_pos, x_pos = np.meshgrid(x, y)
                axes[0].imshow(test_image, 'gray')
                Q = axes[j].quiver(y_pos[::skip, ::skip],
                                    x_pos[::skip, ::skip],
                                    results_all_methods[cur_method][str(i)][::skip, ::skip, 0]/max_truth,
                                    -results_all_methods[cur_method][str(i)][::skip, ::skip, 1]/max_truth,
                                    # scale=4.0,
                                    scale_units='inches',
                                    color='green')
                Q._init()
                assert isinstance(Q.scale, float)
                axes[j].set_title(f'{cur_method}')
                axes[j].set_xlabel('x')
                axes[j].set_ylabel('y')

                # label error when not ground truth
                if cur_method != 'ground_truth':
                    if loss == 'MSE':
                        cur_loss = np.square(ground_truth[str(i)] - results_all_methods[cur_method][str(i)]).mean(axis=None)
                    elif loss == 'RMSE':
                        cur_loss = np.sqrt(np.square(ground_truth[str(i)] - results_all_methods[cur_method][str(i)]).mean(axis=None))
                    axes[j].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')

            # save the image
            test_quiver_dir = os.path.join(output_dir, 'test_quiver')
            os.makedirs(test_quiver_dir, exist_ok=True)
            test_quiver_path = os.path.join(test_quiver_dir, f'test_quiver_{str(i).zfill(4)}.png')
            plt.savefig(test_quiver_path, bbox_inches='tight', dpi=my_dpi)
            fig.clf()
            plt.close(fig)
            # print(f'\nSuperimposed test quiver plot has been saved to {test_quiver_path}')


        # color encoding plots
        if plot_color_encoded:
            # plot ground truth and all the prediction results
            fig, axes = plt.subplots(nrows=1, ncols=len(methods)+1, figsize=(5*(len(methods)+1), 5))
            plt.suptitle(f'Color-encoded {mode} quiver plot at t = {i}')
            skip = 7

            # visualize ground truth
            if mode == 'velocity':
                flow_vis, _ = plot.visualize_flow(ground_truth[str(i)], max_vel=max_truth)
                # convert to Image
                flow_vis_image = Image.fromarray(flow_vis)
                # show the image
                axes[0].imshow(flow_vis_image)

                # superimpose quiver plot on color-coded images
                x = np.linspace(0, img_size-1, img_size)
                y = np.linspace(0, img_size-1, img_size)
                y_pos, x_pos = np.meshgrid(x, y)
                Q = axes[j].quiver(y_pos[::skip, ::skip],
                                    x_pos[::skip, ::skip],
                                    results_all_methods[cur_method][str(i)][::skip, ::skip, 0]/max_truth,
                                    -results_all_methods[cur_method][str(i)][::skip, ::skip, 1]/max_truth,
                                    # scale=4.0,
                                    scale_units='inches')
                Q._init()
                assert isinstance(Q.scale, float)
                axes[0].set_title(f'Ground truth')
                axes[0].set_xlabel('x')
                axes[0].set_ylabel('y')

            elif mode == 'vorticity':
                # vorticity simply uses a heatmap-like color encoding
                axes[0].imshow(ground_truth[str(i)], vmin=min_truth, vmax=max_truth, cmap=plt.get_cmap('bwr'))
                axes[0].set_title(f'Ground truth')
                axes[0].set_xlabel('x')
                axes[0].set_ylabel('y')

            # for each method
            for j, cur_method in enumerate(methods):
                if mode == 'velocity':
                    flow_vis, _ = plot.visualize_flow(results_all_methods[cur_method][str(i)], max_vel=max_truth)
                    # convert to Image
                    flow_vis_image = Image.fromarray(flow_vis)
                    # show the image
                    axes[j+1].imshow(flow_vis_image)

                    # superimpose quiver plot on color-coded images
                    x = np.linspace(0, img_size-1, img_size)
                    y = np.linspace(0, img_size-1, img_size)
                    y_pos, x_pos = np.meshgrid(x, y)
                    Q = axes[j].quiver(y_pos[::skip, ::skip],
                                        x_pos[::skip, ::skip],
                                        results_all_methods[cur_method][str(i)][::skip, ::skip, 0]/max_truth,
                                        -results_all_methods[cur_method][str(i)][::skip, ::skip, 1]/max_truth,
                                        # scale=4.0,
                                        scale_units='inches')
                    Q._init()
                    assert isinstance(Q.scale, float)
                    axes[j+1].set_title(f'{cur_method}')
                    axes[j+1].set_xlabel('x')
                    axes[j+1].set_ylabel('y')

                elif mode == 'vorticity':
                    # vorticity simply uses a heatmap-like color encoding
                    axes[j+1].imshow(results_all_methods[cur_method][str(i)], vmin=min_truth, vmax=max_truth, cmap=plt.get_cmap('bwr'))
                    axes[j+1].set_title(f'{cur_method}')
                    axes[j+1].set_xlabel('x')
                    axes[j+1].set_ylabel('y')

                # label error
                if loss == 'MSE':
                    cur_loss = np.square(ground_truth[str(i)] - results_all_methods[cur_method][str(i)]).mean(axis=None)
                elif loss == 'RMSE':
                    cur_loss = np.sqrt(np.square(ground_truth[str(i)] - results_all_methods[cur_method][str(i)]).mean(axis=None))

                if mode == 'velocity':
                    axes[j+1].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')
                elif mode == 'vorticity':
                    axes[j+1].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='black', fontsize='medium')

            # save the image
            color_encoded_dir = os.path.join(output_dir, f'{mode}_color_encoded')
            os.makedirs(color_encoded_dir, exist_ok=True)
            color_encoded_path = os.path.join(color_encoded_dir, f'{mode}_color_encoded_{str(i).zfill(4)}.png')
            plt.savefig(color_encoded_path, bbox_inches='tight', dpi=my_dpi)
            fig.clf()
            plt.close(fig)
            # print(f'\nColor-encoded plot has been saved to {color_encoded_path}')


        # aee heatmap plot
        if plot_aee_heatmap:

            # plot includes number of methods - 1 (no ground truth) subplots
            fig, axes = plt.subplots(nrows=1, ncols=len(methods), figsize=(5*len(methods), 5))
            plt.suptitle(f'{mode} average endpoint error at t = {i}')

            for j, cur_method in enumerate(methods):

                # average end point error for all the outputs
                if mode == 'velocity':
                    cmap_range = [0, 1]
                    cur_method_aee = np.sqrt((results_all_methods[cur_method][str(i)][:,:,0]-ground_truth[str(i)][:,:,0])**2 + (results_all_methods[cur_method][str(i)][:,:,1]-ground_truth[str(i)][:,:,1])**2)
                elif mode == 'vorticity':
                    cmap_range = [0, 1]
                    cur_method_aee = np.sqrt((results_all_methods[cur_method][str(i)][:,:,0]-ground_truth[str(i)][:,:,0])**2)

                axes[j].imshow(cur_method_aee, vmin=cmap_range[0], vmax=cmap_range[1], cmap=plt.get_cmap('viridis'))
                axes[j].set_title(f'{cur_method}')
                axes[j].set_xlabel('x')
                axes[j].set_ylabel('y')
                axes[j].annotate(f'AEE: ' + '{:.3f}'.format(cur_method_aee.mean()), (5, 10), color='white', fontsize='medium')


            # cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
            # plt.colorbar(im, cax=cax, **kw)

            # save the image
            aee_dir = os.path.join(output_dir, f'{mode}_aee_plot')
            os.makedirs(aee_dir, exist_ok=True)
            aee_path = os.path.join(aee_dir, f'{mode}_aee_{str(i).zfill(4)}.png')
            plt.savefig(aee_path, bbox_inches='tight', dpi=my_dpi)
            fig.clf()
            plt.close(fig)
            # print(f'\nAEE plot has been saved to {aee_path}')


        # energy plot
        if plot_energy:

            # plot includes four subplots
            fig, axes = plt.subplots(nrows=1, ncols=len(methods), figsize=(5*len(methods), 5))
            plt.suptitle(f'Energy plot at t = {i}')
            skip = 7

            # compute energy for ground truth
            ground_truth_energy = get_energy(ground_truth[str(i)])
            energy_cmap_range = [np.min(ground_truth_energy), np.max(ground_truth_energy)]

            # superimpose quiver plot on color-coded images
            x = np.linspace(0, img_size-1, img_size)
            y = np.linspace(0, img_size-1, img_size)
            y_pos, x_pos = np.meshgrid(x, y)
            axes[0].imshow(ground_truth_energy, vmin=energy_cmap_range[0], vmax=energy_cmap_range[1], cmap=plt.get_cmap('viridis'))
            Q = axes[j+1].quiver(y_pos[::skip, ::skip],
                                x_pos[::skip, ::skip],
                                ground_truth[str(i)][::skip, ::skip, 0]/max_truth,
                                -ground_truth[str(i)][::skip, ::skip, 1]/max_truth,
                                # scale=4.0,
                                scale_units='inches',
                                color='black')
            Q._init()
            assert isinstance(Q.scale, float)
            axes[0].set_title('Ground truth')
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('y')

            # plot each prediction method
            for j, cur_method in enumerate(methods):
                cur_energy = get_energy(results_all_methods[cur_method][str(i)])
                axes[j+1].imshow(cur_energy, vmin=energy_cmap_range[0], vmax=energy_cmap_range[1], cmap=plt.get_cmap('viridis'))
                Q = axes[j+1].quiver(y_pos[::skip, ::skip],
                                    x_pos[::skip, ::skip],
                                    results_all_methods[cur_method][str(i)][::skip, ::skip, 0]/max_truth,
                                    -results_all_methods[cur_method][str(i)][::skip, ::skip, 1]/max_truth,
                                    # scale=4.0,
                                    scale_units='inches',
                                    color='black')
                Q._init()
                assert isinstance(Q.scale, float)
                axes[j].set_title('Ground truth')
                axes[j].set_xlabel('x')
                axes[j].set_ylabel('y')

                # compute and annotate loss
                if loss == 'MSE(Energy)':
                    cur_loss = np.square(ground_truth_energy - cur_energy).mean(axis=None)
                elif loss == 'RMSE':
                    cur_loss = np.sqrt(np.square(ground_truth_energy - cur_energy).mean(axis=None))
                elif loss == 'RMSE_relative':
                    cur_loss = np.sqrt(np.square(ground_truth_energy - cur_energy).mean(axis=None)) / ground_truth_energy.mean(axis=None)

                # add annotation
                if loss == 'RMSE_relative':
                    axes[j].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')
                else:
                    axes[j].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss) + '%', (5, 10), color='white', fontsize='medium')


            # save the image
            energy_dir = os.path.join(output_dir, 'energy_plot')
            os.makedirs(energy_dir, exist_ok=True)
            energy_path = os.path.join(energy_dir, f'energy_{str(i).zfill(4)}.png')
            plt.savefig(energy_path, bbox_inches='tight', dpi=my_dpi)
            fig.clf()
            plt.close(fig)


        # velocity error line plot
        if plot_error_line_plot:
            for j, cur_method in enumerate(methods):
                if loss == 'MSE':
                    errors_all_methods[cur_method].append(np.square(ground_truth[str(i)] - results_all_methods[cur_method][str(i)]).mean(axis=None))
                elif loss == 'RMSE':
                    errors_all_methods[cur_method].append(np.sqrt(np.square(ground_truth[str(i)] - results_all_methods[cur_method][str(i)]).mean(axis=None)))


        # plot the pdf of the array
        if plot_probability_density:

            fig, ax = plt.subplots()
            plt.suptitle(f'Probability density at t = {i}')
            num_bins = 100

            # plot the ground truth first
            gt_hist, bins = np.histogram(ground_truth[str(i)].flatten(), num_bins, density=True)
            ax.plot(bins[:num_bins], gt_hist, label='ground truth')

            # plot each prediction method
            for j, cur_method in enumerate(methods):
                cur_hist, _ = np.histogram(results_all_methods[cur_method][str(i)].flatten(), bins=bins, density=True)
                ax.plot(bins[:num_bins], cur_hist, label=cur_method)

            plt.legend()
            plt.xlabel('Vorticity')
            plt.ylabel('Probability density')
            plt.yscale('log')
            pdf_dir = os.path.join(output_dir, 'probability_density')
            os.makedirs(pdf_dir, exist_ok=True)
            pdf_path = os.path.join(pdf_dir, f'pdf_{str(i).zfill(4)}.png')
            plt.savefig(pdf_path, bbox_inches='tight', dpi=my_dpi)
            fig.clf()
            plt.close(fig)


    if plot_error_line_plot:
        fig, ax = plt.subplots()
        plt.suptitle(f'{mode} {loss} for each frame')

        for j, cur_method in enumerate(methods):
            ax.plot(vis_frames, errors_all_methods[cur_method], label=f'{cur_method}')

        ax.set(xlabel='timestamp', ylabel=f'{loss}')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        vel_loss_curve_path = os.path.join(output_dir, f'all_frames_{mode}_losses.png')
        fig.savefig(vel_loss_curve_path, bbox_inches='tight', dpi=my_dpi)
        print(f'\n{mode} losses of all frames plot has been saved to {vel_loss_curve_path}')


if __name__ == "__main__":
    main()
