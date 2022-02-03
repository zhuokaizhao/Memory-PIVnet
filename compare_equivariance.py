# the script compares how different methods preserve equivariance properties such as reflection
import os
import cv2
import glob
import h5py
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage import io
import matplotlib as mpl
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
from matplotlib.ticker import MaxNLocator

import plot
import vorticity_numpy


# the function read all the original test images
def read_images(image_dir):
    # image format
    image_type = None

    for filename in os.listdir(image_dir):
        if (os.path.isfile(os.path.join(image_dir, filename))
            and not filename.endswith('.npy')):
            # save image type if never been saved
            if image_type == None:
                image_type = filename.split('.')[1]
            # images have to be one type
            else:
                if not image_type == filename.split('.')[1]:
                    raise Exception('read_images: Images have to be one type')

    # get all the image and velocity paths as a list
    all_image_paths = glob.glob(os.path.join(image_dir, f'*.{image_type}'))

    # rank these images, maps and labels based on name index
    all_image_paths.sort(key = lambda x: x.split('/')[-1].split('_')[-1].split('.')[0])
    num_images = len(all_image_paths)

    print(f'{num_images} test images have been loaded')

    # load all the images and labels
    # load one image to know the image size
    image_size = io.imread(all_image_paths[0], as_gray=True).shape
    all_images = np.zeros((num_images, image_size[0], image_size[1]))
    for i in range(num_images):
        all_images[i] = io.imread(all_image_paths[i], as_gray=True)
        # all_images[i] = cv2.imread(all_image_paths[i], cv2.IMREAD_GRAYSCALE)

    return all_images


# return all_vorticity_fields
def compute_vorticity(cur_velocity_field, xx, yy):
    # all_velocity_fields has shape (height, width, 2)
    # curl function takes (dim, num_rows, num_cols)
    udata = np.moveaxis(cur_velocity_field, [0, 1, 2], [1, 2, 0])
    cur_vorticity = vorticity_numpy.curl(udata, xx=xx, yy=yy)

    return np.array(cur_vorticity)


def main():

    parser = argparse.ArgumentParser()
    # mode (velocity or vorticity)
    parser.add_argument('--mode', required=True, action='store', nargs=1, dest='mode')
    # data name (isotropic_1024 or rotationsl)
    parser.add_argument('--data', required=True, action='store', nargs=1, dest='data')
    # start and end t used when testing (both inclusive)
    parser.add_argument('--start-t', action='store', nargs=1, dest='start_t')
    parser.add_argument('--end-t', action='store', nargs=1, dest='end_t')
    # loss function
    parser.add_argument('-l', '--loss', action='store', nargs=1, dest='loss')
    # output figure directory
    parser.add_argument('-o', '--output-dir', action='store', nargs=1, dest='output_dir')

    args = parser.parse_args()
    mode = args.mode[0]
    data = args.data[0]
    start_t = int(args.start_t[0])
    end_t = int(args.end_t[0])
    loss = args.loss[0]
    output_dir = args.output_dir[0]
    my_dpi = 100

    if mode == 'velocity':
        if data == 'isotropic_1024':
            # corresponding data path or directory
            test_images_dir = '/home/zhuokai/Desktop/nvme1n1p1/Data/LMSI/Zhao_JHTDB/Isotropic_1024/Figs/test/z_662_762/50000/'
            # first is normal, second is reversed
            ground_truth_paths = ['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/true_vel_field/',
                                    '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/reversed_Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/true_vel_field/']

            # methods = ['memory-piv-net', 'pyramid', 'widim']
            # methods = ['pyramid']
            methods = ['widim']
            # result_dirs = [['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/blend_vel_field/',
            #                 '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/reversed_Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/blend_vel_field/'],
                            #  ['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/pyramid/TR_Pyramid(2,5)_MPd(1x8x8_50ov)_2x32x32.h5',
                            #   '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/reversed_Isotropic_1024/velocity/pyramid/TR_Pyramid(2,5)_MPd(1x8x8_50ov).h5']]
            result_dirs = [['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/widim/TR_PIV_MPd(1x8x8_50ov)_2x32x32.h5',
                            '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/reversed_Isotropic_1024/velocity/widim/TR_PIV_MPd(1x16x16_50ov).h5']]


        elif data == 'rotational':
            # corresponding data path or directory
            test_images_dir = '/home/zhuokai/Desktop/nvme1n1p1/Data/LMSI/Zhao_JHTDB/Rotational/Figs/test/z_662_762/4000/'
            ground_truth_paths = ['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Rotational/velocity/amnesia_memory/4000_seeds/time_span_5/true_vel_field/',
                                    '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/reversed_Rotational/velocity/amnesia_memory/4000_seeds/time_span_5/true_vel_field/']
            methods = ['memory-piv-net']
            result_dirs = ['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Rotational/velocity/amnesia_memory/4000_seeds/time_span_5/blend_vel_field/',
                            '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/reversed_Rotational/velocity/amnesia_memory/4000_seeds/time_span_5/blend_vel_field/']
        else:
            raise(Exception(f'Unknown dataset {data}'))

    # when vorticity, still velocity results is loaded except ground truth and memory-piv-net
    elif mode == 'vorticity':
        ground_truth_path = '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/vorticity/amnesia_memory/50000_seeds/no_pe/time_span_5/true_vor_field/'
        # list of methods
        methods = ['memory-piv-net', 'memory-piv-net-velocity', 'pyramid', 'widim']
        result_dirs = ['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/vorticity/amnesia_memory/50000_seeds/no_pe/time_span_5/blend_vor_field/',
                        '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/amnesia_memory/50000_seeds/no_pe/time_span_5/blend_vel_field/',
                        '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/pyramid/TR_Pyramid(2,5)_MPd(1x8x8_50ov)_2x32x32.h5',
                        '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/widim/TR_PIV_MPd(1x8x8_50ov)_2x32x32.h5']
    else:
        raise Exception(f'Unknown mode {mode}')


    # sanity check
    # if len(methods) != len(result_dirs):
    #     raise Exception(f'Number of methods should equal to number of result paths (both normal and reversed)')

    # start and end time (both inclusive)
    time_range = [start_t, end_t]
    # frame 81, 153, 154 have broken ground truth
    if data == 'isotropic_1024':
        non_vis_frames = [80, 81, 153, 154]
    else:
        non_vis_frames = []
    img_size = 256

    # frames that we are visualizing
    vis_frames = list(range(time_range[0], time_range[1]+1))
    for i in non_vis_frames:
        if i in vis_frames:
            vis_frames.remove(i)

    # different types of visualizations
    if mode == 'velocity':
        blur_ground_truth = False

        plot_particle_density = False
        plot_image_quiver = False
        plot_color_encoded = False
        plot_loss_magnitude_heatmap = False
        plot_energy = False
        plot_error_line_plot = True
        plot_result_pdf = False
        plot_error_pdf = False
        plot_scatter = False
    elif mode == 'vorticity':
        blur_ground_truth = False

        plot_particle_density = False
        plot_image_quiver = False
        plot_color_encoded = False
        plot_loss_magnitude_heatmap = False
        plot_energy = False
        plot_error_line_plot = True
        plot_result_pdf = False
        plot_error_pdf = False
        plot_scatter = True

    # loaded velocity fields
    ground_truth = [{}, {}]
    results_all_methods = [{}, {}]
    if plot_error_line_plot:
        errors_all_methods = [{}, {}]
        energy_errors_all_methods = [{}, {}]

    if plot_scatter:
        all_truth_error_pairs = [{}, {}]

    # load ground truth of both normal and reversed
    # for sanity check
    all_x = [[], []]
    all_y = [[], []]
    all_x_rms = [[], []]
    all_y_rms = [[], []]
    for k, order in enumerate(['normal', 'reversed']):
        for t in range(time_range[0], time_range[1]+1):
            cur_path = os.path.join(ground_truth_paths[k], f'true_{mode}_{t}.npz')
            ground_truth[k][str(t)] = np.load(cur_path)[f'{mode}']

            all_x[k].append(np.mean(np.abs(ground_truth[k][str(t)][:, :, 0])))
            all_y[k].append(np.mean(np.abs(ground_truth[k][str(t)][:, :, 1])))
            all_x_rms[k].append(np.sqrt(np.sum(ground_truth[k][str(t)][:, :, 0]**2) / (256 * 256)))
            all_y_rms[k].append(np.sqrt(np.sum(ground_truth[k][str(t)][:, :, 1]**2) / (256 * 256)))

        print(f'\nAverage {order} velocity in x is {np.mean(all_x[k])} pixels/frame')
        print(f'Average {order} velocity in y is {np.mean(all_y[k])} pixels/frame')
        print(f'Average {order} RMS velocity in x is {np.mean(all_x_rms[k])} pixels/frame')
        print(f'Average {order} RMS velocity in y is {np.mean(all_y_rms[k])} pixels/frame')

    # when computing vorticity, xx and yy grids are required
    if mode == 'vorticity':
        with h5py.File(result_dirs[2], mode='r') as f:
            xx, yy = f['x'][...], f['y'][...]

        print(f'Loaded ground truth {mode} has shape ({len(ground_truth)}, {ground_truth[str(time_range[0])].shape})')

    # load results from each method
    for k, order in enumerate(['normal', 'reversed']):
        for i, cur_method in enumerate(methods):
            results_all_methods[k][cur_method] = {}
            if plot_error_line_plot:
                errors_all_methods[k][cur_method] = []
                energy_errors_all_methods[k][cur_method] = []

            if cur_method == 'memory-piv-net':
                # load the velocity fields of the specified time range
                for t in range(time_range[0], time_range[1]+1):
                    cur_path = os.path.join(result_dirs[i][k], f'test_{mode}_blend_{t}.npz')
                    results_all_methods[k][cur_method][str(t)] = np.load(cur_path)[f'{mode}']

            if cur_method == 'LiteFlowNet-en':
                # load the velocity fields of the specified time range
                for t in range(time_range[0], time_range[1]+1):
                    cur_path = os.path.join(result_dirs[i][k], f'lfn_{mode}_{t}.npz')
                    results_all_methods[k][cur_method][str(t)] = np.load(cur_path)[f'{mode}']

            # for pyramid and standar methods
            elif cur_method == 'pyramid' or cur_method == 'widim':
                cur_path = result_dirs[i][k]
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
                        results_all_methods[k][cur_method][str(t)] = cur_velocity
                    elif mode == 'vorticity':
                        # compute vorticity from velocity
                        cur_vorticity = compute_vorticity(velocity[t], xx, yy)
                        results_all_methods[k][cur_method][str(t)] = cur_vorticity.repeat(ratio, axis=0).repeat(ratio, axis=1)

    # print all the shapes
    # for i, cur_method in enumerate(methods):
    #     print(f'Loaded {cur_method} {mode} has shape ({len(results_all_methods[cur_method])}, {results_all_methods[cur_method][str(time_range[0])].shape})')


    # load test images
    # all_pixel_values_sums = []
    # all_test_images = read_images(test_images_dir)

    # max velocity from ground truth is useful for normalization
    if mode == 'velocity':
        max_truth = 3
        min_truth = -3
    elif mode == 'vorticity':
        max_truth = 1
        min_truth = -1


    # visualizing the results
    for i in tqdm(vis_frames):

        # color encoding plots
        if plot_color_encoded:
            # plot ground truth and all the prediction results
            fig, axes = plt.subplots(nrows=2, ncols=len(methods)+1, figsize=(5*(len(methods)+1), 5*2))
            # plt.suptitle(f'Color-encoded {mode} quiver plot at t = {i}')
            skip = 7

            # visualize ground truth
            if mode == 'velocity':
                # k=1: normal, k=2: reversed
                for k, order in enumerate(['normal', 'reversed']):
                    # visualize
                    if order == 'normal':
                        flow_vis, _ = plot.visualize_flow(ground_truth[0][str(i)], max_vel=max_truth)
                    elif order == 'reversed':
                        flow_vis, _ = plot.visualize_flow(ground_truth[1][str(start_t+end_t-int(i))], max_vel=max_truth)
                    # convert to Image
                    flow_vis_image = Image.fromarray(flow_vis)
                    # show the image
                    axes[k, 0].imshow(flow_vis_image)

                    # superimpose quiver plot on color-coded images
                    x = np.linspace(0, img_size-1, img_size)
                    y = np.linspace(0, img_size-1, img_size)
                    y_pos, x_pos = np.meshgrid(x, y)
                    if order == 'normal':
                        Q = axes[k, 0].quiver(y_pos[::skip, ::skip],
                                            x_pos[::skip, ::skip],
                                            ground_truth[0][str(i)][::skip, ::skip, 0]/max_truth,
                                            -ground_truth[0][str(i)][::skip, ::skip, 1]/max_truth,
                                            # scale=4.0,
                                            scale_units='inches')
                    elif order == 'reversed':
                        Q = axes[k, 0].quiver(y_pos[::skip, ::skip],
                                            x_pos[::skip, ::skip],
                                            ground_truth[1][str(start_t+end_t-int(i))][::skip, ::skip, 0]/max_truth,
                                            -ground_truth[1][str(start_t+end_t-int(i))][::skip, ::skip, 1]/max_truth,
                                            # scale=4.0,
                                            scale_units='inches')
                    Q._init()
                    assert isinstance(Q.scale, float)
                    axes[0, 0].set_title(f'Ground truth', fontsize=18)
                    axes[k, 0].set_xlabel('x')
                    axes[k, 0].set_ylabel('y')

            elif mode == 'vorticity':
                # vorticity simply uses a heatmap-like color encoding
                axes[0].imshow(ground_truth[str(i)], vmin=min_truth, vmax=max_truth, cmap=plt.get_cmap('bwr'))
                axes[0].set_title(f'Ground truth')
                axes[0].set_xlabel('x')
                axes[0].set_ylabel('y')

            # for each method
            for j, cur_method in enumerate(methods):
                # k=1: normal, k=2: reversed
                for k, order in enumerate(['normal', 'reversed']):
                    if mode == 'velocity':
                        if order == 'normal':
                            flow_vis, _ = plot.visualize_flow(results_all_methods[k][cur_method][str(i)], max_vel=max_truth)
                        elif order == 'reversed':
                            flow_vis, _ = plot.visualize_flow(results_all_methods[k][cur_method][str(start_t+end_t-int(i))], max_vel=max_truth)

                        # convert to Image
                        flow_vis_image = Image.fromarray(flow_vis)
                        # show the image
                        axes[k, j+1].imshow(flow_vis_image)

                        # superimpose quiver plot on color-coded images
                        x = np.linspace(0, img_size-1, img_size)
                        y = np.linspace(0, img_size-1, img_size)
                        y_pos, x_pos = np.meshgrid(x, y)
                        if order == 'normal':
                            Q = axes[k, j+1].quiver(y_pos[::skip, ::skip],
                                                    x_pos[::skip, ::skip],
                                                    results_all_methods[k][cur_method][str(i)][::skip, ::skip, 0]/max_truth,
                                                    -results_all_methods[k][cur_method][str(i)][::skip, ::skip, 1]/max_truth,
                                                    scale_units='inches')
                        elif order == 'reversed':
                            Q = axes[k, j+1].quiver(y_pos[::skip, ::skip],
                                                    x_pos[::skip, ::skip],
                                                    -1*results_all_methods[k][cur_method][str(start_t+end_t-int(i))][::skip, ::skip, 0]/max_truth,
                                                    -1*-1*results_all_methods[k][cur_method][str(start_t+end_t-int(i))][::skip, ::skip, 1]/max_truth,
                                                    scale_units='inches')
                        Q._init()
                        assert isinstance(Q.scale, float)
                        if k == 0:
                            axes[k, j+1].set_title(f'{cur_method}', fontsize=18)
                            axes[k, 0].set_ylabel('Normal', fontsize=18)
                        elif k == 1:
                            axes[k, 0].set_ylabel('Reversed', fontsize=18)

                    elif mode == 'vorticity':
                        # vorticity simply uses a heatmap-like color encoding
                        axes[j+1].imshow(results_all_methods[cur_method][str(i)], vmin=min_truth, vmax=max_truth, cmap=plt.get_cmap('bwr'))
                        axes[j+1].set_title(f'{cur_method}')
                        axes[j+1].set_xlabel('x')
                        axes[j+1].set_ylabel('y')

                    # label error
                    if order == 'normal':
                        if loss == 'MAE':
                            cur_loss = np.abs(ground_truth[k][str(i)] - results_all_methods[k][cur_method][str(i)]).mean(axis=None)
                        elif loss == 'MSE':
                            cur_loss = np.square(ground_truth[k][str(i)] - results_all_methods[k][cur_method][str(i)]).mean(axis=None)
                        elif loss == 'RMSE':
                            cur_loss = np.sqrt(np.square(ground_truth[k][str(i)] - results_all_methods[k][cur_method][str(i)]).mean(axis=None))
                    elif order == 'reversed':
                        if loss == 'MAE':
                            cur_loss = np.abs(ground_truth[k][str(start_t+end_t-int(i))] - results_all_methods[k][cur_method][str(start_t+end_t-int(i))]).mean(axis=None)
                        elif loss == 'MSE':
                            cur_loss = np.square(ground_truth[k][str(start_t+end_t-int(i))] - results_all_methods[k][cur_method][str(start_t+end_t-int(i))]).mean(axis=None)
                        elif loss == 'RMSE':
                            cur_loss = np.sqrt(np.square(ground_truth[k][str(start_t+end_t-int(i))] - results_all_methods[k][cur_method][str(start_t+end_t-int(i))]).mean(axis=None))


                    if mode == 'velocity':
                        axes[k, j+1].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='black', fontsize='medium')
                    elif mode == 'vorticity':
                        axes[k, j+1].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='black', fontsize='medium')

            # save the image
            if blur_ground_truth:
                color_encoded_dir = os.path.join(output_dir, f'{mode}_color_encoded_blurred_dpi{my_dpi}')
            else:
                color_encoded_dir = os.path.join(output_dir, f'{mode}_color_encoded_dpi{my_dpi}')

            os.makedirs(color_encoded_dir, exist_ok=True)
            color_encoded_path = os.path.join(color_encoded_dir, f'{mode}_color_encoded_{str(i).zfill(4)}.png')
            plt.savefig(color_encoded_path, bbox_inches='tight', dpi=my_dpi)
            fig.clf()
            plt.close(fig)


        # velocity error line plot
        if plot_error_line_plot:
            for k, order in enumerate(['normal', 'reversed']):
                for j, cur_method in enumerate(methods):
                    if order == 'normal':
                        if loss == 'MAE':
                            errors_all_methods[k][cur_method].append(np.abs(ground_truth[k][str(i)] - results_all_methods[k][cur_method][str(i)]).mean(axis=None))
                            # energy_errors_all_methods[cur_method].append(np.abs(ground_truth_energy - cur_energy_all_methods[cur_method]).mean(axis=None))
                        elif loss == 'MSE':
                            errors_all_methods[k][cur_method].append(np.square(ground_truth[k][str(i)] - results_all_methods[k][cur_method][str(i)]).mean(axis=None))
                            # energy_errors_all_methods[cur_method].append(np.square(ground_truth_energy - cur_energy_all_methods[cur_method]).mean(axis=None))
                        elif loss == 'RMSE':
                            errors_all_methods[k][cur_method].append(np.sqrt(np.square(ground_truth[k][str(i)] - results_all_methods[k][cur_method][str(i)]).mean(axis=None)))
                            # energy_errors_all_methods[cur_method].append(np.sqrt(np.square(ground_truth_energy - cur_energy_all_methods[cur_method]).mean(axis=None)))

                    elif order == 'reversed':
                        if loss == 'MAE':
                            errors_all_methods[k][cur_method].append(np.abs(ground_truth[k][str(start_t+end_t-int(i))] - results_all_methods[k][cur_method][str(start_t+end_t-int(i))]).mean(axis=None))
                            # energy_errors_all_methods[cur_method].append(np.abs(ground_truth_energy - cur_energy_all_methods[cur_method]).mean(axis=None))
                        elif loss == 'MSE':
                            errors_all_methods[k][cur_method].append(np.square(ground_truth[k][str(start_t+end_t-int(i))] - results_all_methods[k][cur_method][str(start_t+end_t-int(i))]).mean(axis=None))
                            # energy_errors_all_methods[cur_method].append(np.square(ground_truth_energy - cur_energy_all_methods[cur_method]).mean(axis=None))
                        elif loss == 'RMSE':
                            errors_all_methods[k][cur_method].append(np.sqrt(np.square(ground_truth[k][str(start_t+end_t-int(i))] - results_all_methods[k][cur_method][str(start_t+end_t-int(i))]).mean(axis=None)))
                            # energy_errors_all_methods[cur_method].append(np.sqrt(np.square(ground_truth_energy - cur_energy_all_methods[cur_method]).mean(axis=None)))

            # cur_test_image = all_test_images[i]
            # pixel_values_sum = np.sum(cur_test_image)
            # all_pixel_values_sums.append(pixel_values_sum)

    if plot_error_line_plot:
        fig, ax = plt.subplots(figsize=(10, 3))
        plt.suptitle(f'{mode} {loss} for each frame')

        # error line plot ordering
        # methods_order = ['LiteFlowNet-en', 'memory-piv-net', 'widim', 'pyramid']
        colors = ['blue', 'orange', 'green']
        styles = ['solid', 'dashed']
        for k, order in enumerate(['normal', 'reversed']):
            cur_style = styles[k]
            for m, cur_method in enumerate(methods):
                color = colors[m]
                ax.plot(vis_frames, errors_all_methods[k][cur_method], label=f'{cur_method} {order}', linestyle=cur_style, c=color)
                # print average result
                cur_avg_loss = np.nanmean(errors_all_methods[k][cur_method])
                print(f'{order} {cur_method} {mode} {loss} = {cur_avg_loss}')

        # also plot the "quality" of the particle images
        # ax.plot(vis_frames, np.array(all_pixel_values_sums)/(255*10000.0), label=f'Pixel value count/10000')

        ax.set(xlabel='time step', ylabel=f'{loss}')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        if blur_ground_truth:
            vel_loss_curve_path = os.path.join(output_dir, f'all_frames_{mode}_losses_blurred_dpi{my_dpi}.png')
        else:
            vel_loss_curve_path = os.path.join(output_dir, f'all_frames_{mode}_losses_dpi{my_dpi}.png')
        fig.savefig(vel_loss_curve_path, bbox_inches='tight', dpi=my_dpi*2)
        print(f'\n{mode} {loss} of all frames plot has been saved to {vel_loss_curve_path}\n')


if __name__ == "__main__":
    main()