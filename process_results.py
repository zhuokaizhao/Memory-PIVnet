# the script serves to process and visualize the results from various methods
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
    # curl function takes (dim, num_rows, num_cols)
    udata = np.moveaxis(cur_velocity_field, [0, 1, 2], [1, 2, 0])
    cur_vorticity = vorticity_numpy.curl(udata, xx=xx, yy=yy)

    return np.array(cur_vorticity)


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


# the function detects particles in a particle image
def detect_particle_locations(particle_image, vis=False):

    gray = cv2.GaussianBlur(particle_image, (5, 5), 0).astype('uint8')
    max_value = 255
    adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C#cv2.ADAPTIVE_THRESH_MEAN_C
    threshold_type = cv2.THRESH_BINARY#cv2.THRESH_BINARY_INV
    # odd number like 3, 5, 7, 9, 11
    block_size = 5
    # constant to be subtracted
    c = -5
    # threshold the image
    im_thresholded = cv2.adaptiveThreshold(gray, max_value, adaptive_method, threshold_type, block_size, c)
    # im_thresholded_partial = im_thresholded[10:20, 10:20]

    # label the particles (consider symmetric shape)
    label_array, particle_count = ndimage.measurements.label(im_thresholded)

    # get all particle locations
    all_particle_locations = np.array(ndimage.measurements.center_of_mass(im_thresholded, label_array, index=list(range(1, particle_count+1))))

    # visualize the processed particle image
    if vis:
        fig, axes = plt.subplots(ncols=3, figsize=(24, 8))
        axes[0].imshow(particle_image, cmap='gray', aspect='auto', origin='lower')
        axes[0].invert_yaxis()
        axes[0].set_aspect('equal', 'box')
        axes[0].set_title('Original image')

        axes[1].imshow(im_thresholded)
        axes[1].plot(all_particle_locations[:, 1], all_particle_locations[:, 0], '.', color='blue', markersize=5, alpha=0.5)
        axes[1].set_aspect('equal', 'box')
        axes[1].set_title('Detected particles')

        axes[2].imshow(particle_image, cmap='gray', aspect='auto', origin='lower')
        axes[2].plot(all_particle_locations[:, 1], all_particle_locations[:, 0], '.', color='red', markersize=5, alpha=0.5)
        axes[2].invert_yaxis()
        axes[2].set_aspect('equal', 'box')
        axes[2].set_title('Overlay detected particles with original image')
        # plt.show()
        particle_count_path = '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/figs/Isotropic_1024/velocity/50000_seeds/time_span_5/particle_count.png'
        fig.savefig(particle_count_path, bbox_inches='tight', dpi=100)

        exit()

    return all_particle_locations



def main():

    parser = argparse.ArgumentParser()
    # mode (velocity or vorticity, or both)
    parser.add_argument('--mode', required=True, action='store', nargs=1, dest='mode')
    # data name (isotropic_1024 or rotational)
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

    # corresponding data path or directory
    test_images_dir = '/home/zhuokai/Desktop/nvme1n1p1/Data/LMSI/Zhao_JHTDB/Isotropic_1024/Figs/test/z_662_762/50000/'
    # load these test images
    all_pixel_values_sums = []
    all_test_images = read_images(test_images_dir)

    # when we are only evaluating velocity
    if mode == 'velocity':
        if data == 'isotropic_1024':
            # ground_truth_path = '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/original/true_vel_field/'
            # methods = ['memory-piv-net-3', 'memory-piv-net-5', 'LiteFlowNet-en', 'pyramid', 'widim']
            # result_dirs = ['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_3/blend_vel_field/',
            #                 '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/blend_vel_field/',
            #                 '/home/zhuokai/Desktop/UChicago/Research/PIV-LiteFlowNet-en-Pytorch/output/Isotropic_1024/50000_seeds/lfn_vel_field/',
            #                 '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/pyramid/TR_Pyramid(2,5)_MPd(1x8x8_50ov)_2x32x32.h5',
            #                 '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/widim/TR_PIV_MPd(1x8x8_50ov)_2x32x32.h5']

            # for showing augmentations
            methods = ['mpn-5', 'mpn-5-aug']
            ground_truth_path = {'velocity': ['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/original/true_vel_field/',
                                              '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/rot90/true_vel_field/',
                                              '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/rot180/true_vel_field/',
                                              '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/rot270/true_vel_field/']}

            result_dirs = [['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/rot270/blend_vel_field/',
                            '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024_augmented/velocity/amnesia_memory/50000_seeds/time_span_5/epoch25/rot270/blend_vel_field/'],
                            ['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/rot270/blend_vel_field/',
                            '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024_augmented/velocity/amnesia_memory/50000_seeds/time_span_5/epoch25/rot270/blend_vel_field/'],
                            ['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/rot270/blend_vel_field/',
                            '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024_augmented/velocity/amnesia_memory/50000_seeds/time_span_5/epoch25/rot270/blend_vel_field/'],
                            ['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/rot270/blend_vel_field/',
                            '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024_augmented/velocity/amnesia_memory/50000_seeds/time_span_5/epoch25/rot270/blend_vel_field/']]
        elif data == 'rotational':
            ground_truth_path = {'velocity': '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Rotational/velocity/amnesia_memory/4000_seeds/no_pe/time_span_5/true_vel_field/'}
            methods = ['memory-piv-net', 'LiteFlowNet-en', 'pyramid', 'widim']
            result_dirs = ['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Rotational/velocity/amnesia_memory/4000_seeds/no_pe/time_span_5/blend_vel_field/',
                            '/home/zhuokai/Desktop/UChicago/Research/PIV-LiteFlowNet-en-Pytorch/output/Rotational/4000_seeds/lfn_vel_field/',
                            '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Rotational/velocity/pyramid/TR_Pyramid(1,3)_MPd(1x16x16_50ov).h5',
                            '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Rotational/velocity/widim/TR_PIV_MPd(1x16x16_50ov).h5']
        else:
            raise(Exception(f'Unknown dataset {data}'))

    # when vorticity, still velocity results is loaded except ground truth and memory-piv-net
    elif mode == 'vorticity':
        ground_truth_path = {'vorticity': '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/vorticity/amnesia_memory/50000_seeds/no_pe/time_span_5/true_vor_field/'}
        # list of methods
        methods = ['memory-piv-net', 'memory-piv-net-velocity', 'pyramid', 'widim']
        result_dirs = ['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/vorticity/amnesia_memory/50000_seeds/no_pe/time_span_5/blend_vor_field/',
                        '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/amnesia_memory/50000_seeds/no_pe/time_span_5/blend_vel_field/',
                        '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/pyramid/TR_Pyramid(2,5)_MPd(1x8x8_50ov)_2x32x32.h5',
                        '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/widim/TR_PIV_MPd(1x8x8_50ov)_2x32x32.h5']

    # when we are evaluating both velocity and vorticity
    elif mode == 'both':
        ground_true_path = {'velocity': '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/original/true_vel_field/',
                            'vorticity': '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/vorticity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/true_vor_field/'}

        # list of methods
        methods = ['mpn-vel', 'mpn-vor', 'mpn-combined', 'pyramid']

        result_dirs = ['/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/original/blend_vel_field/',
                        '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/vorticity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/blend_vor_field/',
                        '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/combined_loss/blend_vel_field/',
                        '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/pyramid/TR_Pyramid(2,5)_MPd(1x8x8_50ov)_2x32x32.h5']

    else:
        raise Exception(f'Unknown mode {mode}')


    # sanity check
    if len(methods) != len(result_dirs):
        raise Exception(f'Number of methods should equal to number of result paths')

    # start and end time (both inclusive)
    time_range = [start_t, end_t]
    if data == 'isotropic_1024' or data == 'rotational':
        # frame 81, 153, 154 have broken ground truth for isotropic data
        non_vis_frames = [80, 81, 153, 154]
        img_size = 256
    else:
        non_vis_frames = []

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
        plot_color_encoded = True
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
        plot_scatter = False

    elif mode == 'both':
        blur_ground_truth = False

        plot_particle_density = False
        plot_image_quiver = False
        plot_color_encoded = False
        plot_loss_magnitude_heatmap = False
        plot_energy = False
        plot_error_line_plot = False
        plot_result_pdf = False
        plot_error_pdf = False
        plot_scatter = False
        plot_velocity_histogram = True


    # loaded velocity fields
    ground_truth = {}
    results_all_methods = {}
    if plot_error_line_plot:
        errors_all_methods = {}
        if plot_energy:
            energy_errors_all_methods = {}

    # load ground truth
    if mode == 'both':
        truth_types = ['velocity', 'vorticity']
    else:
        truth_types = [mode]

    for cur_type in truth_types:
        if cur_type == 'velocity':
            all_x = []
            all_y = []
            all_x_rms = []
            all_y_rms = []
        if cur_type == 'vorticity':
            all_v = []
            all_v_rms = []

        # type of ground truth
        ground_truth[cur_type] = {}
        for t in range(time_range[0], time_range[1]+1):
            cur_path = os.path.join(ground_true_path[cur_type], f'true_{cur_type}_{t}.npz')
            ground_truth[cur_type][str(t)] = np.load(cur_path)[f'{cur_type}']
            # for sanity check use
            if cur_type == 'velocity':
                all_x.append(np.mean(np.abs(ground_truth[cur_type][str(t)][:, :, 0])))
                all_y.append(np.mean(np.abs(ground_truth[cur_type][str(t)][:, :, 1])))
                all_x_rms.append(np.sqrt(np.sum(ground_truth[cur_type][str(t)][:, :, 0]**2) / (256 * 256)))
                all_y_rms.append(np.sqrt(np.sum(ground_truth[cur_type][str(t)][:, :, 1]**2) / (256 * 256)))
            elif cur_type == 'vorticity':
                all_v.append(np.mean(np.abs(ground_truth[cur_type][str(t)][:, :, 0])))
                all_v_rms.append(np.sqrt(np.sum(ground_truth[cur_type][str(t)][:, :, 0]**2) / (256 * 256)))

        print(f'\nLoaded ground truth {cur_type} has shape ({len(ground_truth[cur_type])}, {ground_truth[cur_type][str(time_range[0])].shape})')

        if cur_type == 'velocity':
            print(f'Average {cur_type} in x is {np.mean(all_x)} pixels/frame')
            print(f'Average {cur_type} in y is {np.mean(all_y)} pixels/frame')
            print(f'Average RMS {cur_type} in x is {np.mean(all_x_rms)} pixels/frame')
            print(f'Average RMS {cur_type} in y is {np.mean(all_y_rms)} pixels/frame')
        elif cur_type == 'vorticity':
            print(f'Average {cur_type} is {np.mean(all_v)}')
            print(f'Average RMS {cur_type} is {np.mean(all_v_rms)}')

    # load results from each method
    if mode == 'both':
        result_types = ['velocity', 'vorticity']
    else:
        result_types = [mode]

    # when computing vorticity from velocity, xx and yy grids are required
    if mode == 'vorticity' or mode == 'both':
        with h5py.File(result_dirs[3], mode='r') as f:
            xx, yy = f['x'][...], f['y'][...]

    for cur_type in result_types:
        results_all_methods[cur_type] = {}

        for i, cur_method in enumerate(methods):
            results_all_methods[cur_type][cur_method] = {}

            # load from memory-piv-net and its variants
            if ('memory-piv-net' in cur_method or 'mpn' in cur_method) and not 'vor' in cur_method:
                # if we are loading velocities
                if cur_type == 'velocity':
                    # load the velocity fields of the specified time range
                    for t in range(time_range[0], time_range[1]+1):
                        cur_path = os.path.join(result_dirs[i], f'test_{cur_type}_blend_{t}.npz')
                        results_all_methods[cur_type][cur_method][str(t)] = np.load(cur_path)[f'{cur_type}']

                # if we are loading vorticities (needs to be computed from velocity)
                elif cur_type == 'vorticity':
                    # compute the vorticity fields of the specified time range
                    for t in range(time_range[0], time_range[1]+1):
                        results_all_methods[cur_type][cur_method][str(t)] = compute_vorticity(results_all_methods['velocity'][cur_method][str(t)], xx, yy)

            # load from memory-piv-net that was trained on vorticity (which produces vorticity directly)
            if ('memory-piv-net' in cur_method or 'mpn' in cur_method) and 'vor' in cur_method:
                # velocity cannot be computed from vorticity
                if cur_type == 'velocity':
                    continue
                elif cur_type == 'vorticity':
                    for t in range(time_range[0], time_range[1]+1):
                        cur_path = os.path.join(result_dirs[i], f'test_{cur_type}_blend_{t}.npz')
                        results_all_methods[cur_type][cur_method][str(t)] = np.load(cur_path)[f'{cur_type}']

            # liteflownet result
            if cur_method == 'LiteFlowNet-en':
                if cur_type == 'velocity':
                    # load the velocity fields of the specified time range
                    for t in range(time_range[0], time_range[1]+1):
                        cur_path = os.path.join(result_dirs[i], f'lfn_{mode}_{t}.npz')
                        results_all_methods[cur_method][str(t)] = np.load(cur_path)[f'{mode}']
                if cur_type == 'vorticity':
                    # compute the vorticity fields of the specified time range
                    for t in range(time_range[0], time_range[1]+1):
                        results_all_methods[cur_type][cur_method][str(t)] = compute_vorticity(results_all_methods['velocity'][cur_method][str(t)], xx, yy)

            # for pyramid and standar methods
            elif cur_method == 'pyramid' or cur_method == 'widim':
                cur_path = result_dirs[i]
                with h5py.File(cur_path, mode='r') as f:
                    # print('The h5 file contains ', list(f.keys()))
                    ux, uy = f['ux'][...], f['uy'][...]

                velocity = np.stack((ux, uy))
                velocity = np.moveaxis(velocity, [0, 1, 2, 3], [3, 1, 2, 0])

                # upsampling pyramid or cc results to full image resolution by duplicating
                ratio = img_size // velocity.shape[1]
                for t in range(time_range[0], time_range[1]+1):
                    if cur_type == 'velocity':
                        cur_velocity = velocity[t].repeat(ratio, axis=0).repeat(ratio, axis=1)
                        results_all_methods[cur_type][cur_method][str(t)] = cur_velocity
                    elif cur_type == 'vorticity':
                        # compute vorticity from velocity
                        cur_vorticity = compute_vorticity(velocity[t], xx, yy)
                        results_all_methods[cur_type][cur_method][str(t)] = cur_vorticity.repeat(ratio, axis=0).repeat(ratio, axis=1)

    # print all the shapes
    for cur_type in result_types:
        print('\n')
        for i, cur_method in enumerate(methods):
            if 'vor' in cur_method and cur_type == 'velocity':
                continue
            num_frames = len(results_all_methods[cur_type][cur_method])
            print(f'Loaded {cur_method} {cur_type} has mean {np.nanmean(sum(results_all_methods[cur_type][cur_method].values()))/num_frames}, shape ({num_frames}, {results_all_methods[cur_type][cur_method][str(time_range[0])].shape})')

    # save npz if needed
    # result_path = os.path.join(output_dir, 'vorticity_slice.npz')
    # np.savez(result_path,
    #          ground_truth=ground_truth['0'],
    #          memory_piv_net=results_all_methods['memory-piv-net']['0'],
    #          memory_piv_net_velocity=results_all_methods['memory-piv-net-velocity']['0'],
    #          pyramid=results_all_methods['pyramid']['0'],
    #          widim=results_all_methods['widim']['0'])
    # print(results_all_methods['memory-piv-net']['0'].mean())
    # print(results_all_methods['memory-piv-net-velocity']['0'].mean())
    # exit()

    # initialize error array
    if plot_error_line_plot:
        for cur_type in result_types:
            errors_all_methods[cur_type] = {}
            for cur_method in methods:
                if cur_type == 'velocity' and 'vor' in cur_method:
                    continue
                errors_all_methods[cur_type][cur_method] = []
        # energy is not velocity/vorticity-specific
        if plot_energy and plot_error_line_plot:
            for cur_method in methods:
                energy_errors_all_methods[cur_method] = []

    # max velocity from ground truth is useful for normalization
    # format: [min, max]
    plot_data_range = {'velocity': [-3, 3], 'vorticity': [-1, 1]}

    # visualizing the results
    for i in tqdm(vis_frames):

        # determine if blurring true vorticity, where the blurring level is determined by particle counts,
        # can improve the recovered vorticity accuracy
        if blur_ground_truth:
            # load the image
            cur_test_image = all_test_images[i]
            # count the particles
            cur_particle_locations = detect_particle_locations(cur_test_image, vis=False)
            num_particles = len(cur_particle_locations)
            # square root the number of particles
            sqrt_num_particles = np.sqrt(num_particles)
            # compute the kernel size of blurring
            blur_kernel_size = int(256 / sqrt_num_particles)

            # blur true vorticity by this ratio
            ground_truth[str(i)] = cv2.blur(ground_truth[str(i)], (blur_kernel_size, blur_kernel_size))


        # stand-alone particle density plot
        if plot_particle_density:

            # load the image
            cur_test_image = all_test_images[i]
            cur_particle_locations = detect_particle_locations(cur_test_image, vis=False)
            x = cur_particle_locations[:, 1]
            y = cur_particle_locations[:, 0]

            # kernel-density estimate using Gaussian kernels
            k = gaussian_kde(np.vstack([x, y]))
            # kernel size
            kernel_size = 4
            xi, yi = np.mgrid[x.min():x.max():kernel_size, y.min():y.max():kernel_size]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))

            # from the list of particle positions, plot density heatmap
            fig, ax = plt.subplots()

            # alpha=0.5 will make the plots semitransparent
            ax.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5, shading='auto')

            # overlay test image
            ax.imshow(cur_test_image, cmap='gray', aspect='auto', origin='lower')
            ax.invert_yaxis()
            ax.set_aspect('equal', 'box')

            # save figure
            particle_density_dir = os.path.join(output_dir, f'particle_density_dpi{my_dpi}')
            os.makedirs(particle_density_dir, exist_ok=True)
            particle_density_path = os.path.join(particle_density_dir, f'particle_density_{str(i).zfill(4)}.png')
            fig.savefig(particle_density_path, bbox_inches='tight', dpi=my_dpi)
            fig.clf()
            plt.close(fig)


        # test image superimposed quiver plot
        if plot_image_quiver:
            test_image_path = f'/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/amnesia_memory/50000_seeds/no_pe/time_span_5/test_images/test_velocity_{i}.png'
            test_image = Image.open(test_image_path)

            # each method is a subplot
            fig, axes = plt.subplots(nrows=1, ncols=len(methods)+1, figsize=(5*(len(methods)+1), 5))
            plt.suptitle(f'Particle image quiver plot at t = {i}')
            skip = 7

            # draw the ground truth quiver
            x = np.linspace(0, img_size-1, img_size)
            y = np.linspace(0, img_size-1, img_size)
            y_pos, x_pos = np.meshgrid(x, y)
            axes[0].imshow(test_image, 'gray')
            Q = axes[0].quiver(y_pos[::skip, ::skip],
                                x_pos[::skip, ::skip],
                                ground_truth[str(i)][::skip, ::skip, 0]/max_truth,
                                -ground_truth[str(i)][::skip, ::skip, 1]/max_truth,
                                # scale=4.0,
                                scale_units='inches',
                                color='green')
            Q._init()
            assert isinstance(Q.scale, float)
            axes[0].set_title(f'Ground truth')
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('y')


            # superimpose quiver plot on color-coded images
            for j, cur_method in enumerate(methods):
                # all methods use the same test image
                axes[j+1].imshow(test_image, 'gray')
                Q = axes[j+1].quiver(y_pos[::skip, ::skip],
                                    x_pos[::skip, ::skip],
                                    results_all_methods[cur_method][str(i)][::skip, ::skip, 0]/max_truth,
                                    -results_all_methods[cur_method][str(i)][::skip, ::skip, 1]/max_truth,
                                    # scale=4.0,
                                    scale_units='inches',
                                    color='green')
                Q._init()
                assert isinstance(Q.scale, float)
                axes[j+1].set_title(f'{cur_method}')
                axes[j+1].set_xlabel('x')
                axes[j+1].set_ylabel('y')

                # label error
                if loss == 'MAE':
                    cur_loss = np.abs(ground_truth[str(i)] - results_all_methods[cur_method][str(i)]).mean(axis=None)
                elif loss == 'MSE':
                    cur_loss = np.square(ground_truth[str(i)] - results_all_methods[cur_method][str(i)]).mean(axis=None)
                elif loss == 'RMSE':
                    cur_loss = np.sqrt(np.square(ground_truth[str(i)] - results_all_methods[cur_method][str(i)]).mean(axis=None))
                axes[j].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='black', fontsize='medium')

            # save the image
            if blur_ground_truth:
                test_quiver_dir = os.path.join(output_dir, f'{mode}_quiver_plot_truth_blurred_dpi{my_dpi}')
            else:
                test_quiver_dir = os.path.join(output_dir, f'{mode}_quiver_plot_dpi{my_dpi}')

            os.makedirs(test_quiver_dir, exist_ok=True)
            test_quiver_path = os.path.join(test_quiver_dir, f'test_quiver_{str(i).zfill(4)}.png')
            plt.savefig(test_quiver_path, bbox_inches='tight', dpi=my_dpi)
            fig.clf()
            plt.close(fig)
            # print(f'\nSuperimposed test quiver plot has been saved to {test_quiver_path}')


        # color encoding plots
        if plot_color_encoded:
            for cur_type in result_types:
                # plot ground truth and all the prediction results
                if cur_type == 'velocity':
                    # velocity has less when method includes pure vorticity method
                    num_less = 0
                    for cur_method in methods:
                        if 'vor' in cur_method:
                            num_less += 1

                    fig, axes = plt.subplots(nrows=1, ncols=len(methods)+1-num_less, figsize=(5*(len(methods)+1-num_less), 5))
                elif cur_type == 'vorticity':
                    fig, axes = plt.subplots(nrows=1, ncols=len(methods)+1, figsize=(5*(len(methods)+1), 5))

                # plt.suptitle(f'Color-encoded {mode} quiver plot at t = {i}')
                # skip is for plotting arrows
                skip = 7

                # visualize ground truth
                if cur_type == 'velocity':
                    flow_vis, _ = plot.visualize_flow(ground_truth[cur_type][str(i)], max_vel=plot_data_range[cur_type][1])
                    # convert to Image
                    flow_vis_image = Image.fromarray(flow_vis)
                    # show the image
                    axes[0].imshow(flow_vis_image)

                    # superimpose quiver plot on color-coded images
                    x = np.linspace(0, img_size-1, img_size)
                    y = np.linspace(0, img_size-1, img_size)
                    y_pos, x_pos = np.meshgrid(x, y)
                    Q = axes[0].quiver(y_pos[::skip, ::skip],
                                        x_pos[::skip, ::skip],
                                        results_all_methods[cur_type][cur_method][str(i)][::skip, ::skip, 0]/plot_data_range[cur_type][1],
                                        -results_all_methods[cur_type][cur_method][str(i)][::skip, ::skip, 1]/plot_data_range[cur_type][1],
                                        # scale=4.0,
                                        scale_units='inches')
                    Q._init()
                    assert isinstance(Q.scale, float)
                    axes[0].set_title(f'Ground truth')
                    axes[0].set_xlabel('x')
                    axes[0].set_ylabel('y')

                elif cur_type == 'vorticity':
                    # vorticity simply uses a heatmap-like color encoding
                    axes[0].imshow(ground_truth[cur_type][str(i)], vmin=plot_data_range[cur_type][0], vmax=plot_data_range[cur_type][1], cmap=plt.get_cmap('bwr'))
                    axes[0].set_title(f'Ground truth')
                    axes[0].set_xlabel('x')
                    axes[0].set_ylabel('y')

                # for each method
                j = 0
                for cur_method in methods:
                    if cur_type == 'velocity':
                        if 'vor' in cur_method:
                            continue

                        flow_vis, _ = plot.visualize_flow(results_all_methods[cur_type][cur_method][str(i)], max_vel=plot_data_range[cur_type][1])
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
                                            results_all_methods[cur_type][cur_method][str(i)][::skip, ::skip, 0]/plot_data_range[cur_type][1],
                                            -results_all_methods[cur_type][cur_method][str(i)][::skip, ::skip, 1]/plot_data_range[cur_type][1],
                                            # scale=4.0,
                                            scale_units='inches')
                        Q._init()
                        assert isinstance(Q.scale, float)
                        axes[j+1].set_title(f'{cur_method}')
                        axes[j+1].set_xlabel('x')
                        axes[j+1].set_ylabel('y')

                    elif cur_type == 'vorticity':
                        # vorticity simply uses a heatmap-like color encoding
                        if 'mpn' in cur_method and not 'vor' in cur_method:
                            axes[j+1].imshow(results_all_methods[cur_type][cur_method][str(i)]*5, vmin=plot_data_range[cur_type][0], vmax=plot_data_range[cur_type][1], cmap=plt.get_cmap('bwr'))
                        else:
                            axes[j+1].imshow(results_all_methods[cur_type][cur_method][str(i)], vmin=plot_data_range[cur_type][0], vmax=plot_data_range[cur_type][1], cmap=plt.get_cmap('bwr'))
                        axes[j+1].set_title(f'{cur_method}')
                        axes[j+1].set_xlabel('x')
                        axes[j+1].set_ylabel('y')

                    # label error
                    if loss == 'MAE':
                        cur_loss = np.abs(ground_truth[cur_type][str(i)] - results_all_methods[cur_type][cur_method][str(i)]).mean(axis=None)
                    elif loss == 'MSE':
                        cur_loss = np.square(ground_truth[cur_type][str(i)] - results_all_methods[cur_type][cur_method][str(i)]).mean(axis=None)
                    elif loss == 'RMSE':
                        cur_loss = np.sqrt(np.square(ground_truth[cur_type][str(i)] - results_all_methods[cur_type][cur_method][str(i)]).mean(axis=None))

                    if cur_type == 'velocity':
                        axes[j+1].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='black', fontsize='medium')
                    elif cur_type == 'vorticity':
                        axes[j+1].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='black', fontsize='medium')

                    # increase counter
                    j += 1

                # save the image
                if blur_ground_truth:
                    color_encoded_dir = os.path.join(output_dir, f'{mode}_color_encoded_blurred_dpi{my_dpi}', cur_type)
                else:
                    color_encoded_dir = os.path.join(output_dir, f'{mode}_color_encoded_dpi{my_dpi}', cur_type)

                os.makedirs(color_encoded_dir, exist_ok=True)
                color_encoded_path = os.path.join(color_encoded_dir, f'{cur_type}_color_encoded_{str(i).zfill(4)}.png')
                plt.savefig(color_encoded_path, bbox_inches='tight', dpi=my_dpi)
                fig.clf()
                plt.close(fig)
                # print(f'\nColor-encoded plot has been saved to {color_encoded_path}')


        # loss heatmap
        if plot_loss_magnitude_heatmap:

            # plot includes number of methods - 1 (no ground truth) subplots
            fig, axes = plt.subplots(nrows=1, ncols=len(methods)+1, figsize=(5*len(methods), 5))
            # plt.suptitle(f'{mode} |{loss}| at t = {i}')

            # first subplot is the particle density
            # load the image
            cur_test_image = all_test_images[i]
            if plot_particle_density:
                cur_particle_locations = detect_particle_locations(cur_test_image, vis=False)
                x = cur_particle_locations[:, 1]
                y = cur_particle_locations[:, 0]

                # kernel-density estimate using Gaussian kernels
                k = gaussian_kde(np.vstack([x, y]))
                # kernel size
                kernel_size = 4
                xi, yi = np.mgrid[x.min():x.max():kernel_size, y.min():y.max():kernel_size]
                zi = k(np.vstack([xi.flatten(), yi.flatten()]))

                # alpha=0.5 will make the plots semitransparent
                axes[0].pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5, shading='auto')
                axes[0].set_title('Particle density plot')
            # else:
                # axes[0].set_title('Particle image')

            # overlay test image
            axes[0].imshow(cur_test_image, cmap='gray', aspect='auto', origin='lower')
            axes[0].invert_yaxis()
            axes[0].set_aspect('equal', 'box')

            # loss magnitude plots for each method
            for j, cur_method in enumerate(methods):

                # average end point error for all the outputs
                if mode == 'velocity':
                    cmap_range = [0, 2]
                    if loss == 'MAE':
                        cur_loss = np.abs(ground_truth[str(i)] - results_all_methods[cur_method][str(i)]).mean(axis=2)
                    elif loss == 'MSE':
                        cur_loss = np.square(ground_truth[str(i)] - results_all_methods[cur_method][str(i)]).mean(axis=2)
                    elif loss == 'RMSE':
                        cur_loss = np.sqrt(np.square(ground_truth[str(i)] - results_all_methods[cur_method][str(i)])).mean(axis=2)
                    elif loss == 'AEE':
                        cur_loss = np.sqrt((results_all_methods[cur_method][str(i)][:,:,0]-ground_truth[str(i)][:,:,0])**2 + (results_all_methods[cur_method][str(i)][:,:,1]-ground_truth[str(i)][:,:,1])**2)

                elif mode == 'vorticity':
                    cmap_range = [0, 0.1]
                    if loss == 'MAE':
                        cur_loss = np.abs(ground_truth[str(i)] - results_all_methods[cur_method][str(i)])
                    elif loss == 'MSE':
                        cur_loss = np.square(ground_truth[str(i)] - results_all_methods[cur_method][str(i)])
                    elif loss == 'RMSE':
                        cur_loss = np.sqrt(np.square(ground_truth[str(i)] - results_all_methods[cur_method][str(i)]))
                    elif loss == 'AEE':
                        cur_loss = np.sqrt((results_all_methods[cur_method][str(i)][:,:,0]-ground_truth[str(i)][:,:,0])**2)

                im = axes[j+1].imshow(cur_loss, vmin=cmap_range[0], vmax=cmap_range[1], cmap=plt.get_cmap('viridis'))
                # axes[j+1].set_title(f'{cur_method}')
                axes[j+1].set_xlabel('x')
                axes[j+1].set_ylabel('y')
                axes[j+1].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss.mean()), (5, 10), color='white', fontsize='medium')


            # add color bar at the last subplot
            # add space for colour bar
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.87, 0.245, 0.01, 0.501])
            fig.colorbar(im, cax=cbar_ax)

            # save the image
            if blur_ground_truth:
                loss_magnitude_dir = os.path.join(output_dir, f'{mode}_{loss}_magnitude_plot_blurred_dpi{my_dpi}')
            else:
                loss_magnitude_dir = os.path.join(output_dir, f'{mode}_{loss}_magnitude_plot_dpi{my_dpi}')

            os.makedirs(loss_magnitude_dir, exist_ok=True)
            aee_path = os.path.join(loss_magnitude_dir, f'{mode}_{loss}_{str(i).zfill(4)}.png')
            plt.savefig(aee_path, bbox_inches='tight', dpi=my_dpi)
            fig.clf()
            plt.close(fig)


        # energy computation from velocity fields
        if plot_energy and (mode == 'velocity' or mode == 'both'):
            # compute energy for ground truth
            ground_truth_energy = get_energy(ground_truth['velocity'][str(i)])
            # compute energy for each method (energy can only be computed from velocity)
            cur_energy_all_methods = {}
            for j, cur_method in enumerate(methods):
                cur_energy_all_methods[cur_method] = get_energy(results_all_methods['velocity'][cur_method][str(i)])

        # energy plot
        if plot_energy:

            # plot includes four subplots
            fig, axes = plt.subplots(nrows=1, ncols=len(methods)+1, figsize=(5*(len(methods)+1), 5))
            plt.suptitle(f'Energy plot at t = {i}')
            skip = 7


            energy_cmap_range = [0, 5]

            # superimpose quiver plot on color-coded images
            x = np.linspace(0, img_size-1, img_size)
            y = np.linspace(0, img_size-1, img_size)
            y_pos, x_pos = np.meshgrid(x, y)
            axes[0].imshow(ground_truth_energy, vmin=energy_cmap_range[0], vmax=energy_cmap_range[1], cmap=plt.get_cmap('viridis'))
            Q = axes[0].quiver(y_pos[::skip, ::skip],
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
                cur_energy = cur_energy_all_methods[cur_method]
                im = axes[j+1].imshow(cur_energy, vmin=energy_cmap_range[0], vmax=energy_cmap_range[1], cmap=plt.get_cmap('viridis'))
                Q = axes[j+1].quiver(y_pos[::skip, ::skip],
                                    x_pos[::skip, ::skip],
                                    results_all_methods[cur_method][str(i)][::skip, ::skip, 0]/max_truth,
                                    -results_all_methods[cur_method][str(i)][::skip, ::skip, 1]/max_truth,
                                    # scale=4.0,
                                    scale_units='inches',
                                    color='black')
                Q._init()
                assert isinstance(Q.scale, float)
                axes[j+1].set_title(f'{cur_method}')
                axes[j+1].set_xlabel('x')
                axes[j+1].set_ylabel('y')

                # compute and annotate loss
                if loss == 'MAE':
                    cur_loss = np.abs(ground_truth_energy - cur_energy).mean(axis=None)
                elif loss == 'MSE':
                    cur_loss = np.square(ground_truth_energy - cur_energy).mean(axis=None)
                elif loss == 'RMSE':
                    cur_loss = np.sqrt(np.square(ground_truth_energy - cur_energy).mean(axis=None))

                # add annotation
                axes[j+1].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='medium')

            # add color bar at the last subplot
            # add space for colour bar
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.87, 0.15, 0.01, 0.7])
            fig.colorbar(im, cax=cbar_ax)

            # save the image
            if blur_ground_truth:
                energy_dir = os.path.join(output_dir, f'energy_plot_blurred_dpi{my_dpi}')
            else:
                energy_dir = os.path.join(output_dir, f'energy_plot_dpi{my_dpi}')

            os.makedirs(energy_dir, exist_ok=True)
            energy_path = os.path.join(energy_dir, f'energy_{str(i).zfill(4)}.png')
            plt.savefig(energy_path, bbox_inches='tight', dpi=my_dpi)
            fig.clf()
            plt.close(fig)


        # velocity error line plot
        if plot_error_line_plot:
            for cur_type in result_types:

                for cur_method in methods:
                    if cur_type == 'velocity' and 'vor' in cur_method:
                        continue

                    if loss == 'MAE':
                        errors_all_methods[cur_type][cur_method].append(np.abs(ground_truth[cur_type][str(i)] - results_all_methods[cur_type][cur_method][str(i)]).mean(axis=None))
                        if plot_energy:
                            energy_errors_all_methods[cur_method].append(np.abs(ground_truth_energy - cur_energy_all_methods[cur_method]).mean(axis=None))
                    elif loss == 'MSE':
                        errors_all_methods[cur_type][cur_method].append(np.square(ground_truth[cur_type][str(i)] - results_all_methods[cur_type][cur_method][str(i)]).mean(axis=None))
                        if plot_energy:
                            energy_errors_all_methods[cur_method].append(np.square(ground_truth_energy - cur_energy_all_methods[cur_method]).mean(axis=None))
                    elif loss == 'RMSE':
                        errors_all_methods[cur_type][cur_method].append(np.sqrt(np.square(ground_truth[cur_type][str(i)] - results_all_methods[cur_type][cur_method][str(i)]).mean(axis=None)))
                        if plot_energy:
                            energy_errors_all_methods[cur_method].append(np.sqrt(np.square(ground_truth_energy - cur_energy_all_methods[cur_method]).mean(axis=None)))

            cur_test_image = all_test_images[i]
            pixel_values_sum = np.sum(cur_test_image)
            all_pixel_values_sums.append(pixel_values_sum)

        # plot result pdf
        if plot_result_pdf:

            num_bins = 100

            if mode == 'velocity':
                # plot the ground truth first
                fig, axes = plt.subplots(ncols=2)
                # plot x and y individually
                for k in range(2):
                    gt_hist, bins = np.histogram(ground_truth[str(i)][:, :, k].flatten(), num_bins, density=True)
                    axes[k].plot(bins[:num_bins], gt_hist, label='ground truth')

                    # plot each prediction method
                    for j, cur_method in enumerate(methods):
                        cur_hist, _ = np.histogram(results_all_methods[cur_method][str(i)][:, :, k].flatten(), bins=bins, density=True)
                        axes[k].plot(bins[:num_bins], cur_hist, label=cur_method)

            elif mode == 'vorticity':
                fig, ax = plt.subplots()
                gt_hist, bins = np.histogram(ground_truth[str(i)].flatten(), num_bins, density=True)
                if blur_ground_truth:
                    ax.plot(bins[:num_bins], gt_hist, label=f'ground truth (blurred, box {blur_kernel_size})')
                else:
                    ax.plot(bins[:num_bins], gt_hist, label=f'ground truth')

                # plot each prediction method
                for j, cur_method in enumerate(methods):
                    cur_hist, _ = np.histogram(results_all_methods[cur_method][str(i)].flatten(), bins=bins, density=True)
                    ax.plot(bins[:num_bins], cur_hist, label=cur_method)

            plt.suptitle(f'Probability density of {mode} at t = {i}')
            plt.legend()
            plt.xlabel(f'{mode}')
            plt.ylabel('Probability density')
            plt.yscale('log')

            if blur_ground_truth:
                pdf_dir = os.path.join(output_dir, f'{mode}_probability_density_blurred_dpi{my_dpi}')
                os.makedirs(pdf_dir, exist_ok=True)
                pdf_path = os.path.join(pdf_dir, f'{mode}_pdf_{str(i).zfill(4)}_blurred.png')
            else:
                pdf_dir = os.path.join(output_dir, f'{mode}_probability_density_dpi{my_dpi}')
                os.makedirs(pdf_dir, exist_ok=True)
                pdf_path = os.path.join(pdf_dir, f'{mode}_pdf_{str(i).zfill(4)}.png')

            plt.savefig(pdf_path, bbox_inches='tight', dpi=my_dpi)
            fig.clf()
            plt.close(fig)


        # plot error pdf
        if plot_error_pdf:

            num_bins = 100

            # plot each prediction method
            for j, cur_method in enumerate(methods):
                if mode == 'velocity':
                    fig, axes = plt.subplots(ncols=2)
                    for k in range(2):
                        # compute error
                        if loss == 'MAE':
                            cur_loss = np.abs(ground_truth[str(i)][:, :, k] - results_all_methods[cur_method][str(i)][:, :, k]).mean(axis=None)
                        elif loss == 'MSE':
                            cur_loss = np.square(ground_truth[str(i)][:, :, k] - results_all_methods[cur_method][str(i)][:, :, k]).mean(axis=None)
                        elif loss == 'RMSE':
                            cur_loss = np.sqrt(np.square(ground_truth[str(i)][:, :, k] - results_all_methods[cur_method][str(i)][:, :, k]).mean(axis=None))

                        # plot error pdf
                        if j == 0:
                            cur_hist, bins = np.histogram(cur_loss.flatten(), num_bins, density=True)
                        else:
                            cur_hist, _ = np.histogram(cur_loss.flatten(), bins=bins, density=True)

                        axes[k].plot(bins[:num_bins], cur_hist, label=cur_method)

                elif mode == 'vorticity':
                    fig, ax = plt.subplots()
                    # compute error
                    if loss == 'MAE':
                        cur_loss = np.abs(ground_truth[str(i)] - results_all_methods[cur_method][str(i)]).mean(axis=None)
                    elif loss == 'MSE':
                        cur_loss = np.square(ground_truth[str(i)] - results_all_methods[cur_method][str(i)]).mean(axis=None)
                    elif loss == 'RMSE':
                        cur_loss = np.sqrt(np.square(ground_truth[str(i)] - results_all_methods[cur_method][str(i)]).mean(axis=None))

                    # plot error pdf
                    if j == 0:
                        cur_hist, bins = np.histogram(cur_loss.flatten(), num_bins, density=True)
                    else:
                        cur_hist, _ = np.histogram(cur_loss.flatten(), bins=bins, density=True)

                    ax.plot(bins[:num_bins], cur_hist, label=cur_method)

            plt.suptitle(f'Probability density of {mode} error ({loss}) at t = {i}')
            plt.legend()
            plt.xlabel(f'{loss}')
            plt.ylabel('Probability density')
            # plt.xscale('log')
            plt.yscale('log')

            if blur_ground_truth:
                error_pdf_dir = os.path.join(output_dir, f'{mode}_error_probability_density_blurred_dpi{my_dpi}')
            else:
                error_pdf_dir = os.path.join(output_dir, f'{mode}_error_probability_density_dpi{my_dpi}')
            os.makedirs(error_pdf_dir, exist_ok=True)
            error_pdf_path = os.path.join(error_pdf_dir, f'{mode}_error_pdf_{str(i).zfill(4)}.png')
            plt.savefig(error_pdf_path, bbox_inches='tight', dpi=my_dpi)
            fig.clf()
            plt.close(fig)


        # scatter plot of error/ground truth, shows correlation between flow and error
        if plot_scatter:
            import seaborn as sns
            import pandas as pd
            import matplotlib.gridspec as gridspec

            # x-axis is ground truth, y axis is error
            fig = plt.figure(figsize=(5*len(methods), 5*2))
            gs = gridspec.GridSpec(2, len(methods))

            # import nrrd
            # true_path = os.path.join(output_dir, 'for_gordon', 'ground_truth', f'ground_truth_{i}.nrrd')
            # nrrd.write(true_path, ground_truth[str(i)])

            # plot each prediction method
            for j, cur_method in enumerate(methods):

                # save the method output
                # temp_path = os.path.join(output_dir, 'for_gordon', f'{cur_method}', f'{cur_method}_output_{i}.nrrd')
                # nrrd.write(temp_path, results_all_methods[cur_method][str(i)])
                # continue

                # compute error
                errors = results_all_methods[cur_method][str(i)] - ground_truth[str(i)]
                x_errors = list(errors[:, :, 0].flatten())
                x_truths = list(ground_truth[str(i)][:, :, 0].flatten())
                y_errors = list(errors[:, :, 1].flatten())
                y_truths = list(ground_truth[str(i)][:, :, 1].flatten())

                # put in pandas dataframe
                x_df = pd.DataFrame({
                    'v_x': x_truths,
                    'delta_v_x': x_errors
                })
                x_df.head(n=2)

                y_df = pd.DataFrame({
                    'v_y': y_truths,
                    'delta_v_y': y_errors
                })
                y_df.head(n=2)

                # scatter plots for x and y
                # joint_x = sns.jointplot(x='v_x', y='delta_v_x', data=x_df, kind='hist', xlim = (-4, 4), ylim = (-4, 4), joint_kws = {'hist_kws':dict(cbar=True)})
                joint_x = sns.jointplot(x='v_x', y='delta_v_x', data=x_df, kind='hist', xlim = (-4, 4), ylim = (-4, 4))
                joint_y = sns.jointplot(x='v_y', y='delta_v_y', data=y_df, kind='hist', xlim = (-4, 4), ylim = (-4, 4))

                plot.SeabornFig2Grid(joint_x, fig, gs[0, j])
                joint_x.ax_marg_x.set_title(f'{methods[j]}')
                # joint_x.ax_joint.set_aspect('equal')
                plot.SeabornFig2Grid(joint_y, fig, gs[1, j])
                # joint_y.ax_joint.set_aspect('equal')

            gs.tight_layout(fig)
            plt.show()

            # save the image
            # manutal screent shot is needed for some reasons....
            # if blur_ground_truth:
            #     scatter_dir = os.path.join(output_dir, f'error_scatter_plot_blurred_dpi{my_dpi}')
            # else:
            #     scatter_dir = os.path.join(output_dir, f'error_scatter_plot_dpi{my_dpi}')

            # os.makedirs(scatter_dir, exist_ok=True)
            # scatter_path = os.path.join(scatter_dir, f'error_scatter_{str(i).zfill(4)}.png')
            # plt.savefig(scatter_path, bbox_inches='tight', dpi=my_dpi)
            # fig.clf()
            # plt.close(fig)


    if plot_error_line_plot:
        for cur_type in result_types:
            fig, ax = plt.subplots(figsize=(10, 3))
            plt.suptitle(f'{cur_type} {loss} for each frame')

            # error line plot ordering
            if data == 'rotational':
                methods_order = ['LiteFlowNet-en', 'memory-piv-net', 'widim', 'pyramid']
                colors = ['blue', 'orange', 'red', 'green']
            elif data == 'isotropic_1024':
                # methods_order = ['LiteFlowNet-en', 'memory-piv-net-3', 'widim', 'pyramid', 'memory-piv-net-5']
                # colors = ['blue', 'purple', 'red', 'green', 'orange']
                # methods_order = ['memory-piv-net-3', 'memory-piv-net-9']
                # colors = ['black', 'orange']
                # methods_order = ['mpn-5', 'mpn-5-aug']
                # colors = ['black', 'orange']
                if cur_type == 'velocity':
                    methods_order = ['mpn-vel', 'mpn-combined', 'pyramid']
                    colors = ['blue', 'purple', 'red']
                elif cur_type == 'vorticity':
                    methods_order = ['mpn-vel', 'mpn-vor', 'mpn-combined', 'pyramid']
                    colors = ['blue', 'purple', 'red', 'green']

            for j, cur_method in enumerate(methods_order):
                ax.plot(vis_frames, errors_all_methods[cur_type][cur_method], label=f'{cur_method}', c=colors[j])
                # print average result
                cur_avg_loss = np.nanmean(errors_all_methods[cur_type][cur_method])
                print(f'{cur_method} {cur_type} {loss} = {cur_avg_loss}')

            # also plot the "quality" of the particle images
            # ax.plot(vis_frames, np.array(all_pixel_values_sums)/(255*10000.0), label=f'Pixel value count/10000')

            ax.set(xlabel='time step', ylabel=f'{loss}')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.legend()
            if blur_ground_truth:
                vel_loss_curve_path = os.path.join(output_dir, f'all_frames_{cur_type}_losses_blurred_dpi{my_dpi}.png')
            else:
                vel_loss_curve_path = os.path.join(output_dir, f'all_frames_{cur_type}_losses_dpi{my_dpi}.png')
            fig.savefig(vel_loss_curve_path, bbox_inches='tight', dpi=my_dpi*2)
            print(f'\n{mode} {loss} of all frames plot has been saved to {vel_loss_curve_path}\n')


            # energy error plot
            if plot_energy:
                fig, ax = plt.subplots(figsize=(10, 3))
                plt.suptitle(f'Energy {loss} for each frame')

                for j, cur_method in enumerate(methods):
                    ax.plot(vis_frames, energy_errors_all_methods[cur_method], label=f'{cur_method}')
                    # print average result
                    cur_avg_loss = np.nanmean(energy_errors_all_methods[cur_method])
                    print(f'{cur_method} energy {loss} = {cur_avg_loss}')

                # also plot the "quality" of the particle images
                # ax.plot(vis_frames, np.array(all_pixel_values_sums)/(255*10000.0), label=f'Pixel value count/10000')

                ax.set(xlabel='time step', ylabel=f'{loss}')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.legend()
                if blur_ground_truth:
                    energy_loss_curve_path = os.path.join(output_dir, f'all_frames_energy_{mode}_losses_blurred_dpi{my_dpi}.png')
                else:
                    energy_loss_curve_path = os.path.join(output_dir, f'all_frames_energy_{mode}_losses_dpi{my_dpi}.png')
                fig.savefig(energy_loss_curve_path, bbox_inches='tight', dpi=my_dpi*2)
                print(f'\nEnergy {loss} of all frames plot has been saved to {energy_loss_curve_path}\n')


    # simple histogram on velocity and vorticity
    if plot_velocity_histogram:
        for cur_type in result_types:
            # plot ground truth and all the prediction results
            if cur_type == 'velocity':
                # velocity has less when method includes pure vorticity method
                num_less = 0
                for cur_method in methods:
                    if 'vor' in cur_method:
                        num_less += 1

                fig, axes = plt.subplots(nrows=1, ncols=len(methods)+1-num_less, figsize=(5*(len(methods)+1-num_less), 5))
            elif cur_type == 'vorticity':
                fig, axes = plt.subplots(nrows=1, ncols=len(methods)+1, figsize=(5*(len(methods)+1), 5))

            num_bins = 10
            gt_hist, bins = np.histogram(np.array(list(ground_truth[cur_type].values())).flatten(), num_bins)
            axes[0].plot(bins[:num_bins], gt_hist, label='Ground truth')
            i = 0
            for cur_method in methods:
                if 'vor' in cur_method and cur_type == 'velocity':
                    continue
                # plot histogram
                temp = np.array(list(results_all_methods[cur_type][cur_method].values())).flatten()
                cur_hist, _ = np.histogram(temp[~np.isnan(temp).any()], num_bins, density=True)
                axes[i+1].plot(bins[:num_bins], cur_hist, label=cur_method)
                i += 1

            plt.show()


if __name__ == "__main__":
    main()
