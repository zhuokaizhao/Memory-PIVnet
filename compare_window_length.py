# The script compares the performance of models trained with different window lengths
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
import vorticity


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

    # all the window lengths
    window_lengths = [3, 5]
    methods = []
    result_dirs = []

    if mode == 'velocity':
        if data == 'isotropic_1024':
            ground_truth_path = '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/true_vel_field/'
            for length in window_lengths:
                methods.append(f'memory-pir-net-{length}')
                result_dirs.append(f'/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_{length}/blend_vel_field/')

        elif data == 'rotational':
            ground_truth_path = '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Rotational/velocity/memory_piv_net/amnesia_memory/4000_seeds/time_span_5/true_vel_field/'
            for length in window_lengths:
                methods.append(f'memory-pir-net-{length}')
                result_dirs.append(f'/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Rotational/velocity/memory_piv_net/amnesia_memory/4000_seeds/time_span_{length}/blend_vel_field/')

        else:
            raise(Exception(f'Unknown dataset {data}'))

    # start and end time (both inclusive)
    img_size = 256
    time_range = [start_t, end_t]
    # frames that we are visualizing
    vis_frames = list(range(time_range[0], time_range[1]+1))

    # frame 81, 153, 154 have broken ground truth for isotropic_1024
    if data == 'isotropic_1024':
        non_vis_frames = [80, 81, 153, 154]

        for i in non_vis_frames:
            if i in vis_frames:
                vis_frames.remove(i)

    # load ground truth
    ground_truth = {}
    # for sanity check
    all_x = []
    all_y = []
    all_x_rms = []
    all_y_rms = []
    for t in range(time_range[0], time_range[1]+1):
        cur_path = os.path.join(ground_truth_path, f'true_{mode}_{t}.npz')
        ground_truth[str(t)] = np.load(cur_path)[f'{mode}']

        all_x.append(np.mean(np.abs(ground_truth[str(t)][:, :, 0])))
        all_y.append(np.mean(np.abs(ground_truth[str(t)][:, :, 1])))
        all_x_rms.append(np.sqrt(np.sum(ground_truth[str(t)][:, :, 0]**2) / (256 * 256)))
        all_y_rms.append(np.sqrt(np.sum(ground_truth[str(t)][:, :, 1]**2) / (256 * 256)))

    print(f'\nAverage velocity in x is {np.mean(all_x)} pixels/frame')
    print(f'Average velocity in y is {np.mean(all_y)} pixels/frame')
    print(f'Average RMS velocity in x is {np.mean(all_x_rms)} pixels/frame')
    print(f'Average RMS velocity in y is {np.mean(all_y_rms)} pixels/frame\n')

    # load results from all window lengths
    results_all_windows = {}

    for i, length in enumerate(window_lengths):
        results_all_windows[length] = {}

        # load the velocity fields of the specified time range
        for t in range(time_range[0], time_range[1]+1):
            cur_path = os.path.join(result_dirs[i], f'test_{mode}_blend_{t}.npz')
            results_all_windows[length][str(t)] = np.load(cur_path)[f'{mode}']

    for i, length in enumerate(window_lengths):
        print(f'memory-piv-net-{length} {mode} has shape ({len(results_all_windows[length])}, {results_all_windows[length][str(time_range[0])].shape})')


    # max velocity from ground truth is useful for deciding bins
    if mode == 'velocity':
        max_truth = 4
        min_truth = -4
    elif mode == 'vorticity':
        max_truth = 1
        min_truth = -1


    bin_width = 0.05
    bins = np.arange(min_truth, max_truth, bin_width)
    # visualizing the results
    for i in tqdm(vis_frames):
        for k in range(2):
            # plot histogram about the velocities
            gt_hist, bins = np.histogram(ground_truth[str(i)][:, :, k].flatten(), bins, density=False)
            fig, ax = plt.subplots()
            print(gt_hist)
            ax.plot(bins[:-1], gt_hist)
            # plt.show()
            for j in range(len(bins)):
                bin_left = bins[j]
                bin_right = bins[j+1]
                a = np.where(np.logical_and(ground_truth[str(i)][:, :, k]>bin_left, ground_truth[str(i)][:, :, k]<bin_right))
                print(a)
                exit()








if __name__ == "__main__":
    main()
