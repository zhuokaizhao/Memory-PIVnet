# The script visualizes the test 2 heatmap (-16 to 16)
import os
import plotly
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt

if __name__ == "__main__":

    input_nonaug_dir = 'output/Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/'
    input_aug_dir = 'output/Isotropic_1024_augmented/velocity/amnesia_memory/50000_seeds/time_span_5/epoch25/'
    output_dir = 'figs/Isotropic_1024_augmented/velocity/amnesia_memory/50000_seeds/time_span_5/'

    all_models = ['mpn-5-aug', 'mpn-5-aug']
    all_rotations = ['original', 'rot90', 'rot180', 'rot270']

    time_range = [0, 251]
    non_vis_frames = [80, 81, 153, 154]
    # frames that we are visualizing
    vis_frames = list(range(time_range[0], time_range[1]+1))
    for i in non_vis_frames:
        if i in vis_frames:
            vis_frames.remove(i)

    # loaded velocity fields
    ground_truth = []
    for i in range(len(all_rotations)):
        ground_truth.append([])

    results_all_methods = []
    for i in range(len(all_models)):
        results_all_methods.append([])
        for j in range(len(all_rotations)):
            results_all_methods[i].append([])

    for i, rot in enumerate(all_rotations):
        # load ground truths for all rotations
        ground_truth_path = os.path.join(input_aug_dir, f'{rot}', 'true_vel_field')

        for t in vis_frames:
            cur_path = os.path.join(ground_truth_path, f'true_velocity_{t}.npz')
            ground_truth[i].append(np.load(cur_path)[f'velocity'])

    print(f'Loaded ground truth velocity has shape ({len(ground_truth[0])}, {ground_truth[0][0].shape})')

    # load all methods results
    for i, model in enumerate(all_models):
        if model == 'non-aug':
            result_dirs = os.path.join(input_nonaug_dir)
        else:
            result_dirs = os.path.join(input_aug_dir)

        for j, rot in enumerate(all_rotations):
            cur_dir = os.path.join(result_dirs, f'{rot}', 'blend_vel_field')

            for t in vis_frames:
                cur_path = os.path.join(cur_dir, f'test_velocity_blend_{t}.npz')
                results_all_methods[i][j].append(np.load(cur_path)[f'velocity'])

    for i, cur_model in enumerate(all_models):
        print(f'Loaded {cur_model} velocity has shape ({len(results_all_methods[i][0])}, {results_all_methods[i][0][0].shape})')

    # compute average error
    avg_error = np.zeros((len(all_models), len(all_rotations)))
    for i, model in enumerate(all_models):
        for j, rot in enumerate(all_rotations):
            cur_sum = 0
            for t in range(len(vis_frames)):
                cur_sum += np.sqrt(np.square(ground_truth[j][t] - results_all_methods[i][j][t]).mean(axis=None))
            avg_error[i, j] = cur_sum / len(vis_frames)

    # create an figure with 3*4 subplots
    # first row: non-aug
    # second row: epoch6 aug
    # third row: epoch25 aug
    fig, axes = plt.subplots(nrows=1,
                            ncols=len(all_models),
                            figsize=(3*len(all_models), 3),
                            subplot_kw={'projection': 'polar'})

    # load results for all rotations
    all_degrees = [0, 90, 180, 270, 360]
    for i, model in enumerate(all_models):

        # to close the ring
        plot_data = list(avg_error[i])
        plot_data.append(avg_error[i][0])
        # degrees needs to be converted to rad
        axes[i].plot(np.array(all_degrees)/180*np.pi, plot_data)
        axes[i].set_rlim(0, 3)

        # col titles are the jittering levels
        for ax, col in zip(axes, all_models):
            ax.set_title(col, rotation=0, size='x-large')

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    output_path = os.path.join(output_dir, f'rot_eqv_test.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    # plt.show()

    print(f'Rot eqv test plot has been saved to {output_path}')


