# The script visualizes the comparison between non-aug and aug model of memory piv net
import os
# import plotly
import numpy as np
from PIL import Image
from tqdm import tqdm
# import plotly.express as px
# import plotly.graph_objects as go
from matplotlib import pyplot as plt

import plot

if __name__ == "__main__":

    plot_color_encoded = False
    plot_avg_error_polar = True
    loss = 'RMSE'
    my_dpi = 100

    time_range = [0, 251]
    non_vis_frames = [80, 81, 153, 154]

    # frames that we are visualizing
    vis_frames = list(range(time_range[0], time_range[1]+1))
    for i in non_vis_frames:
        if i in vis_frames:
            vis_frames.remove(i)

    # all_models = ['mpn-5', 'mpn-5-aug']
    all_models = ['mpn-5']
    all_rotations = ['original', 'rot90', 'rot180', 'rot270']

    # rough input dir
    input_nonaug_dir = 'output/Isotropic_1024/velocity/memory_piv_net/amnesia_memory/50000_seeds/time_span_5/'
    input_aug_dir = 'output/Isotropic_1024_augmented/velocity/amnesia_memory/50000_seeds/time_span_5/epoch25/'
    output_dir = 'figs/Isotropic_1024_augmented/velocity/amnesia_memory/50000_seeds/time_span_5/'

    # initiate velocity fields
    ground_truth = []
    # each rotation angle has a list of ground truths
    for i in range(len(all_rotations)):
        ground_truth.append([])

    results_all_models = []
    # each model has a list results for each rotation angle
    for i in range(len(all_models)):
        results_all_models.append([])
        for j in range(len(all_rotations)):
            results_all_models[i].append([])

    # load ground truth
    for i, rot in enumerate(all_rotations):
        # load ground truths for all rotations
        ground_truth_path = os.path.join(input_nonaug_dir, f'{rot}', 'true_vel_field')

        for t in vis_frames:
            cur_path = os.path.join(ground_truth_path, f'true_velocity_{t}.npz')
            ground_truth[i].append(np.load(cur_path)[f'velocity'])

    ground_truth = np.array(ground_truth)
    print(f'Loaded ground truth velocity has shape {ground_truth.shape}')

    # load all methods results
    for i, model in enumerate(all_models):
        if model == 'mpn-5':
            result_dir = input_nonaug_dir
        else:
            result_dir = input_aug_dir

        for j, rot in enumerate(all_rotations):
            cur_dir = os.path.join(result_dir, f'{rot}', 'blend_vel_field')

            for t in vis_frames:
                cur_path = os.path.join(cur_dir, f'test_velocity_blend_{t}.npz')
                results_all_models[i][j].append(np.load(cur_path)[f'velocity'])

    results_all_models = np.array(results_all_models)

    for i, cur_model in enumerate(all_models):
        print(f'Loaded {cur_model} velocity has shape {results_all_models[i].shape}')

    if plot_color_encoded:
        max_truth = 3
        min_truth = -3
        img_size = 256

        for t in tqdm(range(len(vis_frames))):
            # plot ground truth and all the prediction results
            fig, axes = plt.subplots(nrows=len(all_models)+1, ncols=len(all_rotations), figsize=(5*len(all_rotations), 5*(len(all_models)+1)))
            # plt.suptitle(f'Color-encoded {mode} quiver plot at t = {i}')
            skip = 7

            # each row is for a method, ground truth + methods
            for i in range((len(all_models)+1)):
                # each column is for a rotation angle
                for j, rot in enumerate(all_rotations):
                    # first row is ground truth
                    if i == 0:
                        # visualize ground truth
                        flow_vis, _ = plot.visualize_flow(ground_truth[j, t], max_vel=max_truth)
                        # convert to Image
                        flow_vis_image = Image.fromarray(flow_vis)
                        # display the image
                        axes[i, j].imshow(flow_vis_image)

                        # superimpose quiver plot on color-coded images
                        x = np.linspace(0, img_size-1, img_size)
                        y = np.linspace(0, img_size-1, img_size)
                        y_pos, x_pos = np.meshgrid(x, y)
                        Q = axes[i, j].quiver(y_pos[::skip, ::skip],
                                            x_pos[::skip, ::skip],
                                            ground_truth[j, t][::skip, ::skip, 0]/max_truth,
                                            -ground_truth[j, t][::skip, ::skip, 1]/max_truth,
                                            # scale=4.0,
                                            scale_units='inches')
                        Q._init()
                        assert isinstance(Q.scale, float)
                        axes[i, j].set_title(f'{rot}', fontsize=18)
                        axes[i, 0].set_ylabel('Ground truth', fontsize=18)

                    # otherwise it is each model's
                    else:
                        # visualize ground truth
                        flow_vis, _ = plot.visualize_flow(results_all_models[i-1, j, t], max_vel=max_truth)
                        # convert to Image
                        flow_vis_image = Image.fromarray(flow_vis)
                        # display the image
                        axes[i, j].imshow(flow_vis_image)

                        # superimpose quiver plot on color-coded images
                        x = np.linspace(0, img_size-1, img_size)
                        y = np.linspace(0, img_size-1, img_size)
                        y_pos, x_pos = np.meshgrid(x, y)
                        Q = axes[i, j].quiver(y_pos[::skip, ::skip],
                                            x_pos[::skip, ::skip],
                                            results_all_models[i-1, j, t][::skip, ::skip, 0]/max_truth,
                                            -results_all_models[i-1, j, t][::skip, ::skip, 1]/max_truth,
                                            # scale=4.0,
                                            scale_units='inches')
                        Q._init()
                        assert isinstance(Q.scale, float)
                        axes[i, 0].set_ylabel(f'{cur_model}', fontsize=18)


                        # label error
                        if loss == 'MAE':
                            cur_loss = np.abs(ground_truth[j, t] - results_all_models[i-1, j, t]).mean(axis=None)
                        elif loss == 'MSE':
                            cur_loss = np.square(ground_truth[j, t] - results_all_models[i-1, j, t]).mean(axis=None)
                        elif loss == 'RMSE':
                            cur_loss = np.sqrt(np.square(ground_truth[j, t] - results_all_models[i-1, j, t]).mean(axis=None))

                        axes[i, j].annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='black', fontsize='medium')

            # save the image
            color_encoded_dir = os.path.join(output_dir, f'rot_eqv_velocity_color_encoded_dpi{my_dpi}')

            os.makedirs(color_encoded_dir, exist_ok=True)
            color_encoded_path = os.path.join(color_encoded_dir, f'rot_eqv_velocity_color_encoded_{str(t).zfill(4)}.png')
            plt.savefig(color_encoded_path, bbox_inches='tight', dpi=my_dpi)
            fig.clf()
            plt.close(fig)

        print(f'All color encoded plots have been saved to {output_dir}')

    if plot_avg_error_polar:
        # compute average error
        avg_error = np.zeros((len(all_models), len(all_rotations)))
        for i, model in enumerate(all_models):
            for j, rot in enumerate(all_rotations):
                avg_error[i, j] = np.mean(np.sqrt(np.square(ground_truth[j, :] - results_all_models[i, j, :]).mean(axis=-1)))

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
            plot_data.append(avg_error[i, 0])
            # degrees needs to be converted to rad
            axes.plot(np.array(all_degrees)/180*np.pi, plot_data)
            axes.set_rlim(0, 0.6)

            # for ax, col in zip(axes, all_models):
            #     ax.set_title(col, rotation=0, size='x-large')
            axes.set_title(model, rotation=0, size='x-large')

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        output_path = os.path.join(output_dir, f'rot_eqv_test.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        # plt.show()

        print(f'Rot eqv test plot has been saved to {output_path}')


