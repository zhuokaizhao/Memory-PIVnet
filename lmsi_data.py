# Data-related functions of project Learning how to measure scientific images
# Note: Please use Julia to generate image data from HDF5
# Author: Zhuokai Zhao

import os
import math
import glob
import h5py
import torch
import timeit
import struct
import random
import imageio
import numpy as np
import multiprocessing
from skimage import io
import tensorflow as tf
import matplotlib.pyplot as plt


# convert from image/labels to pytorch dataset
# mode: train, val or test
# data_dir: data folder directory that has all the images and labels
# output_dir: folder directory that you want to save pt dataset to
# tile_size: individual tile size when spliting the original images
# neighbor_size: tile_size + padding (neighboring information)
# target_dim: velocity label dimension (max 3)
# time_span: controls the padding(repeating) at the start and end of the sequence (for flow estimation purpose)
def generate_pt_dataset(mode, data_dir, output_dir, tile_size, neighbor_size, target_dim, time_span=9):
    # data_dir may contain data initialized with different number of points/objects
    all_sequences = sorted(os.listdir(data_dir), key=lambda x : x.split('_')[1])
    print(f'\n{mode} dataset includes {all_sequences} sequences')

    data_path = os.path.join(output_dir, f'{mode}_data_tilesize_{tile_size[0]}_{tile_size[1]}.h5')
    hf = h5py.File(data_path, 'w')

    # padding(repeating) at the start and end of the sequence
    num_repeat = time_span // 2
    print(f'Repeating {num_repeat} frames at the start and end of the sequence for flow estimation purpose')

    # identify the image type
    image_type = None
    image_size = None
    for cur_sequence in all_sequences:
        cur_sequence_dir = os.path.join(data_dir, cur_sequence)
        all_num_points = sorted(os.listdir(cur_sequence_dir), key=int)

        for cur_num_points in all_num_points:
            print(f'Loading {cur_num_points}-point dataset from sequence {cur_sequence}')
            cur_data_dir = os.path.join(cur_sequence_dir, cur_num_points)

            for filename in os.listdir(cur_data_dir):
                if (os.path.isfile(os.path.join(cur_data_dir, filename))
                    and not filename.endswith('.npy')):
                    # save image type if never been saved
                    if image_type == None:
                        image_type = filename.split('.')[1]
                    # images have to be one type
                    else:
                        if not image_type == filename.split('.')[1]:
                            raise Exception('generate_pt_dataset: Images have to be one type')

            # get all the image and velocity paths as a list
            all_image_paths = glob.glob(os.path.join(cur_data_dir, f'*.{image_type}'))
            all_vel_paths = glob.glob(os.path.join(cur_data_dir, '*vel*.npy'))

            # rank these images, maps and labels based on name index
            all_image_paths.sort(key = lambda x: x.split('/')[-1].split('_')[-1].split('.')[0])
            all_vel_paths.sort(key = lambda x: x.split('/')[-1].split('_')[-1].split('.')[0])

            # check if lengths match
            num_images = len(all_image_paths)
            num_vel_labels = len(all_vel_paths)
            if (num_images != num_vel_labels):
                raise Exception(f'generate_tfrecord: Images size {num_images} does not \
                                    match labels size {num_vel_labels}')

            # load all the images and labels
            # load one image to know the image size
            image_size = io.imread(all_image_paths[0], as_gray=True).shape
            all_images = np.zeros((num_images, image_size[0], image_size[1], 1))
            all_labels = np.zeros((num_vel_labels, image_size[0], image_size[1], target_dim))
            for i in range(num_images):
                all_images[i] = io.imread(all_image_paths[i], as_gray=True).reshape(image_size[0], image_size[1], 1)
                all_labels[i] = np.load(all_vel_paths[i])[:, :, :target_dim]

            # break all images and velocity labels into tiles
            all_image_tiles, all_label_tiles = generate_batch_tiles(all_images, all_labels, tile_size)

            # generate padded neighbor tiles
            all_neighbor_tiles = generate_batch_neighbors(all_images, all_image_tiles, tile_size, neighbor_size)

            # re-order tiles and labels in sequence
            # supposed there are 10 images, each gets split into 16 tiles, there would be 16 tile sequences
            # in short, 10*256*256*1 becomes 16*10*64*64*1, where 4 means four tile-position-based sequence
            num_tiles_per_image = (image_size[0]//tile_size[0]) * (image_size[1]//tile_size[1])
            print(f'{num_tiles_per_image} tiles per image')
            all_neighbor_tiles_reorder = np.zeros((num_tiles_per_image, num_images+2*num_repeat, neighbor_size[0], neighbor_size[1], 1))
            all_label_tiles_reorder = np.zeros((num_tiles_per_image, num_images+2*num_repeat, tile_size[0], tile_size[1], target_dim))
            for i in range(num_tiles_per_image):
                # fill the first num_repeat frames and last num_repeat frames with first and last image
                # first
                all_neighbor_tiles_reorder[i, 0:num_repeat] = np.repeat(all_neighbor_tiles[i:i+1], repeats=num_repeat, axis=0)
                all_label_tiles_reorder[i, 0:num_repeat] = np.repeat(all_label_tiles[i:i+1], repeats=num_repeat, axis=0)
                # end
                # when i = 0 (first tile pos), num_tiles_per_image = 16
                # -(num_tiles_per_image-i) = -16
                if i != num_tiles_per_image-1:
                    all_neighbor_tiles_reorder[i, -num_repeat:] = np.repeat(all_neighbor_tiles[-(num_tiles_per_image-i):-(num_tiles_per_image-i)+1], repeats=num_repeat, axis=0)
                    all_label_tiles_reorder[i, -num_repeat:] = np.repeat(all_label_tiles[-(num_tiles_per_image-i):-(num_tiles_per_image-i)+1], repeats=num_repeat, axis=0)
                else:
                    all_neighbor_tiles_reorder[i, -num_repeat:] = np.repeat(all_neighbor_tiles[-(num_tiles_per_image-i):], repeats=num_repeat, axis=0)
                    all_label_tiles_reorder[i, -num_repeat:] = np.repeat(all_label_tiles[-(num_tiles_per_image-i):], repeats=num_repeat, axis=0)

                # normal frames
                tile_index = np.linspace(i, (num_images-1)*num_tiles_per_image+i, num_images).astype(int)
                all_neighbor_tiles_reorder[i, num_repeat:num_repeat+num_images] = all_neighbor_tiles[tile_index]
                all_label_tiles_reorder[i, num_repeat:num_repeat+num_images] = all_label_tiles[tile_index]


            hf.create_dataset(f'image_sequences_{cur_sequence}_{cur_num_points}', data=all_neighbor_tiles_reorder)
            hf.create_dataset(f'label_sequences_{cur_sequence}_{cur_num_points}', data=all_label_tiles_reorder)

    # close the file
    print(f'{mode} dataset has been saved to {data_path}')
    hf.close()


# function that splits the single image into tiles with input sizes
# input image is expected to have shape 3 (height, width, channel)
# tile_size is the size of output tile images
# center_distance is the distance in pixel between two consecutive image centers
def split_image(image, num_tiles_per_image, tile_size, center_distance):
    height = image.shape[0]
    width = image.shape[1]
    channel = image.shape[2]
    # reshape to be rank 4 for tensorflow functions
    image = np.reshape(image, (1, height, width, channel))

    # change the tile_size and center_distance as well
    tile_size = [1, tile_size[0], tile_size[1], channel]
    center_distance = [1, center_distance[0], center_distance[1], channel]
    # generate tile_size patches with padding
    raw_tiles = tf.image.extract_patches(images=image,
                                         sizes=tile_size,
                                         strides=center_distance,
                                         rates=[1, 1, 1, 1],
                                         padding='VALID')

    # reshape dimensions (ntoe that tile_size is now 1*4)
    tiles = np.zeros((num_tiles_per_image, tile_size[1], tile_size[2], channel))

    for height in range(raw_tiles.shape[1]):
        for width in range(raw_tiles.shape[2]):
            # get the current patch and reshape
            patch = raw_tiles.numpy()[0, height, width]
            patch = patch.reshape(tile_size[1], tile_size[2], channel)
            tiles[height*raw_tiles.shape[2]+width] = patch

    return tiles


# function that splits velocity label's 3D numpy array
def split_vel_label(label, num_tiles_per_image, tile_size, center_distance):
    # initialize output
    all_labels = np.zeros((num_tiles_per_image, tile_size[0], tile_size[1], label.shape[-1]))

    # number of tiles in x and y direction
    num_tiles_x = int(np.floor(label.shape[0] / tile_size[0]))
    num_tiles_y = int(np.floor(label.shape[1] / tile_size[1]))

    # make the split
    # numpy vsplit cuts horizontally, hsplit cuts vertically
    # split data into num_tiles_x rows
    x_splits = np.vsplit(label, num_tiles_x)
    for i in range(num_tiles_x):
        # for each row, cut into piece
        y_splits = np.hsplit(x_splits[i], num_tiles_y)
        for j in range(num_tiles_y):
            all_labels[i*num_tiles_y + j] = y_splits[j]

    return all_labels


# function that put batch_size image and label into small tiles
def generate_batch_tiles(batch_images, batch_labels, tile_size):

    if len(batch_images) != len(batch_labels):
        raise Exception('Non-matched batch image and label sizes')

    # print(f'Generating tiles for batch images')

    # number of images
    num_images = len(batch_images)
    # obtain number of tiles per image
    # assume that all images have the same height and width
    image_height = batch_images.shape[1]
    image_width = batch_images.shape[2]
    channel = batch_images.shape[3]

    tile_height = tile_size[0]
    tile_width = tile_size[1]
    num_tiles_per_image_float = (image_height / tile_height) * (image_width / tile_width)
    num_tiles_per_image = int(num_tiles_per_image_float)
    # we should have integer devision result
    if (num_tiles_per_image_float != num_tiles_per_image):
        raise Exception('Non-integer tile devision result')

    # create tiles for class object
    num_tiles_all_images = int(num_images * num_tiles_per_image)
    image_tiles = np.zeros((num_tiles_all_images, tile_height, tile_width, channel))
    label_tiles = np.zeros((num_tiles_all_images, tile_height, tile_width, batch_labels.shape[-1]))


    for i in range(num_images):
        cur_image = batch_images[i]
        cur_label = batch_labels[i]

        # split current image into 2D tiles, distance = tile size
        cur_image_tiles = split_image(cur_image, num_tiles_per_image, tile_size, tile_size)
        image_tiles[i*num_tiles_per_image:(i+1)*num_tiles_per_image] = cur_image_tiles

        # get the corresponding label
        cur_velocity_tiles = split_vel_label(cur_label, num_tiles_per_image, tile_size, tile_size)
        label_tiles[i*num_tiles_per_image:(i+1)*num_tiles_per_image] = cur_velocity_tiles


    return image_tiles, label_tiles


def generate_batch_neighbors(batch_images, batch_image_tiles, tile_size, neighbor_size):

    # number of images
    num_images = len(batch_images)
    # obtain number of tiles per image
    # assume that all images have the same height and width
    image_height = batch_images.shape[1]
    image_width = batch_images.shape[2]
    channel = batch_images.shape[3]

    tile_height = tile_size[0]
    tile_width = tile_size[1]
    num_tiles_per_image_float = (image_height / tile_height) * (image_width / tile_width)
    num_tiles_per_image = int(num_tiles_per_image_float)
    # we should have integer devision result
    if (num_tiles_per_image_float != num_tiles_per_image):
        raise Exception('Non-integer tile devision result')

    # create neighbor tiles
    num_neighbors_all_images = int(num_images * num_tiles_per_image)
    neighbor_tiles = np.zeros((num_neighbors_all_images, neighbor_size[0], neighbor_size[1], channel))

    # fill out neighbor tiles
    # extra_length is the number of pixels around the center tile
    extra_height = (neighbor_size[0]-tile_size[0])//2
    extra_width = (neighbor_size[1]-tile_size[1])//2

    # the center will be occupied by the tile
    # need to locate the positions
    # no need to minus one since second number is not included
    tile_height_range = ( extra_height, extra_height+tile_height )
    tile_width_range = ( extra_width, extra_width+tile_width )

    # print('Processing neighoring tiles')

    for i in range(num_neighbors_all_images):
        # the center will be occupied by the tile
        neighbor_tiles[i, tile_height_range[0]:tile_height_range[1], tile_width_range[0]:tile_width_range[1], :] = batch_image_tiles[i]

        # fill in the actual neighboring parts
        # corresponding full image number
        image_number = i // num_tiles_per_image
        # tile number
        tile_number = i % num_tiles_per_image
        # row and col number of this tile within the full image
        row = tile_number // (image_width // tile_width)
        col = tile_number % (image_width // tile_width)

        # fill out current neighbor
        for h in range(neighbor_size[0]):
            for w in range(neighbor_size[1]):
                # if pixel is inside the center tile area, continue
                # for 128x128 tile size, range would be (0, 128)
                # so lower bound included, upper bound excluded
                if ((tile_height_range[0] <= h and h < tile_height_range[1]) and
                    (tile_width_range[0] <= w and w < tile_width_range[1])):

                    continue

                # convert distance between current (h, w) to the top left corner of the center tile
                h_diff = h - tile_height_range[0]
                w_diff = w - tile_width_range[0]

                # find the top left corner position in the full image
                top_left_image_h = row * tile_height
                top_left_image_w = col * tile_width

                # convert (h, w) to full image
                h_image = top_left_image_h + h_diff
                w_image = top_left_image_w + w_diff

                # boundary check
                if h_image < 0 or h_image >= image_height or w_image < 0 or w_image >= image_width:
                    pixel_value = 0
                else:
                    pixel_value = batch_images[image_number, h_image, w_image, :]

                # assign pixel value
                neighbor_tiles[i, h, w, :] = pixel_value

    return neighbor_tiles


# functiont hat gets the padded image from all images with index id
def get_padded_tile(image, tile_size, tile_index, target_height=None, target_width=None):

    # generate tiles of current image
    num_tiles_per_image = int(image.shape[0]/tile_size[0] * image.shape[1]/tile_size[1])

    tiles = split_image(image, num_tiles_per_image, tile_size, tile_size)

    chosen_tile = tiles[tile_index]

    # get the surrounding images and pad it if needed
    if target_height != None and target_width != None:
        chosen_tile = tf.image.resize_with_pad(chosen_tile,
                                               target_height,
                                               target_width,
                                               method=tf.image.ResizeMethod.BILINEAR,
                                               antialias=True)

    return chosen_tile




