# main functions of project Learning how to measure scientific images
# Author: Zhuokai Zhao

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# not printing tf warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import math
import glob
import time
import h5py
import copy
import torch
import random
import argparse
import subprocess
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import doc
import models
import plot
import tools
import pair_data

# preferably use the non-display gpu for training
os.environ['CUDA_VISIBLE_DEVICES']='0, 1'
# os.environ['CUDA_VISIBLE_DEVICES']='0'
# preferably use the display gpu for testing
# os.environ['CUDA_VISIBLE_DEVICES']='2'

print('\n\nPython VERSION:', sys.version)
print('PyTorch VERSION:', torch.__version__)
from subprocess import call
# call(["nvcc", "--version"]) does not work
# ! nvcc --version
print('CUDNN VERSION:', torch.backends.cudnn.version())
print('Number CUDA Devices:', torch.cuda.device_count())
print('Devices')
call(['nvidia-smi', '--format=csv', '--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free'])
print('Active CUDA Device: GPU', torch.cuda.current_device())

print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())

# perform some system checks
def check_system():
    if sys.version_info.minor < 4 or sys.version_info.minor > 7:
        raise Exception('Python 3.4 - 3.7 is required')

    if not int(str('').join(torch.__version__.split('.')[0:2])) >= 13:
        raise Exception('At least PyTorch version 1.3.0 is needed')


# Print iterations progress
def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    # if iteration == total:
    #     print()


# helper functions on blendings
# blend the top let part of the input tile
def blend_top_left(i,
                    device,
                    lmsi_model,
                    cur_t_image_block,
                    cur_t_tile_label_pred,
                    image_tile_height,
                    image_tile_width,
                    label_tile_height,
                    label_tile_width,
                    num_tile_column,
                    target_dim):

    print('blending top left')

    # construct data for its y00
    # cur_t_image_block has shape (16, time_span, 128, 128)
    # cur_t_tile_block_00 has shape (1, time_span, 128, 128)
    cur_t_tile_block_00 = torch.zeros(cur_t_image_block[0:1].shape)
    # top left is the previous-row previous(left) tile's center part
    if i-num_tile_column-1 >= 0:
        cur_t_tile_block_00[:, :, :image_tile_height//2, :image_tile_width//2] = cur_t_image_block[i-num_tile_column-1:i-num_tile_column, :, image_tile_width//4:image_tile_width*3//4, image_tile_width//4:image_tile_width*3//4]
    # top right is the previous-row tile's center part
    if i-num_tile_column >= 0:
        cur_t_tile_block_00[:, :, :image_tile_height//2, image_tile_width//2:] = cur_t_image_block[i-num_tile_column:i-num_tile_column+1, :, image_tile_width//4:image_tile_width*3//4, image_tile_width//4:image_tile_width*3//4]
    # bottom left is the previous tile's center part
    if i-1 >= 0:
        cur_t_tile_block_00[:, :, image_tile_height//2:, :image_tile_width//2] = cur_t_image_block[i-1:i, :, image_tile_width//4:image_tile_width*3//4, image_tile_width//4:image_tile_width*3//4]
    # bottom right is the current tile's center part
    cur_t_tile_block_00[:, :, image_tile_height//2:, image_tile_width//2:] = cur_t_image_block[i:i+1, :, image_tile_width//4:image_tile_width*3//4, image_tile_width//4:image_tile_width*3//4]
    # get prediction y00
    prediction_00 = lmsi_model(cur_t_tile_block_00.to(device))
    # put on cpu and permute to channel last
    y00 = prediction_00.cpu().data
    y00 = y00.permute(0, 2, 3, 1).numpy()
    y00 = y00[0, label_tile_height//2:, label_tile_width//2:]

    # construct data for its y01
    # cur_t_image_block has shape (16, time_span, 128, 128)
    # cur_t_tile_block_01 has shape (1, time_span, 128, 128)
    cur_t_tile_block_01 = torch.zeros(cur_t_image_block[0:1].shape)
    # top half is the previous-row tile's horizontal central slice
    if i-num_tile_column >= 0:
        cur_t_tile_block_01[:, :, :image_tile_height//2, :] = cur_t_image_block[i-num_tile_column:i-num_tile_column+1, :, image_tile_width//4:image_tile_width*3//4, :]
    # bottom half is the current tile's horizontal central slice
    cur_t_tile_block_01[:, :, image_tile_height//2:, :] = cur_t_image_block[i:i+1, :, image_tile_width//4:image_tile_width*3//4, :]
    # get prediction y01
    prediction_01 = lmsi_model(cur_t_tile_block_01.to(device))
    # put on cpu and permute to channel last
    y01 = prediction_01.cpu().data
    y01 = y01.permute(0, 2, 3, 1).numpy()
    y01 = y01[0, label_tile_height//2:, :label_tile_width//2]

    # construct data for its y10
    # cur_t_image_block has shape (16, time_span, 128, 128)
    # cur_t_tile_block_10 has shape (1, time_span, 128, 128)
    cur_t_tile_block_10 = torch.zeros(cur_t_image_block[0:1].shape)
    # left half is the previous tile's vertical central slice
    if i-1 >= 0:
        cur_t_tile_block_10[:, :, :, :image_tile_width//2] = cur_t_image_block[i-1:i, :, :, image_tile_width//4:image_tile_width*3//4]
    # right half is the current tile's vertical central slice
    cur_t_tile_block_10[:, :, :, image_tile_width//2:] = cur_t_image_block[i:i+1, :, :, image_tile_width//4:image_tile_width*3//4]
    # get prediction y10
    prediction_10 = lmsi_model(cur_t_tile_block_10.to(device))
    # put on cpu and permute to channel last
    y10 = prediction_10.cpu().data
    y10 = y10.permute(0, 2, 3, 1).numpy()
    y10 = y10[0, :label_tile_height//2, label_tile_width//2:]

    # current prediction is its y11
    y11 = cur_t_tile_label_pred[:label_tile_height//2, :label_tile_width//2]

    # bilinear interpolate
    cur_t_tile_label_pred_blend_00 = np.zeros((label_tile_height//2, label_tile_width//2, target_dim))
    for s in range(label_tile_height//2):
        for r in range(label_tile_width//2):

            s_uni = s / (label_tile_height//2-1)
            r_uni = r / (label_tile_width//2-1)

            cur_t_tile_label_pred_blend_00[s, r] = (1-r_uni)*(1-s_uni)*y00[s, r] + r_uni*(1-s_uni)*y01[s, r] \
                                                    + (1-r_uni)*s_uni*y10[s, r] + r_uni*s_uni*y11[s, r]

    return cur_t_tile_label_pred_blend_00


# blend the top right part of the input tile
def blend_top_right(i,
                    device,
                    lmsi_model,
                    cur_t_image_block,
                    cur_t_tile_label_pred,
                    image_tile_height,
                    image_tile_width,
                    label_tile_height,
                    label_tile_width,
                    num_tile_column,
                    target_dim):

    print('blending top right')

    # construct data for its y00
    # cur_t_image_block has shape (16, time_span, 128, 128)
    # cur_t_tile_block_00 has shape (1, time_span, 128, 128)
    cur_t_tile_block_00 = torch.zeros(cur_t_image_block[0:1].shape)
    # top half is the previous-row tile's horizontal center slice
    # remain black if unavailable
    if i-num_tile_column >= 0:
        cur_t_tile_block_00[:, :, :image_tile_height//2, :] = cur_t_image_block[i-num_tile_column:i-num_tile_column+1, :, image_tile_width//4:image_tile_width*3//4, :]
    # bottom half is the current tile's horizontal center slice
    cur_t_tile_block_00[:, :, image_tile_height//2:, :] = cur_t_image_block[i:i+1, :, image_tile_width//4:image_tile_width*3//4, :]
    # get prediction y00
    prediction_00 = lmsi_model(cur_t_tile_block_00.to(device))
    # put on cpu and permute to channel last
    y00 = prediction_00.cpu().data
    y00 = y00.permute(0, 2, 3, 1).numpy()

    # if i==0:
    #     y00_vis = lmsi_plot.visualize_flow(y00[0])
    #     # convert to Image
    #     y00_image = Image.fromarray(y00_vis)
    #     y00_image_path = os.path.join('/home/zhuokai/Desktop/UChicago/Research/Learning-How-To-Measure-Scientific-Images/Python/figs/MemoryFlowNet3/Blending_test/', f'tile_{i}_00.png')
    #     y00_image.save(y00_image_path)
    #     print(f'y00 has been saved to {y00_image_path}')

    y00 = y00[0, label_tile_height//2:, label_tile_width//2:]

    # construct data for its y01
    # cur_t_image_block has shape (16, time_span, 128, 128)
    # cur_t_tile_block_01 has shape (1, time_span, 128, 128)
    cur_t_tile_block_01 = torch.zeros(cur_t_image_block[0:1].shape)
    # top left is the previous-row tile's center part
    if i-num_tile_column >= 0:
        cur_t_tile_block_01[:, :, :image_tile_height//2, :image_tile_width//2] = cur_t_image_block[i-num_tile_column:i-num_tile_column+1, :, image_tile_width//4:image_tile_width*3//4, image_tile_width//4:image_tile_width*3//4]
    # top right is the previous-row next tile's center part
    if i-num_tile_column+1 >= 0:
        cur_t_tile_block_01[:, :, :image_tile_height//2, image_tile_width//2:] = cur_t_image_block[i-num_tile_column+1:i-num_tile_column+2, :, image_tile_width//4:image_tile_width*3//4, image_tile_width//4:image_tile_width*3//4]
    # bottom left is the current tile's center part
    cur_t_tile_block_01[:, :, image_tile_height//2:, :image_tile_width//2] = cur_t_image_block[i:i+1, :, image_tile_width//4:image_tile_width*3//4, image_tile_width//4:image_tile_width*3//4]
    # bottom right is the next tile's center part
    if i+1 < num_tile_column**2:
        cur_t_tile_block_01[:, :, image_tile_height//2:, image_tile_width//2:] = cur_t_image_block[i+1:i+2, :, image_tile_width//4:image_tile_width*3//4, image_tile_width//4:image_tile_width*3//4]
    # get prediction y01
    prediction_01 = lmsi_model(cur_t_tile_block_01.to(device))
    # put on cpu and permute to channel last
    y01 = prediction_01.cpu().data
    y01 = y01.permute(0, 2, 3, 1).numpy()
    y01 = y01[0, label_tile_height//2:, :label_tile_width//2]

    # current prediction is its y10
    y10 = cur_t_tile_label_pred[:label_tile_height//2, label_tile_width//2:]



    # construct data for its y11
    # cur_t_image_block has shape (16, time_span, 128, 128)
    # cur_t_tile_block_11 has shape (1, time_span, 128, 128)
    cur_t_tile_block_11 = torch.zeros(cur_t_image_block[0:1].shape)
    # left half is the current tile's vertical central slice
    cur_t_tile_block_11[:, :, :, :image_tile_width//2] = cur_t_image_block[i:i+1, :, :, image_tile_width//4:image_tile_width*3//4]
    # right half is the next tile's vertical central slice
    if i+1 < num_tile_column**2:
        cur_t_tile_block_11[:, :, :, image_tile_width//2:] = cur_t_image_block[i+1:i+2, :, :, image_tile_width//4:image_tile_width*3//4]
    # get prediction y11
    prediction_11 = lmsi_model(cur_t_tile_block_11.to(device))
    # put on cpu and permute to channel last
    y11 = prediction_11.cpu().data
    y11 = y11.permute(0, 2, 3, 1).numpy()

    y11 = y11[0, :label_tile_height//2, :label_tile_width//2]

    # bilinear interpolate
    cur_t_tile_label_pred_blend_01 = np.zeros((label_tile_height//2, label_tile_width//2, target_dim))
    for s in range(label_tile_height//2):
        for r in range(label_tile_width//2):

            s_uni = s / (label_tile_height//2-1)
            r_uni = r / (label_tile_width//2-1)

            cur_t_tile_label_pred_blend_01[s, r] = (1-r_uni)*(1-s_uni)*y00[s, r] + r_uni*(1-s_uni)*y01[s, r] \
                                                    + (1-r_uni)*s_uni*y10[s, r] + r_uni*s_uni*y11[s, r]

    return cur_t_tile_label_pred_blend_01


# blend the bottom left part of the input tile
def blend_bottom_left(i,
                        device,
                        lmsi_model,
                        cur_t_image_block,
                        cur_t_tile_label_pred,
                        image_tile_height,
                        image_tile_width,
                        label_tile_height,
                        label_tile_width,
                        num_tile_column,
                        target_dim):

    print('blending bottom left')

    # construct data for its y00
    # cur_t_image_block has shape (16, time_span, 128, 128)
    # cur_t_tile_block_00 has shape (1, time_span, 128, 128)
    cur_t_tile_block_00 = torch.zeros(cur_t_image_block[0:1].shape)
    # left half is the previous(left) tile's vertical central slice
    if i-1 >= 0:
        cur_t_tile_block_00[:, :, :, :image_tile_width//2] = cur_t_image_block[i-1:i, :, :, image_tile_width//4:image_tile_width*3//4]
    # right half is the current tile's vertical central slice
    cur_t_tile_block_00[:, :, :, image_tile_width//2:] = cur_t_image_block[i:i+1, :, :, image_tile_width//4:image_tile_width*3//4]
    # get prediction y00
    prediction_00 = lmsi_model(cur_t_tile_block_00.to(device))
    # put on cpu and permute to channel last
    y00 = prediction_00.cpu().data
    y00 = y00.permute(0, 2, 3, 1).numpy()
    y00 = y00[0, label_tile_height//2:, label_tile_width//2:]

    # current prediction is its y01
    y01 = cur_t_tile_label_pred[label_tile_height//2:, :label_tile_width//2]

    # construct data for its y10
    # cur_t_image_block has shape (16, time_span, 128, 128)
    # cur_t_tile_block_10 has shape (1, time_span, 128, 128)
    cur_t_tile_block_10 = torch.zeros(cur_t_image_block[0:1].shape)
    # top left is the previous(left) tile's central part
    if i-1 >= 0:
        cur_t_tile_block_10[:, :, :image_tile_height//2, :image_tile_width//2] = cur_t_image_block[i-1:i, :, image_tile_width//4:image_tile_width*3//4, image_tile_width//4:image_tile_width*3//4]
    # top right is the current tile's central part
    cur_t_tile_block_10[:, :, :image_tile_height//2, image_tile_width//2:] = cur_t_image_block[i:i+1, :, image_tile_width//4:image_tile_width*3//4, image_tile_width//4:image_tile_width*3//4]
    # bottom left is the next-row previous tile's central part
    if i+num_tile_column-1 < num_tile_column**2:
        cur_t_tile_block_10[:, :, image_tile_height//2:, :image_tile_width//2] = cur_t_image_block[i+num_tile_column-1:i+num_tile_column, :, image_tile_width//4:image_tile_width*3//4, image_tile_width//4:image_tile_width*3//4]
    # bottom right is the next-row tile's central part
    if i+num_tile_column < num_tile_column**2:
        cur_t_tile_block_10[:, :, image_tile_height//2:, image_tile_width//2:] = cur_t_image_block[i+num_tile_column:i+num_tile_column+1, :, image_tile_width//4:image_tile_width*3//4, image_tile_width//4:image_tile_width*3//4]
    # get prediction y10
    prediction_10 = lmsi_model(cur_t_tile_block_10.to(device))
    # put on cpu and permute to channel last
    y10 = prediction_10.cpu().data
    y10 = y10.permute(0, 2, 3, 1).numpy()
    y10 = y10[0, :label_tile_height//2, label_tile_width//2:]

    # construct data for its y11
    # cur_t_image_block has shape (16, time_span, 128, 128)
    # cur_t_tile_block_11 has shape (1, time_span, 128, 128)
    cur_t_tile_block_11 = torch.zeros(cur_t_image_block[0:1].shape)
    # top half is the current tile's horizontal center slice
    cur_t_tile_block_11[:, :, :image_tile_height//2, :] = cur_t_image_block[i:i+1, :, image_tile_width//4:image_tile_width*3//4, :]
    # bottom half is the next-row tile's horizontal center slice
    if i+num_tile_column < num_tile_column**2:
        cur_t_tile_block_11[:, :, image_tile_height//2:, :] = cur_t_image_block[i+num_tile_column:i+num_tile_column+1, :, image_tile_width//4:image_tile_width*3//4, :]
    # get prediction y11
    prediction_11 = lmsi_model(cur_t_tile_block_11.to(device))
    # put on cpu and permute to channel last
    y11 = prediction_11.cpu().data
    y11 = y11.permute(0, 2, 3, 1).numpy()
    y11 = y11[0, :label_tile_height//2, :label_tile_width//2]

    # bilinear interpolate
    cur_t_tile_label_pred_blend_10 = np.zeros((label_tile_height//2, label_tile_width//2, target_dim))
    for s in range(label_tile_height//2):
        for r in range(label_tile_width//2):

            s_uni = s / (label_tile_height//2-1)
            r_uni = r / (label_tile_width//2-1)

            cur_t_tile_label_pred_blend_10[s, r] = (1-r_uni)*(1-s_uni)*y00[s, r] + r_uni*(1-s_uni)*y01[s, r] \
                                                    + (1-r_uni)*s_uni*y10[s, r] + r_uni*s_uni*y11[s, r]

    return cur_t_tile_label_pred_blend_10


# blend the bottom right part of the input tile
def blend_bottom_right(i,
                        device,
                        lmsi_model,
                        cur_t_image_block,
                        cur_t_tile_label_pred,
                        image_tile_height,
                        image_tile_width,
                        label_tile_height,
                        label_tile_width,
                        num_tile_column,
                        target_dim):

    print('blending bottom right')

    # current prediction is its y00
    y00 = cur_t_tile_label_pred[label_tile_height//2:, label_tile_width//2:]

    # construct data for its y01
    # cur_t_image_block has shape (16, time_span, 128, 128)
    # cur_t_tile_block_01 has shape (1, time_span, 128, 128)
    cur_t_tile_block_01 = torch.zeros(cur_t_image_block[0:1].shape)
    # left half is the current tile's vertical center slice
    cur_t_tile_block_01[:, :, :, :image_tile_width//2] = cur_t_image_block[i:i+1, :, :, image_tile_height//4:image_tile_height*3//4]
    # right half is the next tile's vertical center slice
    if i+1 < num_tile_column**2:
        cur_t_tile_block_01[:, :, :, image_tile_width//2:] = cur_t_image_block[i+1:i+2, :, :, image_tile_height//4:image_tile_height*3//4]

    # get prediction y01
    prediction_01 = lmsi_model(cur_t_tile_block_01.to(device))
    # put on cpu and permute to channel last
    y01 = prediction_01.cpu().data
    y01 = y01.permute(0, 2, 3, 1).numpy()

    y01 = y01[0, label_tile_height//2:, :label_tile_width//2]

    # construct data for its y10
    # cur_t_image_block has shape (16, time_span, 128, 128)
    # cur_t_tile_block_10 has shape (1, time_span, 128, 128)
    cur_t_tile_block_10 = torch.zeros(cur_t_image_block[0:1].shape)
    # top half is the current tile's horizontal center slice
    cur_t_tile_block_10[:, :, :image_tile_height//2, :] = cur_t_image_block[i:i+1, :, image_tile_height//4:image_tile_height*3//4, :]
    # bottom half is the next-row tile's horizontal center slice
    if i+num_tile_column < num_tile_column**2:
        cur_t_tile_block_10[:, :, image_tile_height//2:, :] = cur_t_image_block[i+num_tile_column:i+num_tile_column+1, :, image_tile_height//4:image_tile_height*3//4, :]

    # get prediction y10
    prediction_10 = lmsi_model(cur_t_tile_block_10.to(device))
    # put on cpu and permute to channel last
    y10 = prediction_10.cpu().data
    y10 = y10.permute(0, 2, 3, 1).numpy()

    y10 = y10[0, :label_tile_height//2, label_tile_width//2:]

    # construct data for its y11
    # cur_t_image_block has shape (16, time_span, 128, 128)
    # cur_t_tile_block_11 has shape (1, time_span, 128, 128)
    cur_t_tile_block_11 = torch.zeros(cur_t_image_block[0:1].shape)
    # top left is the current tile's central part
    cur_t_tile_block_11[:, :, :image_tile_height//2, :image_tile_width//2] = cur_t_image_block[i:i+1, :, image_tile_height//4:image_tile_height*3//4, image_tile_height//4:image_tile_height*3//4]
    # top right is the next tile's central part
    if i+1 < num_tile_column**2:
        cur_t_tile_block_11[:, :, :image_tile_height//2, image_tile_width//2:] = cur_t_image_block[i+1:i+2, :, image_tile_height//4:image_tile_height*3//4, image_tile_height//4:image_tile_height*3//4]
    # bottom left is the next-row tile's central part
    if i+num_tile_column < num_tile_column**2:
        cur_t_tile_block_11[:, :, image_tile_height//2:, :image_tile_width//2] = cur_t_image_block[i+num_tile_column:i+num_tile_column+1, :, image_tile_height//4:image_tile_height*3//4, image_tile_height//4:image_tile_height*3//4]
    # bottom right is the next-row next tile's central part
    if i+num_tile_column+1 < num_tile_column**2:
        cur_t_tile_block_11[:, :, image_tile_height//2:, image_tile_width//2:] = cur_t_image_block[i+num_tile_column+1:i+num_tile_column+2, :, image_tile_height//4:image_tile_height*3//4, image_tile_height//4:image_tile_height*3//4]

    # get prediction y11
    prediction_11 = lmsi_model(cur_t_tile_block_11.to(device))
    # put on cpu and permute to channel last
    y11 = prediction_11.cpu().data
    y11 = y11.permute(0, 2, 3, 1).numpy()
    y11 = y11[0, :label_tile_height//2, :label_tile_width//2]

    # bilinear interpolate
    cur_t_tile_label_pred_blend_11 = np.zeros((label_tile_height//2, label_tile_width//2, target_dim))
    for s in range(label_tile_height//2):
        for r in range(label_tile_width//2):

            s_uni = s / (label_tile_height//2-1)
            r_uni = r / (label_tile_width//2-1)

            cur_t_tile_label_pred_blend_11[s, r] = (1-r_uni)*(1-s_uni)*y00[s, r] + r_uni*(1-s_uni)*y01[s, r] \
                                                    + (1-r_uni)*s_uni*y10[s, r] + r_uni*s_uni*y11[s, r]

    return cur_t_tile_label_pred_blend_11


# helper function that takes the new-format data and then run inference
# final size is the full image size
# both start_t and end_t are inclusive
def run_test(network_model,
            all_test_image_sequences,
            all_test_label_sequences,
            model_dir,
            figs_dir,
            loss,
            start_t,
            end_t,
            num_channels,
            time_span,
            target_dim,
            num_tiles_per_image,
            final_size,
            device,
            blend=False,
            draw_normal=True,
            draw_glyph=False):

    with torch.no_grad():
        # check if input parameters are valid
        if start_t < 0:
            raise Exception('Invalid start_t')
        elif end_t > all_test_image_sequences.shape[2]-time_span:
            raise Exception(f'Invalid end_t > total number of time frames {all_test_image_sequences.shape[2]-time_span}')

        # define the loss
        if loss == 'MSE' or loss == 'RMSE':
            loss_module = torch.nn.MSELoss()
        elif loss == 'MAE':
            loss_module = torch.nn.L1Loss()

        # memory network parameters
        kwargs = {
                    'num_channels':               num_channels,
                    'time_span':                  time_span,
                    'target_dim':                 target_dim,
                    'mode':                       'test',
                    'N':                          4
                }

        # model, optimizer and loss
        lmsi_model = None
        if network_model == 'memory-piv-net':
            lmsi_model = models.Memory_PIVnet(**kwargs)
        elif network_model == 'memory-piv-net-no-neighbor':
            lmsi_model = models.Memory_PIVnet_No_Neighbor(**kwargs)
            # no blend is available for non-padding data
            blend = False

        lmsi_model.eval()
        lmsi_model.to(device)

        # load trained model
        trained_model = torch.load(model_dir)
        lmsi_model.load_state_dict(trained_model['state_dict'])

        # assume testing data has shape (1, 16, 260, 1, 128, 128)
        # which means 1 sequence, which is splitted into 16 tile-sequences
        # since time span is 9, the total time stamp is 260-time_span+1 = 252
        # each neighbor tile has size 1*128*128 (channel first)
        # take testing sequence order one by one
        test_start_time = time.time()
        sequence_order = list(range(len(all_test_image_sequences)))
        print(f'\ntesting sequence(s) is/are {sequence_order}')

        # test for each sequence
        for sequence_index in sequence_order:
            print(f'testing with sequence {sequence_index}')
            # testing dataset
            # cur_image_sequence has shape (16, 260, 128, 128)
            cur_image_sequence = all_test_image_sequences[sequence_index, :, :, 0, :, :]
            print(f'cur_image_sequence.shape {cur_image_sequence.shape}')
            # cur_label_sequence has shape (16, 260, target_dim, 64, 64)
            cur_label_sequence = all_test_label_sequences[sequence_index]
            print(f'cur_label_sequence.shape {cur_label_sequence.shape}')

            # some values that are needed in the future
            # image tile height and width
            image_tile_height = cur_image_sequence.shape[2]
            image_tile_width = cur_image_sequence.shape[3]
            # label tile height and width
            label_tile_height = cur_label_sequence.shape[3]
            label_tile_width = cur_label_sequence.shape[4]
            num_tile_row = final_size//label_tile_height
            num_tile_column = final_size//label_tile_width

            # for minimal gpu ram purpose, take one time stamp at a time
            num_time_frames = len(cur_image_sequence)
            # we have (end_t-start_t+1) time frames in total
            all_losses = []
            all_losses_blend = []

            for t in range(time_span//2+start_t, time_span//2+end_t+1):
                print(f'\nInferencing t={t-time_span//2}')
                # cur_t_image_block has shape (16, time_span, 128, 128)
                cur_t_image_block = cur_image_sequence[:, t-time_span//2:t+time_span//2+1]
                print(f'cur_t_image_block.shape {cur_t_image_block.shape}')
                # cur_t_label_tile has shape (16, target_dim, 64, 64)
                cur_t_label_tile = cur_label_sequence[:, t]
                print(f'cur_t_label_tile.shape {cur_t_label_tile.shape}')

                # stitched result for the current t
                cur_t_stitched_image = np.zeros((final_size,
                                                final_size,
                                                1))
                cur_t_stitched_label_true = np.zeros((final_size,
                                                        final_size,
                                                        target_dim))
                cur_t_stitched_label_pred = np.zeros((final_size,
                                                        final_size,
                                                        target_dim))

                if blend:
                    cur_t_stitched_label_pred_blend = np.zeros((final_size,
                                                                final_size,
                                                                target_dim))

                # loop through all the tiles
                for i in range(num_tiles_per_image):
                    print(f'\nInferencing tile {i}')
                    # cur_t_tile_block has shape (1, time_span, 128, 128)
                    cur_t_tile_block = cur_t_image_block[i:i+1].to(device)
                    # cur_t_tile_label has shape (64, 64, target_dim)
                    cur_t_tile_label_true = cur_t_label_tile[i].permute(1, 2, 0).numpy()

                    # run inference
                    prediction = lmsi_model(cur_t_tile_block)

                    # put on cpu and permute to channel last
                    cur_t_tile_label_pred = prediction.cpu().data
                    cur_t_tile_label_pred = cur_t_tile_label_pred.permute(0, 2, 3, 1).numpy()
                    cur_t_tile_label_pred = cur_t_tile_label_pred[0]

                    # save the un-blend result
                    h = i // num_tile_column
                    w = i % num_tile_column

                    # take the center part if padded data
                    if network_model == 'memory-piv-net':
                        cur_t_stitched_image[h*label_tile_height:(h+1)*label_tile_height,
                                            w*label_tile_width:(w+1)*label_tile_width,
                                            :] \
                            = cur_t_tile_block.permute(0, 2, 3, 1)[0,
                                                                image_tile_height//4:image_tile_height//4*3,
                                                                image_tile_width//4:image_tile_width//4*3,
                                                                time_span//2:time_span//2+1].cpu().numpy()

                        cur_t_stitched_label_true[h*label_tile_height:(h+1)*label_tile_height,
                                                w*label_tile_width:(w+1)*label_tile_width,
                                                :] \
                            = cur_t_tile_label_true

                        cur_t_stitched_label_pred[h*label_tile_height:(h+1)*label_tile_height,
                                                w*label_tile_width:(w+1)*label_tile_width,
                                                :] \
                            = cur_t_tile_label_pred

                        # if blending is needed
                        if blend:
                            cur_t_tile_label_pred_blend = copy.deepcopy(cur_t_tile_label_pred)
                            # blend all four parts
                            # blend top left
                            cur_t_tile_label_pred_blend_00 = blend_top_left(i,
                                                                            device,
                                                                            lmsi_model,
                                                                            cur_t_image_block,
                                                                            cur_t_tile_label_pred,
                                                                            image_tile_height,
                                                                            image_tile_width,
                                                                            label_tile_height,
                                                                            label_tile_width,
                                                                            num_tile_column,
                                                                            target_dim)

                            # blend top right
                            cur_t_tile_label_pred_blend_01 = blend_top_right(i,
                                                                            device,
                                                                            lmsi_model,
                                                                            cur_t_image_block,
                                                                            cur_t_tile_label_pred,
                                                                            image_tile_height,
                                                                            image_tile_width,
                                                                            label_tile_height,
                                                                            label_tile_width,
                                                                            num_tile_column,
                                                                            target_dim)

                            # blend bottom left
                            cur_t_tile_label_pred_blend_10 = blend_bottom_left(i,
                                                                                device,
                                                                                lmsi_model,
                                                                                cur_t_image_block,
                                                                                cur_t_tile_label_pred,
                                                                                image_tile_height,
                                                                                image_tile_width,
                                                                                label_tile_height,
                                                                                label_tile_width,
                                                                                num_tile_column,
                                                                                target_dim)

                            # blend bottom right
                            cur_t_tile_label_pred_blend_11 = blend_bottom_right(i,
                                                                                device,
                                                                                lmsi_model,
                                                                                cur_t_image_block,
                                                                                cur_t_tile_label_pred,
                                                                                image_tile_height,
                                                                                image_tile_width,
                                                                                label_tile_height,
                                                                                label_tile_width,
                                                                                num_tile_column,
                                                                                target_dim)

                            # replace with the blended result
                            cur_t_tile_label_pred_blend[:label_tile_height//2, :label_tile_width//2, :] = cur_t_tile_label_pred_blend_00
                            cur_t_tile_label_pred_blend[:label_tile_height//2, label_tile_width//2:, :] = cur_t_tile_label_pred_blend_01
                            cur_t_tile_label_pred_blend[label_tile_height//2:, :label_tile_width//2, :] = cur_t_tile_label_pred_blend_10
                            cur_t_tile_label_pred_blend[label_tile_height//2:, label_tile_width//2:, :] = cur_t_tile_label_pred_blend_11

                            # stitch the blended tiles
                            cur_t_stitched_label_pred_blend[h*label_tile_height:(h+1)*label_tile_height,
                                                            w*label_tile_width:(w+1)*label_tile_width,
                                                            :] \
                                = cur_t_tile_label_pred_blend
                    elif network_model == 'memory-piv-net-no-neighbor':
                        cur_t_stitched_image[h*label_tile_height:(h+1)*label_tile_height,
                                            w*label_tile_width:(w+1)*label_tile_width,
                                            :] \
                            = cur_t_tile_block.permute(0, 2, 3, 1)[0,:, :, time_span//2:time_span//2+1].cpu().numpy()

                        cur_t_stitched_label_true[h*label_tile_height:(h+1)*label_tile_height,
                                                w*label_tile_width:(w+1)*label_tile_width,
                                                :] \
                            = cur_t_tile_label_true

                        cur_t_stitched_label_pred[h*label_tile_height:(h+1)*label_tile_height,
                                                w*label_tile_width:(w+1)*label_tile_width,
                                                :] \
                            = cur_t_tile_label_pred

                # scale the result from [0, 256] to [0, 1]
                cur_t_stitched_label_pred = cur_t_stitched_label_pred / final_size
                cur_t_stitched_label_true = cur_t_stitched_label_true / final_size
                if blend:
                    cur_t_stitched_label_pred_blend = cur_t_stitched_label_pred_blend / final_size

                # compute loss
                if loss == 'MAE' or loss == 'MSE' or loss == 'RMSE':
                    loss_unblend = loss_module(torch.from_numpy(cur_t_stitched_label_pred), torch.from_numpy(cur_t_stitched_label_true))
                    if loss == 'RMSE':
                        loss_unblend = torch.sqrt(loss_unblend)
                elif loss == 'AEE':
                    sum_endpoint_error = 0
                    for i in range(final_size):
                        for j in range(final_size):
                            cur_pred = cur_t_stitched_label_pred[i, j]
                            cur_true = cur_t_stitched_label_true[i, j]
                            cur_endpoint_error = np.linalg.norm(cur_pred-cur_true)
                            sum_endpoint_error += cur_endpoint_error

                    loss_unblend = sum_endpoint_error / (final_size*final_size)
                # customized metric that converts into polar coordinates and compare
                elif loss == 'polar':
                    # convert both truth and predictions to polar coordinate
                    cur_t_stitched_label_true_polar = tools.cart2pol(cur_t_stitched_label_true)
                    cur_t_stitched_label_pred_polar = tools.cart2pol(cur_t_stitched_label_pred)
                    # absolute magnitude difference and angle difference
                    r_diff_mean = np.abs(cur_t_stitched_label_true_polar[:, :, 0]-cur_t_stitched_label_pred_polar[:, :, 0]).mean()
                    theta_diff = np.abs(cur_t_stitched_label_true_polar[:, :, 1]-cur_t_stitched_label_pred_polar[:, :, 1])
                    # wrap around for angles larger than pi
                    theta_diff[theta_diff>2*np.pi] = 2*np.pi - theta_diff[theta_diff>2*np.pi]
                    # compute the mean of angle difference
                    theta_diff_mean = theta_diff.mean()
                    # take the sum as single scalar loss
                    loss_unblend = r_diff_mean + theta_diff_mean


                all_losses.append(loss_unblend)
                print(f'\nInference {loss} of unblended image t={t-time_span//2} is {loss_unblend}')

                # absolute error for plotting magnitude
                pred_error = np.sqrt((cur_t_stitched_label_pred[:,:,0]-cur_t_stitched_label_true[:,:,0])**2 \
                                        + (cur_t_stitched_label_pred[:,:,1]-cur_t_stitched_label_true[:,:,1])**2)

                if blend:
                    if loss == 'MAE' or loss == 'MSE' or loss == 'RMSE':
                        loss_blend = loss_module(torch.from_numpy(cur_t_stitched_label_pred_blend), torch.from_numpy(cur_t_stitched_label_true))
                        if loss == 'RMSE':
                            loss_blend = torch.sqrt(loss_blend)
                    elif loss == 'AEE':
                        sum_endpoint_error_blend = 0
                        for i in range(final_size):
                            for j in range(final_size):
                                cur_pred_blend = cur_t_stitched_label_pred_blend[i, j]
                                cur_true = cur_t_stitched_label_true[i, j]
                                cur_endpoint_error_blend = np.linalg.norm(cur_pred_blend-cur_true)
                                sum_endpoint_error_blend += cur_endpoint_error_blend

                        loss_blend = sum_endpoint_error_blend / (final_size*final_size)
                    # customized metric that converts into polar coordinates and compare
                    elif loss == 'polar':
                        # convert both truth and predictions to polar coordinate
                        cur_t_stitched_label_true_polar = tools.cart2pol(cur_t_stitched_label_true)
                        cur_t_stitched_label_pred_blend_polar = tools.cart2pol(cur_t_stitched_label_pred_blend)
                        # absolute magnitude difference and angle difference
                        r_diff_mean_blend = np.abs(cur_t_stitched_label_true_polar[:, :, 0]-cur_t_stitched_label_pred_blend_polar[:, :, 0]).mean()
                        theta_diff_blend = np.abs(cur_t_stitched_label_true_polar[:, :, 1]-cur_t_stitched_label_pred_blend_polar[:, :, 1])
                        # wrap around for angles larger than pi
                        theta_diff_blend[theta_diff_blend>2*np.pi] = 2*np.pi - theta_diff_blend[theta_diff_blend>2*np.pi]
                        # compute the mean of angle difference
                        theta_diff_mean_blend = theta_diff_blend.mean()
                        # take the sum as single scalar loss
                        loss_blend = r_diff_mean_blend + theta_diff_mean_blend

                        all_losses_blend.append(loss_blend)
                        print(f'\nInference {loss} of blended image t={t-time_span//2} is {loss_blend}')
                        # error for plotting magnitude
                        pred_blend_error = np.sqrt((cur_t_stitched_label_pred_blend[:,:,0]-cur_t_stitched_label_true[:,:,0])**2 \
                                                    + (cur_t_stitched_label_pred_blend[:,:,1]-cur_t_stitched_label_true[:,:,1])**2)

                # save the input image, ground truth, prediction, and difference
                if draw_normal:
                    cur_t_test_image = cur_t_stitched_image[:, :, 0].astype(np.uint8).reshape((final_size, final_size))
                    # visualize the true velocity field
                    cur_t_flow_true, max_vel = plot.visualize_flow(cur_t_stitched_label_true)
                    print(f'Label max vel magnitude is {max_vel}')
                    # visualize the pred velocity field with truth's saturation range
                    cur_t_flow_pred, _ = plot.visualize_flow(cur_t_stitched_label_pred, max_vel=max_vel)
                    # visualize the error magnitude
                    plt.figure()
                    plt.imshow(pred_error, cmap='PuBuGn', interpolation='nearest', vmin=0.0,  vmax=1.0)
                    error_path = os.path.join(figs_dir, f'Memory-PIVnet_{t-time_span//2}_pred_unblend_error.svg')
                    plt.axis('off')
                    cbar = plt.colorbar()
                    cbar.set_label('Endpoint error')
                    plt.savefig(error_path, bbox_inches='tight', dpi=1200)
                    print(f'error magnitude plot has been saved to {error_path}')

                    if blend:
                        cur_t_flow_pred_blend, _ = plot.visualize_flow(cur_t_stitched_label_pred_blend, max_vel=max_vel)
                        # error magnitude plot
                        plt.figure()
                        plt.imshow(pred_blend_error, cmap='PuBuGn', interpolation='nearest', vmin=0.0,  vmax=1.0)
                        error_blend_path = os.path.join(figs_dir, f'Memory-PIVnet_{t-time_span//2}_pred_blend_error.svg')
                        plt.axis('off')
                        cbar = plt.colorbar()
                        # cbar.set_label('Endpoint error')
                        plt.savefig(error_blend_path, bbox_inches='tight', dpi=1200)
                        print(f'blended error magnitude plot has been saved to {error_blend_path}')

                    # convert to Image
                    cur_t_test_image = Image.fromarray(cur_t_test_image)
                    cur_t_flow_true = Image.fromarray(cur_t_flow_true)
                    cur_t_flow_pred = Image.fromarray(cur_t_flow_pred)

                    # superimpose quiver plot on color-coded images
                    # ground truth
                    x = np.linspace(0, final_size-1, final_size)
                    y = np.linspace(0, final_size-1, final_size)
                    y_pos, x_pos = np.meshgrid(x, y)
                    skip = 8
                    plt.figure()
                    plt.imshow(cur_t_flow_true)
                    Q = plt.quiver(y_pos[::skip, ::skip],
                                    x_pos[::skip, ::skip],
                                    cur_t_stitched_label_true[::skip, ::skip, 0]/max_vel,
                                    -cur_t_stitched_label_true[::skip, ::skip, 1]/max_vel,
                                    scale=4.0,
                                    scale_units='inches')
                    Q._init()
                    assert isinstance(Q.scale, float)
                    plt.axis('off')
                    true_quiver_path = os.path.join(figs_dir, f'true_{t-time_span//2}.svg')
                    plt.savefig(true_quiver_path, bbox_inches='tight', dpi=1200)
                    print(f'ground truth quiver plot has been saved to {true_quiver_path}')

                    # unblended results
                    plt.figure()
                    plt.imshow(cur_t_flow_pred)
                    plt.quiver(y_pos[::skip, ::skip],
                                x_pos[::skip, ::skip],
                                cur_t_stitched_label_pred[::skip, ::skip, 0]/max_vel,
                                -cur_t_stitched_label_pred[::skip, ::skip, 1]/max_vel,
                                scale=4.0,
                                scale_units='inches')
                    plt.axis('off')
                    # annotate error
                    # annotate error
                    if loss == 'polar':
                        plt.annotate(f'Magnitude MAE: ' + '{:.3f}'.format(r_diff_mean), (5, 10), color='white', fontsize='medium')
                        plt.annotate(f'Angle MAE: ' + '{:.3f}'.format(theta_diff_mean), (5, 20), color='white', fontsize='medium')
                    else:
                        plt.annotate(f'{loss}: ' + '{:.3f}'.format(loss_unblend), (5, 10), color='white', fontsize='large')
                    unblend_quiver_path = os.path.join(figs_dir, f'Memory-PIVnet_{t-time_span//2}_pred_unblend.svg')
                    plt.savefig(unblend_quiver_path, bbox_inches='tight', dpi=1200)
                    print(f'unblend quiver plot has been saved to {unblend_quiver_path}')

                    if blend:
                        cur_t_flow_pred_blend = Image.fromarray(cur_t_flow_pred_blend)
                        plt.figure()
                        plt.imshow(cur_t_flow_pred_blend)
                        plt.quiver(y_pos[::skip, ::skip],
                                    x_pos[::skip, ::skip],
                                    cur_t_stitched_label_pred_blend[::skip, ::skip, 0]/max_vel,
                                    -cur_t_stitched_label_pred_blend[::skip, ::skip, 1]/max_vel,
                                    scale=4.0,
                                    scale_units='inches')
                        plt.axis('off')
                        # annotate error
                        if loss == 'polar':
                            plt.annotate(f'Magnitude MAE: ' + '{:.3f}'.format(r_diff_mean_blend), (5, 10), color='white', fontsize='medium')
                            plt.annotate(f'Angle MAE: ' + '{:.3f}'.format(theta_diff_mean_blend), (5, 20), color='white', fontsize='medium')
                        else:
                            plt.annotate(f'{loss}: ' + '{:.3f}'.format(loss_blend), (5, 10), color='white', fontsize='large')
                        blend_quiver_path = os.path.join(figs_dir, f'Memory-PIVnet_{t-time_span//2}_pred.svg')
                        plt.savefig(blend_quiver_path, bbox_inches='tight', dpi=1200)
                        print(f'blended quiver plot has been saved to {blend_quiver_path}')

                    # finally save the testing image
                    test_image_path = os.path.join(figs_dir, f'test_{t-time_span//2}.png')
                    cur_t_test_image.save(test_image_path)
                    print(f'Test image has been saved to {test_image_path}')


            min_loss = np.min(all_losses)
            min_loss_index = np.where(all_losses == min_loss)
            avg_loss = np.mean(all_losses)
            print(f'Average unblended {loss} is {avg_loss}')
            print(f'Min unblended {loss} is {min_loss} at t={min_loss_index}\n')
            if blend:
                min_loss_blend = np.min(all_losses_blend)
                min_loss_index_blend = np.where(all_losses_blend == min_loss_blend)
                avg_loss_blend = np.mean(all_losses_blend)
                print(f'Average blended {loss} is {avg_loss_blend}')
                print(f'Min blended {loss} is {min_loss_blend} at t={min_loss_index_blend}')



def main():

	# input arguments
    parser = argparse.ArgumentParser(description=doc.description)
    # mode (data, train, or test mode)
    parser.add_argument('--mode', required=True, action='store', nargs=1, dest='mode', help=doc.mode)
    # network method (unet, 3dunet, etc)
    parser.add_argument('-n', '--network-model', action='store', nargs=1, dest='network_model')
    # input dataset ditectory for various non train/test related modes
    parser.add_argument('-i', '--input-dir', action='store', nargs=1, dest='input_dir', help=doc.data_dir)
    # input training dataset director
    parser.add_argument('--train-dir', action='store', nargs=1, dest='train_dir')
    # input validation dataset ditectory
    parser.add_argument('--val-dir', action='store', nargs=1, dest='val_dir')
    # input testing dataset ditectory
    parser.add_argument('--test-dir', action='store', nargs=1, dest='test_dir')
    # dataset property (ours or other existing image-pair datasets)
    parser.add_argument('-d', '--data-type', action='store', nargs=1, dest='data_type')
    # epoch size
    parser.add_argument('-e', '--num-epoch', action='store', nargs=1, dest='num_epoch')
    # batch size
    parser.add_argument('-b', '--batch-size', action='store', nargs=1, dest='batch_size')
    # image tile size
    parser.add_argument('-s', '--tile-size', action='store', nargs=1, dest='tile_size')
    # 3D patch time span
    parser.add_argument('-t', '--time-span', action='store', nargs=1, dest='time_span')
    # loss function
    parser.add_argument('-l', '--loss', action='store', nargs=1, dest='loss')
    # checkpoint path for continuing training
    parser.add_argument('-c', '--checkpoint-path', action='store', nargs=1, dest='checkpoint_path')
    # input or output model directory
    parser.add_argument('-m', '--model-dir', action='store', nargs=1, dest='model_dir', help=doc.model_dir)
    # output directory (tfrecord in 'data' mode, figure in 'training' mode)
    parser.add_argument('-o', '--output-dir', action='store', nargs=1, dest='output_dir', help=doc.figs_dir)
    # verbosity
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False)
    args = parser.parse_args()

    # check the system and directory
    check_system()
    mode = args.mode[0]
    verbose = args.verbose

    # training with customized manager (newest)
    if mode == 'train':

        if torch.cuda.device_count() > 1:
            print('\n', torch.cuda.device_count(), 'GPUs available')
            device = torch.device('cuda')
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if verbose:
            print(f'\nmode: {mode}')

        network_model = args.network_model[0]
        # data_type can be image-pair or multi-frame (default)
        if args.data_type != None:
            data_type = args.data_type[0]
        else:
            data_type = 'multi-frame'

        train_dir = args.train_dir[0]
        val_dir = args.val_dir[0]

        # checkpoint path to load the model from
        if args.checkpoint_path != None:
            checkpoint_path = args.checkpoint_path[0]
        else:
            checkpoint_path = None
        # directory to save the model to
        model_dir = args.model_dir[0]
        # loss graph directory
        figs_dir = args.output_dir[0]
        # train-related parameters
        num_epoch = int(args.num_epoch[0])
        batch_size = int(args.batch_size[0])
        target_dim = 2
        if args.time_span != None:
            time_span = int(args.time_span[0])
        else:
            time_span = None
        loss = args.loss[0]

        # make sure the model_dir is valid
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
            print(f"model_dir {model_dir} did not exist, but has been created")

        # load the data
        print(f'\nLoading {data_type} datasets')
        # one-sided is the half side of multi-frame
        if data_type == 'multi-frame' or data_type == 'one-sided':
            # load training dataset
            train_dataset = h5py.File(train_dir, 'r')
            # list the number of sequences in this dataset
            num_train_sequences = len(list(train_dataset.keys())) // 2

            all_train_image_sequences = []
            all_train_label_sequences = []
            for i in range(len(list(train_dataset.keys()))):
                if list(train_dataset.keys())[i].startswith('image'):
                    all_train_image_sequences.append(train_dataset.get(list(train_dataset.keys())[i]))
                elif list(train_dataset.keys())[i].startswith('label'):
                    all_train_label_sequences.append(train_dataset.get(list(train_dataset.keys())[i]))

            all_train_image_sequences = np.array(all_train_image_sequences)
            all_train_label_sequences = np.array(all_train_label_sequences)

            # prepare pytorch training data
            all_train_image_sequences = torch.from_numpy(all_train_image_sequences).float().permute(0, 1, 2, 5, 3, 4)
            all_train_label_sequences = torch.from_numpy(all_train_label_sequences).float().permute(0, 1, 2, 5, 3, 4)

            # load validation dataset
            val_dataset = h5py.File(val_dir, 'r')
            # list the number of sequences in this dataset
            num_val_sequences = len(list(val_dataset.keys())) // 2

            all_val_image_sequences = []
            all_val_label_sequences = []
            for i in range(len(list(val_dataset.keys()))):
                if list(val_dataset.keys())[i].startswith('image'):
                    all_val_image_sequences.append(val_dataset.get(list(val_dataset.keys())[i]))
                elif list(val_dataset.keys())[i].startswith('label'):
                    all_val_label_sequences.append(val_dataset.get(list(val_dataset.keys())[i]))

            all_val_image_sequences = np.array(all_val_image_sequences)
            all_val_label_sequences = np.array(all_val_label_sequences)

            # prepare pytorch training data
            all_val_image_sequences = torch.from_numpy(all_val_image_sequences).float().permute(0, 1, 2, 5, 3, 4)
            all_val_label_sequences = torch.from_numpy(all_val_label_sequences).float().permute(0, 1, 2, 5, 3, 4)

            # parameters loaded from input data
            tile_size = (all_train_label_sequences.shape[4], all_train_label_sequences.shape[5])
            num_channels = all_train_image_sequences.shape[-3]
            num_tiles_per_image = all_train_image_sequences.shape[1]
        elif data_type == 'image-pair':
            # Read data
            train_img1_name_list, train_img2_name_list, train_gt_name_list = pair_data.read_all(train_dir)
            val_img1_name_list, val_img2_name_list, val_gt_name_list = pair_data.read_all(val_dir)
            # construct dataset
            train_data, train_labels = pair_data.construct_dataset(train_img1_name_list,
                                                                        train_img2_name_list,
                                                                        train_gt_name_list)

            val_data, val_labels = pair_data.construct_dataset(val_img1_name_list,
                                                                    val_img2_name_list,
                                                                    val_gt_name_list)

            num_channels = train_data.shape[1] // 2

        if verbose:
            print(f'\nGPU usage: {device}')
            print(f'netowrk model: {network_model}')
            print(f'dataset type: {data_type}')
            print(f'input training data dir: {train_dir}')
            if data_type == 'multi-frame':
                print(f'input validation data dir: {val_dir}')
            print(f'input checkpoint path: {checkpoint_path}')
            print(f'output model dir: {model_dir}')
            print(f'output figures dir: {figs_dir}')
            print(f'epoch size: {num_epoch}')
            print(f'batch size: {batch_size}')
            print(f'loss function: {loss}')
            print(f'number of image channel: {num_channels}')
            print(f'time span: {time_span}')
            if data_type == 'multi-frame' or data_type == 'one-sided':
                print(f'tile size: ({tile_size[0]}, {tile_size[1]})')
                print(f'num_tiles_per_image: {num_tiles_per_image}')
                print(f'\n{num_train_sequences} sequences of training data is detected')
                print(f'all_train_image_sequences has shape {all_train_image_sequences.shape}')
                print(f'all_train_label_sequences has shape {all_train_label_sequences.shape}')
                print(f'\n{num_val_sequences} sequences of validation data is detected')
                print(f'all_val_image_sequences has shape {all_val_image_sequences.shape}')
                print(f'all_val_label_sequences has shape {all_val_label_sequences.shape}')
            elif data_type == 'image-pair':
                print(f'train_data has shape: {train_data.shape}')
                print(f'train_labels has shape: {train_labels.shape}')
                print(f'val_data has shape: {val_data.shape}')
                print(f'val_labels has shape: {val_labels.shape}')

        # mini-batch training/validation manager for each batch
        class Manager():
            def __init__(self, data_type, lmsi_model, lmsi_loss, optimizer, time_span):
                self.data_type = data_type
                self.model = lmsi_model
                self.lmsi_loss = lmsi_loss
                if lmsi_loss == 'MSE' or lmsi_loss == 'RMSE':
                    self.loss_module = torch.nn.MSELoss()
                elif lmsi_loss == 'MAE':
                    self.loss_module = torch.nn.L1Loss()
                else:
                    raise Exception(f'Unrecognized loss function: {lmsi_loss}')

                self.optimizer = optimizer
                self.time_span = time_span

            def train(self, image_sequence, label_sequence):
                # image_sequence has shape (batch_size, time_range, channel, image_height, image_width)
                # label_sequence has shape (batch_size, time_range, target_dim, image_height//2, image_width//2)
                self.model.train(True)

                # losses for all mini-batches
                all_losses = []
                batch_size = image_sequence.shape[0]
                # image size and number of channels
                image_size = (image_sequence.shape[3], image_sequence.shape[4])
                channel = image_sequence.shape[2]

                # multi-frame is the standard version where [t-T//2, t+T//2] frames are input, estimate frame t
                if self.data_type == 'multi-frame':
                    for t in range(time_span//2, image_sequence.shape[1]-time_span//2):
                        mini_batch_start_time = time.time()

                        # construct an image block with time_span
                        cur_image_block = np.zeros((batch_size, channel*time_span, image_size[0], image_size[1]))
                        cur_block_indices = list(range(t-time_span//2, t+time_span//2+1))

                        # construct image block and send to GPU (we know that channel is 1)
                        cur_image_block = image_sequence[:, cur_block_indices, 0].to(device)
                        # construct label tile and send to GPU
                        cur_label_true = label_sequence[:, t].to(device)

                        # train/validate
                        cur_label_pred = self.model(cur_image_block)

                        # print("Outside: input size", cur_image_block.size(),
                        #         "output_size", cur_label_pred.size())

                        # compute loss
                        loss = self.loss_module(cur_label_pred, cur_label_true)
                        if self.lmsi_loss == 'RMSE':
                            loss = torch.sqrt(loss)

                        # Before the backward pass, use the optimizer object to zero all of the
                        # gradients for the variables it will update
                        self.optimizer.zero_grad()

                        # Backward pass: compute gradient of the loss with respect to model parameters
                        loss.backward()

                        # update to the model parameters
                        self.optimizer.step()

                        # save the loss
                        all_losses.append(loss.detach().item())

                        mini_batch_end_time = time.time()
                        mini_batch_time_cost = mini_batch_end_time - mini_batch_start_time

                        # show mini-batch progress
                        print_progress_bar(iteration=t-time_span//2+1,
                                            total=image_sequence.shape[1]-time_span+1,
                                            prefix=f'Mini-batch {t-time_span//2+1}/{image_sequence.shape[1]-time_span+1},',
                                            suffix='%s loss: %.3f, time: %.2f' % (mode, all_losses[-1], mini_batch_time_cost),
                                            length=50)

                # one-sided is the modified version where [t-T, t] frames are input, estimate frame t
                # we are using the multi-frame with T=9 dataset
                # it has 252 frames padded with 4 additional frames on both beginning and end, in total 260
                # in one-sided (left) version, T=5, we start at index 0, end at index 251
                elif self.data_type == 'one-sided':
                    for t in range(0, 252):
                        mini_batch_start_time = time.time()

                        # construct an image block with time_span
                        cur_image_block = np.zeros((batch_size, channel*time_span, image_size[0], image_size[1]))
                        cur_block_indices = list(range(t, t+time_span))

                        # construct image block and send to GPU (we know that channel is 1)
                        cur_image_block = image_sequence[:, cur_block_indices, 0].to(device)
                        # construct label tile and send to GPU
                        cur_label_true = label_sequence[:, cur_block_indices[-1]].to(device)

                        # train/validate
                        cur_label_pred = self.model(cur_image_block)

                        # compute loss
                        loss = self.loss_module(cur_label_pred, cur_label_true)
                        if self.lmsi_loss == 'RMSE':
                            loss = torch.sqrt(loss)

                        # Before the backward pass, use the optimizer object to zero all of the
                        # gradients for the variables it will update
                        self.optimizer.zero_grad()

                        # Backward pass: compute gradient of the loss with respect to model parameters
                        loss.backward()

                        # update to the model parameters
                        self.optimizer.step()

                        # save the loss
                        all_losses.append(loss.detach().item())

                        mini_batch_end_time = time.time()
                        mini_batch_time_cost = mini_batch_end_time - mini_batch_start_time

                        # show mini-batch progress
                        print_progress_bar(iteration=t+1,
                                            total=252,
                                            prefix=f'Mini-batch {t+1}/252,',
                                            suffix='%s loss: %.3f, time: %.2f' % (mode, all_losses[-1], mini_batch_time_cost),
                                            length=50)
                return all_losses

            def validate(self, image_sequence, label_sequence):
                # image_sequence has shape (batch_size, time_range, channel, image_height, image_width)
                # label_sequence has shape (batch_size, time_range, target_dim, image_height//2, image_width//2)
                # self.model.train(False)
                self.model.eval()

                with torch.no_grad():

                    # losses for all mini-batches
                    all_losses = []
                    batch_size = image_sequence.shape[0]
                    # image size and number of channels
                    image_size = (image_sequence.shape[3], image_sequence.shape[4])
                    channel = image_sequence.shape[2]

                    if self.data_type == 'multi-frame':
                        for t in range(time_span//2, image_sequence.shape[1]-time_span//2):
                            mini_batch_start_time = time.time()

                            # construct an image block with time_span
                            cur_image_block = np.zeros((batch_size, channel*time_span, image_size[0], image_size[1]))
                            cur_block_indices = list(range(t-time_span//2, t+time_span//2+1))

                            # construct image block and send to GPU (we know that channel is 1)
                            cur_image_block = image_sequence[:, cur_block_indices, 0].to(device)
                            # construct label tile and send to GPU
                            cur_label_true = label_sequence[:, t].to(device)

                            # train/validate
                            cur_label_pred = self.model(cur_image_block)

                            # compute loss
                            loss = self.loss_module(cur_label_pred, cur_label_true)
                            if self.lmsi_loss == 'RMSE':
                                loss = torch.sqrt(loss)

                            # save the loss
                            all_losses.append(loss.detach().item())

                            mini_batch_end_time = time.time()
                            mini_batch_time_cost = mini_batch_end_time - mini_batch_start_time

                            # show mini-batch progress
                            print_progress_bar(iteration=t-time_span//2+1,
                                                total=image_sequence.shape[1]-time_span+1,
                                                prefix=f'Mini-batch {t-time_span//2+1}/{image_sequence.shape[1]-time_span+1},',
                                                suffix='%s loss: %.3f, time: %.2f' % (mode, all_losses[-1], mini_batch_time_cost),
                                                length=50)

                    elif data_type == 'one-sided':
                        for t in range(0, 252):
                            mini_batch_start_time = time.time()

                            # construct an image block with time_span
                            cur_image_block = np.zeros((batch_size, channel*time_span, image_size[0], image_size[1]))
                            cur_block_indices = list(range(t, t+time_span))

                            # construct image block and send to GPU (we know that channel is 1)
                            cur_image_block = image_sequence[:, cur_block_indices, 0].to(device)
                            # construct label tile and send to GPU
                            cur_label_true = label_sequence[:, cur_block_indices[-1]].to(device)

                            # train/validate
                            cur_label_pred = self.model(cur_image_block)

                            # compute loss
                            loss = self.loss_module(cur_label_pred, cur_label_true)
                            if self.lmsi_loss == 'RMSE':
                                loss = torch.sqrt(loss)

                            # save the loss
                            all_losses.append(loss.detach().item())

                            mini_batch_end_time = time.time()
                            mini_batch_time_cost = mini_batch_end_time - mini_batch_start_time

                            # show mini-batch progress
                            print_progress_bar(iteration=t+1,
                                                total=252,
                                                prefix=f'Mini-batch {t+1}/252,',
                                                suffix='%s loss: %.3f, time: %.2f' % (mode, all_losses[-1], mini_batch_time_cost),
                                                length=50)


                    return all_losses

        # memory network parameters
        kwargs = {
                    'num_channels':               num_channels,
                    'time_span':                  time_span,
                    'target_dim':                 2,
                 }

        # model, optimizer and loss
        lmsi_model = None
        if network_model == 'memory-piv-net':
            lmsi_model = models.Memory_PIVnet(**kwargs)
        elif network_model == 'memory-piv-net-no-neighbor' or network_model == 'memory-piv-net-ip':
            lmsi_model = models.Memory_PIVnet_No_Neighbor(**kwargs)

        # load checkpoint info if existing
        starting_epoch = 0
        if checkpoint_path != None:
            checkpoint = torch.load(checkpoint_path)
            lmsi_model.load_state_dict(checkpoint['state_dict'])
            starting_epoch = checkpoint['epoch']

        if torch.cuda.device_count() > 1:
            print('\nUsing', torch.cuda.device_count(), 'GPUs')
            lmsi_model = torch.nn.DataParallel(lmsi_model)

        # move model to GPU
        lmsi_model.to(device)

        # define optimizer
        if data_type == 'multi-frame' or data_type == 'one-sided':
            lmsi_optimizer = torch.optim.Adam(lmsi_model.parameters(), lr=1e-4)
        elif data_type == 'image-pair':
            lmsi_optimizer = torch.optim.Adam(lmsi_model.parameters(), lr=1e-4)

        if checkpoint_path != None:
            lmsi_optimizer.load_state_dict(checkpoint['optimizer'])

        # training for a number of epochs
        train_start_time = time.time()
        # train/val losses for all the epochs
        all_epoch_train_losses = []
        all_epoch_val_losses = []
        for i in range(starting_epoch, starting_epoch+num_epoch):
            print(f'\n Starting epoch {i+1}/{starting_epoch+num_epoch}')
            epoch_start_time = time.time()

            # train/val losses for all the batches
            all_batch_train_losses = []
            all_batch_val_losses = []

            if data_type == 'multi-frame' or data_type == 'one-sided':
                # training manager for each batch
                training_manager = Manager(data_type, lmsi_model, loss, lmsi_optimizer, time_span)
                # assume training data has shape (5, 16, 260, 1, 128, 128)
                # which means 5 independent sequences, where each is splitted into 16 tile-sequences
                # since time span is 9, the total time stamp is 252
                # each neighbor tile has size 1*128*128 (channel first)
                # random number to take one of the independent sequences
                for phase in ['train', 'val']:
                    if phase == 'train':
                        sequence_order = random.sample(range(num_train_sequences), num_train_sequences)
                        print(f'\ntraining sequence(s) is/are {sequence_order}')
                    elif phase == 'val':
                        sequence_order = random.sample(range(num_val_sequences), num_val_sequences)
                        print(f'\nvalidation sequence(s) is/are {sequence_order}')

                    # train/validate for each sequence
                    for sequence_index in sequence_order:
                        if phase == 'train':
                            print(f'training with sequence {sequence_index}')
                            cur_image_sequence = all_train_image_sequences[sequence_index]
                            cur_label_sequence = all_train_label_sequences[sequence_index]
                        elif phase == 'val':
                            print(f'validation with sequence {sequence_index}')
                            cur_image_sequence = all_val_image_sequences[sequence_index]
                            cur_label_sequence = all_val_label_sequences[sequence_index]

                        # construct data loader based on this
                        # cur_image_sequence has shape (16, 260, 1, 128, 128)
                        # dataloader randomly loads a batch_size number of tile-sequence(s)
                        data = torch.utils.data.TensorDataset(cur_image_sequence, cur_label_sequence)
                        dataloader = torch.utils.data.DataLoader(data,
                                                                batch_size=batch_size,
                                                                shuffle=True,
                                                                num_workers=4)

                        # number for batches used to plot progress bar
                        num_batch = len(cur_image_sequence) // batch_size
                        for j, (batch_data, batch_labels) in enumerate(dataloader):
                            # has shape (batch_size, 260, 1, 128, 128)
                            batch_start_time = time.time()
                            # run with returned all mini-batch losses
                            if phase == 'train':
                                all_mini_batch_losses = training_manager.train(batch_data, batch_labels)
                            elif phase == 'val':
                                all_mini_batch_losses = training_manager.validate(batch_data, batch_labels)

                            # take the average of mini-batch loss
                            mini_batch_avg_loss = np.mean(all_mini_batch_losses)
                            batch_end_time = time.time()
                            batch_time_cost = batch_end_time - batch_start_time

                            if phase == 'train':
                                all_batch_train_losses.append(mini_batch_avg_loss)
                                print('\nTraining batch %d/%d completed in %.3f seconds, avg train loss: %.3f\n'
                                        % ((j+1), num_batch, batch_time_cost, all_batch_train_losses[-1]))
                            elif phase == 'val':
                                all_batch_val_losses.append(mini_batch_avg_loss)
                                print('\nValidation batch %d/%d completed in %.3f seconds, avg val loss: %.3f\n'
                                        % ((j+1), num_batch, batch_time_cost, all_batch_val_losses[-1]))

                            # del all_mini_batch_losses, mini_batch_avg_loss

            elif data_type == 'image-pair':
                # define loss
                if loss == 'MSE' or loss == 'RMSE':
                    loss_module = torch.nn.MSELoss()
                elif loss == 'MAE':
                    loss_module = torch.nn.L1Loss()
                else:
                    raise Exception(f'Unrecognized loss function: {loss}')

                # assume training data has shape (1800, 2, 256, 256)
                # which corresponds to each image pair
                # have a data loader that select the image pair
                for phase in ['train', 'val']:
                    # dataloader randomly loads a batch_size number of image pairs
                    if phase == 'train':
                        data = torch.utils.data.TensorDataset(train_data, train_labels)
                        dataloader = torch.utils.data.DataLoader(data,
                                                                batch_size=batch_size,
                                                                shuffle=True,
                                                                num_workers=4)
                        # number for batches used to plot progress bar
                        num_batch = int(np.ceil(len(train_data) / batch_size))
                    elif phase == 'val':
                        data = torch.utils.data.TensorDataset(val_data, val_labels)
                        dataloader = torch.utils.data.DataLoader(data,
                                                                batch_size=batch_size,
                                                                shuffle=True,
                                                                num_workers=4)
                        # number for batches used to plot progress bar
                        num_batch = int(np.ceil(len(val_data) / batch_size))

                    for j, (batch_data, batch_labels) in enumerate(dataloader):
                        # has shape (batch_size, 260, 1, 128, 128)
                        batch_start_time = time.time()
                        # send data to GPU
                        batch_data = batch_data.to(device)
                        batch_labels = batch_labels.to(device)

                        if phase == 'train':
                            # train/validate
                            cur_label_pred = lmsi_model(batch_data)

                            # compute loss
                            train_loss = loss_module(cur_label_pred, batch_labels)
                            if loss == 'RMSE':
                                train_loss = torch.sqrt(train_loss)

                            # Before the backward pass, use the optimizer object to zero all of the
                            # gradients for the variables it will update
                            lmsi_optimizer.zero_grad()

                            # Backward pass: compute gradient of the loss with respect to model parameters
                            train_loss.backward()

                            # update to the model parameters
                            lmsi_optimizer.step()

                            # save the loss
                            all_batch_train_losses.append(train_loss.detach().item())

                            # batch end time
                            batch_end_time = time.time()
                            batch_time_cost = batch_end_time - batch_start_time

                            # show mini-batch progress
                            print_progress_bar(iteration=j+1,
                                                total=num_batch,
                                                prefix=f'Batch {j+1}/{num_batch},',
                                                suffix='%s loss: %.3f, time: %.2f' % (phase+' '+loss, all_batch_train_losses[-1], batch_time_cost),
                                                length=50)

                        elif phase == 'val':
                            lmsi_model.eval()

                            with torch.no_grad():

                                # train/validate
                                cur_label_pred = lmsi_model(batch_data)

                                # compute loss
                                val_loss = loss_module(cur_label_pred, batch_labels)
                                if loss == 'RMSE':
                                    val_loss = torch.sqrt(val_loss)

                                # save the loss
                                all_batch_val_losses.append(val_loss.detach().item())

                            # batch end time
                            batch_end_time = time.time()
                            batch_time_cost = batch_end_time - batch_start_time

                            # show mini-batch progress
                            print_progress_bar(iteration=j+1,
                                                total=num_batch,
                                                prefix=f'Batch {j+1}/{num_batch},',
                                                suffix='%s loss: %.3f, time: %.2f' % (phase+' '+loss, all_batch_val_losses[-1], batch_time_cost),
                                                length=50)

                    print('\n')

            epoch_end_time = time.time()
            batch_avg_train_loss = np.mean(all_batch_train_losses)
            batch_avg_val_loss = np.mean(all_batch_val_losses)
            all_epoch_train_losses.append(batch_avg_train_loss)
            all_epoch_val_losses.append(batch_avg_val_loss)
            print('\nEpoch %d completed in %.3f seconds, avg train loss: %.3f, avg val loss: %.3f'
                        % ((i+1), (epoch_end_time-epoch_start_time), all_epoch_train_losses[-1], all_epoch_val_losses[-1]))

            # save loss graph and model
            if checkpoint_path != None:
                prev_train_losses = checkpoint['train_loss']
                prev_val_losses = checkpoint['val_loss']
                all_epoch_train_losses = prev_train_losses + all_epoch_train_losses
                all_epoch_val_losses = prev_val_losses + all_epoch_val_losses

            plt.plot(all_epoch_train_losses, label='Train')
            plt.plot(all_epoch_val_losses, label='Validation')
            plt.title(f'Training and validation loss on {network_model} model')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loc='upper right')
            loss_path = os.path.join(figs_dir, f'{network_model}_{data_type}_{time_span}_{batch_size}_{i+1}_loss.png')
            plt.savefig(loss_path)
            print(f'\nLoss graph has been saved to {loss_path}')

            # save model as a checkpoint so further training could be resumed
            model_path = os.path.join(model_dir, f'{network_model}_{data_type}_{time_span}_{batch_size}_{i+1}.pt')
            # if trained on multiple GPU's, store model.module.state_dict()
            if torch.cuda.device_count() > 1:
                model_checkpoint = {
                                        'epoch': i+1,
                                        'state_dict': lmsi_model.module.state_dict(),
                                        'optimizer': lmsi_optimizer.state_dict(),
                                        'train_loss': all_epoch_train_losses,
                                        'val_loss': all_epoch_val_losses
                                    }
            else:
                model_checkpoint = {
                                        'epoch': i+1,
                                        'state_dict': lmsi_model.state_dict(),
                                        'optimizer': lmsi_optimizer.state_dict(),
                                        'train_loss': all_epoch_train_losses,
                                        'val_loss': all_epoch_val_losses
                                    }

            torch.save(model_checkpoint, model_path)
            print(f'\nTrained model checkpoint has been saved to {model_path}\n')

        train_end_time = time.time()
        print('\nTraining completed in %.3f seconds' % (train_end_time-train_start_time))


    # new test mode for train-new that loads model and perform prediction
    elif mode == 'test':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if verbose:
            print(f'\nmode: {mode}')
            print(f'\nIn {mode} mode, input_dir is the directory that contains preprocessed .h5 dataset')
            print('\nmodel_dir is required in this mode, which is the trained model used to run predictions')
            print('\noutput_dir in this mode is the directory where figures, etc are saved to')

        network_model = args.network_model[0]
        test_dir = args.test_dir[0]
        # data_type can be image-pair or multi-frame (default)
        if args.data_type != None:
            data_type = args.data_type[0]
        else:
            data_type = 'multi-frame'
        model_dir = args.model_dir[0]
        figs_dir = args.output_dir[0]
        time_span = int(args.time_span[0])
        # useful arguments in this mode
        target_dim = 2
        loss = args.loss[0]
        final_size = 256

        # load testing dataset
        if data_type == 'multi-frame':
            test_dataset = h5py.File(test_dir, 'r')
            # list the number of sequences in this dataset
            num_test_sequences = len(list(test_dataset.keys())) // 2

            all_test_image_sequences = []
            all_test_label_sequences = []
            for i in range(len(list(test_dataset.keys()))):
                if list(test_dataset.keys())[i].startswith('image'):
                    all_test_image_sequences.append(test_dataset.get(list(test_dataset.keys())[i]))
                elif list(test_dataset.keys())[i].startswith('label'):
                    all_test_label_sequences.append(test_dataset.get(list(test_dataset.keys())[i]))

            all_test_image_sequences = np.array(all_test_image_sequences)
            all_test_label_sequences = np.array(all_test_label_sequences)

            # prepare pytorch dataset
            all_test_image_sequences = torch.from_numpy(all_test_image_sequences).float().permute(0, 1, 2, 5, 3, 4)
            all_test_label_sequences = torch.from_numpy(all_test_label_sequences).float().permute(0, 1, 2, 5, 3, 4)

            # parameters loaded from input data
            tile_size = (all_test_label_sequences.shape[4], all_test_label_sequences.shape[5])
            num_channels = all_test_image_sequences.shape[-3]
            num_tiles_per_image = all_test_image_sequences.shape[1]
        elif data_type == 'image-pair':
            # Read data
            img1_name_list, img2_name_list, gt_name_list = pair_data.read_all(test_dir)
            # construct dataset
            test_data, test_labels = pair_data.construct_dataset(img1_name_list, img2_name_list, gt_name_list)
            # parameters loaded from input data
            num_channels = test_data.shape[1] // 2

        if verbose:
            print(f'\nGPU usage: {device}')
            print(f'netowrk model: {network_model}')
            print(f'testing data dir: {test_dir}')
            print(f'input model dir: {model_dir}')
            print(f'output figures dir: {figs_dir}')
            print(f'loss function: {loss}')
            print(f'number of image channel: {num_channels}')
            print(f'time span: {time_span}')
            if data_type == 'multi-frame':
                print(f'tile size: ({tile_size[0]}, {tile_size[1]})')
                print(f'num_tiles_per_image: {num_tiles_per_image}')
                # print(f'\n{num_test_sequences} sequences of testing data is detected')
                print(f'all_test_image_sequences has shape {all_test_image_sequences.shape}')
                print(f'all_test_label_sequences has shape {all_test_label_sequences.shape}')
            elif data_type == 'image-pair':
                print(f'test_data has shape: {test_data.shape}')
                print(f'test_labels has shape: {test_labels.shape}')


        # run testing inference
        if data_type == 'multi-frame':
            # start and end of t (both inclusive)
            start_t = 41
            end_t = 41
            print(f'Testing from t = {start_t} to {end_t} (both side inclusive)')

            run_test(network_model,
                    all_test_image_sequences,
                    all_test_label_sequences,
                    model_dir,
                    figs_dir,
                    loss,
                    start_t,
                    end_t,
                    num_channels,
                    time_span,
                    target_dim,
                    num_tiles_per_image,
                    final_size,
                    device,
                    blend=True,
                    draw_normal=True,
                    draw_glyph=False)

            print(f'\nModel inference on image [{start_t}:{end_t}] completed\n')

        elif data_type == 'image-pair':
            with torch.no_grad():
                # start and end of index (both inclusive)
                start_index = 41
                end_index = 41
                visualize = True
                # check if input parameters are valid
                if start_index < 0:
                    raise Exception('Invalid start_index')
                elif end_index > test_data.shape[0]-1:
                    raise Exception(f'Invalid end_index > total number of image pair {test_data.shape[0]}')

                # define the loss
                if loss == 'MSE' or loss == 'RMSE':
                    loss_module = torch.nn.MSELoss()
                elif loss == 'MAE':
                    loss_module = torch.nn.L1Loss()

                # memory network parameters
                kwargs = {
                            'num_channels':               num_channels,
                            'time_span':                  time_span,
                            'target_dim':                 target_dim,
                            'mode':                       'test',
                            'N':                          4
                        }

                # model, optimizer and loss
                lmsi_model = None
                if network_model == 'memory-piv-net':
                    lmsi_model = models.Memory_PIVnet(**kwargs)

                lmsi_model.eval()
                lmsi_model.to(device)

                # load trained model
                trained_model = torch.load(model_dir)
                lmsi_model.load_state_dict(trained_model['state_dict'])

                min_loss = 999
                min_loss_index = 0
                all_losses = []

                for k in range(start_index, end_index+1):
                    cur_image_pair = test_data[k:k+1].to(device)
                    cur_label_true = test_labels[k].permute(1, 2, 0).numpy() / final_size * 100
                    # get prediction from loaded model
                    prediction = lmsi_model(cur_image_pair)

                    # put on cpu and permute to channel last
                    cur_label_pred = prediction.cpu().data
                    cur_label_pred = cur_label_pred.permute(0, 2, 3, 1).numpy()
                    cur_label_pred = cur_label_pred[0] / final_size * 100

                    # compute loss
                    cur_loss = loss_module(torch.from_numpy(cur_label_pred), torch.from_numpy(cur_label_true))
                    if loss == 'RMSE':
                        cur_loss = torch.sqrt(cur_loss)
                    elif loss == 'AEE':
                        sum_endpoint_error = 0
                        for i in range(final_size):
                            for j in range(final_size):
                                cur_pred = cur_label_pred[i, j]
                                cur_true = cur_label_true[i, j]
                                cur_endpoint_error = np.linalg.norm(cur_pred-cur_true)
                                sum_endpoint_error += cur_endpoint_error

                        # compute the average endpoint error
                        cur_loss = sum_endpoint_error / (final_size*final_size)

                    if cur_loss < min_loss:
                        min_loss = cur_loss
                        min_loss_index = k

                    all_losses.append(cur_loss)

                    print(f'\nPrediction {loss} for {k}th image pair is {cur_loss}')

                    # visualize the flow
                    if visualize:
                        cur_flow_true, max_vel = plot.visualize_flow(cur_label_true)
                        cur_flow_pred, _ = plot.visualize_flow(cur_label_pred, max_vel=max_vel)

                        # convert to Image
                        cur_test_image1 = Image.fromarray(test_data[k, :, :, 0].numpy())
                        cur_test_image2 = Image.fromarray(test_data[k, :, :, 1].numpy())
                        cur_flow_true = Image.fromarray(cur_flow_true)
                        cur_flow_pred = Image.fromarray(cur_flow_pred)

                        # superimpose quiver plot on color-coded images
                        # ground truth
                        x = np.linspace(0, final_size-1, final_size)
                        y = np.linspace(0, final_size-1, final_size)
                        y_pos, x_pos = np.meshgrid(x, y)
                        skip = 8
                        plt.figure()
                        plt.imshow(cur_flow_true)
                        Q = plt.quiver(y_pos[::skip, ::skip],
                                        x_pos[::skip, ::skip],
                                        cur_label_true[::skip, ::skip, 0]/max_vel,
                                        -cur_label_true[::skip, ::skip, 1]/max_vel,
                                        scale=4.0,
                                        scale_units='inches')
                        Q._init()
                        assert isinstance(Q.scale, float)
                        print(f'\nQuiver plot scale is {Q.scale}')
                        plt.axis('off')
                        true_quiver_path = os.path.join(figs_dir, f'Memory-PIVnet_{k}_true.svg')
                        plt.savefig(true_quiver_path, bbox_inches='tight', dpi=1200)
                        print(f'ground truth plot has been saved to {true_quiver_path}')

                        # prediction
                        plt.figure()
                        plt.imshow(cur_flow_pred)
                        plt.quiver(y_pos[::skip, ::skip],
                                    x_pos[::skip, ::skip],
                                    cur_label_pred[::skip, ::skip, 0]/max_vel,
                                    -cur_label_pred[::skip, ::skip, 1]/max_vel,
                                    scale=Q.scale,
                                    scale_units='inches')
                        plt.axis('off')
                        pred_quiver_path = os.path.join(figs_dir, f'Memory-PIVnet_{k}_ip_pred.svg')
                        # annotate error
                        plt.annotate(f'RMSE: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='large')
                        plt.savefig(pred_quiver_path, bbox_inches='tight', dpi=1200)
                        print(f'prediction plot has been saved to {pred_quiver_path}')

                        # magnitude difference
                        pred_error = np.sqrt(cur_label_pred[:,:,0]**2 + cur_label_pred[:,:,1]**2) \
                                                - np.sqrt(cur_label_true[:,:,0]**2 + cur_label_true[:,:,1]**2)
                        plt.figure()
                        plt.imshow(pred_error, cmap='RdBu', interpolation='nearest', vmin=-1,  vmax=1)
                        error_path = os.path.join(figs_dir, f'Memory-PIVnet_{k}_ip_error.svg')
                        plt.axis('off')
                        cbar = plt.colorbar()
                        cbar.set_label('Vector magnitude difference')
                        plt.savefig(error_path, bbox_inches='tight', dpi=1200)
                        print(f'error magnitude plot has been saved to {error_path}')

            print(f'\nModel inference on image [{start_index}:{end_index}] completed\n')

            avg_loss = np.mean(all_losses)
            print(f'Average RMSE across {end_index - start_index + 1} samples is {avg_loss}')
            print(f'Min loss is {min_loss} at index {min_loss_index}')


if __name__ == "__main__":
    main()
