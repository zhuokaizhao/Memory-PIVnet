# Learning models of project Learning how to measure scientific images
# Author: Zhuokai Zhao

import os
import time
import math
import torch
import itertools
import numpy as np
from PIL import Image

# use cudnn
torch.backends.cudnn.enabled = True

# Standard Memory-PIVnet
class Memory_PIVnet(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Memory_PIVnet, self).__init__()

        # date channel
        self.num_channels = kwargs['num_channels']
        # number of time frames (including the current frame)
        self.time_span = kwargs['time_span']
        # target (flow) dimension
        self.target_dim = kwargs['target_dim']
        # GPU info
        self.device = kwargs['device']

        # Multigrid Memory Network
        class MemoryNetwork(torch.nn.Module):

            # input_channels: number of input data channels
            # all_hidden_channels: list of channel number for every memory layer
            def __init__(self, input_channels, all_hidden_channels):
                super(MemoryNetwork, self).__init__()

                self.num_layer = 0
                self.input_channels = input_channels
                self.all_hidden_channels = all_hidden_channels

                # Multigrid Memory layer with batch normalization
                class MemoryLayer(torch.nn.Module):
                    # layer_index: the memory layer index in the memory network
                    # num_level: levels of different resolutions
                    # num_up: number of upsampling needed
                    # num_down: number of downsampling needed
                    # input_channels: number of input data channels
                    # kernel_size: conv kernel for conv lstm
                    def __init__(self, layer_index, num_level, num_up, num_down, input_channels, hidden_channels, kernel_size):
                        super(MemoryLayer, self).__init__()

                        # number of different resolutions
                        self.layer_index = layer_index
                        self.num_level = num_level
                        self.num_up = num_up
                        self.num_down = num_down
                        self.input_channels = input_channels
                        self.hidden_channels = hidden_channels
                        self.kernel_size = kernel_size

                        # ConvLSTM cell
                        class ConvLSTMCell(torch.nn.Module):
                            def __init__(self, input_channels, hidden_channels, kernel_size):
                                super(ConvLSTMCell, self).__init__()

                                assert hidden_channels % 2 == 0

                                self.input_channels = input_channels
                                self.hidden_channels = hidden_channels
                                self.kernel_size = kernel_size

                                # To ensure that the states have the same number of rows
                                # and same number of columns as the inputs, padding is required before convolution
                                self.padding = (kernel_size - 1) // 2

                                # input gate
                                self.Wxi = torch.nn.Conv2d(in_channels=self.input_channels,
                                                            out_channels=self.hidden_channels,
                                                            kernel_size=self.kernel_size,
                                                            stride=1,
                                                            padding=self.padding,
                                                            bias=True)
                                self.Whi = torch.nn.Conv2d(in_channels=self.hidden_channels,
                                                            out_channels=self.hidden_channels,
                                                            kernel_size=self.kernel_size,
                                                            stride=1,
                                                            padding=self.padding,
                                                            bias=False)

                                self.WxiBN = torch.nn.Sequential(torch.nn.Conv2d(in_channels=self.input_channels,
                                                                                 out_channels=self.hidden_channels,
                                                                                 kernel_size=self.kernel_size,
                                                                                 stride=1,
                                                                                 padding=self.padding,
                                                                                 bias=True),
                                                                 torch.nn.BatchNorm2d(num_features=self.hidden_channels))
                                self.WhiBN = torch.nn.Sequential(torch.nn.Conv2d(in_channels=self.hidden_channels,
                                                                                 out_channels=self.hidden_channels,
                                                                                 kernel_size=self.kernel_size,
                                                                                 stride=1,
                                                                                 padding=self.padding,
                                                                                 bias=False),
                                                                 torch.nn.BatchNorm2d(num_features=self.hidden_channels))

                                # forget gate
                                self.Wxf = torch.nn.Conv2d(in_channels=self.input_channels,
                                                            out_channels=self.hidden_channels,
                                                            kernel_size=self.kernel_size,
                                                            stride=1,
                                                            padding=self.padding,
                                                            bias=True)
                                self.Whf = torch.nn.Conv2d(in_channels=self.hidden_channels,
                                                            out_channels=self.hidden_channels,
                                                            kernel_size=self.kernel_size,
                                                            stride=1,
                                                            padding=self.padding,
                                                            bias=False)

                                self.WxfBN = torch.nn.Sequential(torch.nn.Conv2d(in_channels=self.input_channels,
                                                                                out_channels=self.hidden_channels,
                                                                                kernel_size=self.kernel_size,
                                                                                stride=1,
                                                                                padding=self.padding,
                                                                                bias=True),
                                                                torch.nn.BatchNorm2d(num_features=self.hidden_channels))
                                self.WhfBN = torch.nn.Sequential(torch.nn.Conv2d(in_channels=self.hidden_channels,
                                                                                out_channels=self.hidden_channels,
                                                                                kernel_size=self.kernel_size,
                                                                                stride=1,
                                                                                padding=self.padding,
                                                                                bias=False),
                                                                torch.nn.BatchNorm2d(num_features=self.hidden_channels))

                                # candidate
                                self.Wxc = torch.nn.Conv2d(in_channels=self.input_channels,
                                                            out_channels=self.hidden_channels,
                                                            kernel_size=self.kernel_size,
                                                            stride=1,
                                                            padding=self.padding,
                                                            bias=True)
                                self.Whc = torch.nn.Conv2d(in_channels=self.hidden_channels,
                                                            out_channels=self.hidden_channels,
                                                            kernel_size=self.kernel_size,
                                                            stride=1,
                                                            padding=self.padding,
                                                            bias=False)

                                self.WxcBN = torch.nn.Sequential(torch.nn.Conv2d(in_channels=self.input_channels,
                                                                                out_channels=self.hidden_channels,
                                                                                kernel_size=self.kernel_size,
                                                                                stride=1,
                                                                                padding=self.padding,
                                                                                bias=True),
                                                                torch.nn.BatchNorm2d(num_features=self.hidden_channels))
                                self.WhcBN = torch.nn.Sequential(torch.nn.Conv2d(in_channels=self.hidden_channels,
                                                                                out_channels=self.hidden_channels,
                                                                                kernel_size=self.kernel_size,
                                                                                stride=1,
                                                                                padding=self.padding,
                                                                                bias=False),
                                                                torch.nn.BatchNorm2d(num_features=self.hidden_channels))

                                # output
                                self.Wxo = torch.nn.Conv2d(in_channels=self.input_channels,
                                                            out_channels=self.hidden_channels,
                                                            kernel_size=self.kernel_size,
                                                            stride=1,
                                                            padding=self.padding,
                                                            bias=True)
                                self.Who = torch.nn.Conv2d(in_channels=self.hidden_channels,
                                                            out_channels=self.hidden_channels,
                                                            kernel_size=self.kernel_size,
                                                            stride=1,
                                                            padding=self.padding,
                                                            bias=False)

                                self.WxoBN = torch.nn.Sequential(torch.nn.Conv2d(in_channels=self.input_channels,
                                                                                out_channels=self.hidden_channels,
                                                                                kernel_size=self.kernel_size,
                                                                                stride=1,
                                                                                padding=self.padding,
                                                                                bias=True),
                                                                torch.nn.BatchNorm2d(num_features=self.hidden_channels))
                                self.WhoBN = torch.nn.Sequential(torch.nn.Conv2d(in_channels=self.hidden_channels,
                                                                                out_channels=self.hidden_channels,
                                                                                kernel_size=self.kernel_size,
                                                                                stride=1,
                                                                                padding=self.padding,
                                                                                bias=False),
                                                                torch.nn.BatchNorm2d(num_features=self.hidden_channels))

                                self.Wci = None
                                self.Wcf = None
                                self.Wco = None

                                # batch normalization layer for the final hidden state output
                                self.BN = torch.nn.BatchNorm2d(num_features=self.hidden_channels)

                            # prev_h is the previous timestamp hidden state
                            # prev_c is the previous timestamp candidate value
                            def forward(self, cur_x, prev_h, prev_c):
                                # current input gate result
                                # i_t = sigmoid(Wxi * x_t + Whi * h_{t-1} + Wci x c_{t-1} + b_i)
                                cur_i = torch.sigmoid(self.WxiBN(cur_x) + self.WhiBN(prev_h) + self.Wci * prev_c)

                                # current forget gate result
                                # f_t = sigmoid(Wxf * x_t + Whf * h_{t-1} + Wcf x c_{t-1} + b_f)
                                cur_f = torch.sigmoid(self.WxfBN(cur_x) + self.WhfBN(prev_h) + self.Wcf * prev_c)

                                # current candidate value (cell state)
                                # c_t = f_t x c_{t-1} + i_t x tanh(Wxc * x_t + Whc * h_{t-1} + b_c)
                                cur_c = cur_f * prev_c + cur_i * torch.tanh(self.WxcBN(cur_x) + self.WhcBN(prev_h))

                                # current cell output
                                # o_t = sigmoid(Wxo * x_t + Who * h_{t-1} + Wco x c_t + b_o)
                                cur_o = torch.sigmoid(self.WxoBN(cur_x) + self.WhoBN(prev_h) + self.Wco * cur_c)

                                # current hidden state output
                                # h_t = o_t x tanh(c_t)
                                cur_h = cur_o * torch.tanh(self.BN(cur_c))

                                # return the hidden and cell state
                                return cur_h, cur_c

                            def init_hidden(self, batch_size, hidden, shape):
                                if self.Wci is None:
                                    self.Wci = torch.autograd.Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
                                    self.Wcf = torch.autograd.Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
                                    self.Wco = torch.autograd.Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
                                else:
                                    assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
                                    assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'

                                return (torch.autograd.Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                                        torch.autograd.Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())

                        # downsample and upsample
                        self.down_sample = torch.nn.AvgPool2d(kernel_size=2,
                                                              stride=2,
                                                              padding=0)

                        self.double_down = torch.nn.Sequential(
                            torch.nn.AvgPool2d(kernel_size=2,
                                               stride=2,
                                               padding=0),
                            torch.nn.AvgPool2d(kernel_size=2,
                                               stride=2,
                                               padding=0)
                        )

                        # deconv used to upsample to double
                        self.up_sample = torch.nn.ConvTranspose2d(in_channels=self.input_channels,
                                                                  out_channels=self.input_channels,
                                                                  kernel_size=2,
                                                                  stride=2)

                        self.double_up = torch.nn.Sequential(
                            torch.nn.ConvTranspose2d(in_channels=self.input_channels,
                                                    out_channels=self.input_channels,
                                                    kernel_size=2,
                                                    stride=2),
                            torch.nn.ConvTranspose2d(in_channels=self.input_channels,
                                                    out_channels=self.input_channels,
                                                    kernel_size=2,
                                                    stride=2)
                        )

                        # a ConvLSTM on each level within the layer
                        for i in range(num_level):
                            lstm_name = f'ConvLSTM_Layer{self.layer_index}_Level{i}'
                            # input channels number is based on levels
                            if num_level == 2:
                                if i == 0:
                                    lstm_input_channel = 2*self.input_channels
                                elif i == 1:
                                    lstm_input_channel = 2*self.input_channels
                            elif num_level == 3:
                                if i == 0:
                                    lstm_input_channel = 2*self.input_channels
                                elif i == 1:
                                    lstm_input_channel = 3*self.input_channels
                                elif i == 2:
                                    lstm_input_channel = 2*self.input_channels
                            elif num_level == 4:
                                if i == 0:
                                    lstm_input_channel = 2*self.input_channels
                                elif i == 1:
                                    lstm_input_channel = 3*self.input_channels
                                elif i == 2:
                                    lstm_input_channel = 3*self.input_channels
                                elif i == 3:
                                    lstm_input_channel = 2*self.input_channels
                            elif num_level == 5:
                                if i == 0:
                                    lstm_input_channel = 3*self.input_channels
                                elif i == 1:
                                    lstm_input_channel = 4*self.input_channels
                                elif i == 2:
                                    lstm_input_channel = 5*self.input_channels
                                elif i == 3:
                                    lstm_input_channel = 4*self.input_channels
                                elif i == 4:
                                    lstm_input_channel = 3*self.input_channels

                            lstm = ConvLSTMCell(input_channels=lstm_input_channel,
                                                hidden_channels=self.hidden_channels,
                                                kernel_size=self.kernel_size)

                            setattr(self, lstm_name, lstm)

                    # h_prev_layer: h from the previous memory layer of the CURRENT time stamp, used as LSTM input
                    # h_prev_time/c_prev_time: h and c from the same memory layer of the PREVIOUS time stamp
                    def forward(self, t, k, h_prev_layer, h_prev_time, c_prev_time, long_term_memory):
                        # time stamp sanity check
                        if long_term_memory:
                            if t == 0 and k == 0:
                                if h_prev_time != [] or c_prev_time != []:
                                    raise Exception('First time stamp should have empty h_prev_time, c_prev_time')
                            else:
                                if len(h_prev_time) != self.num_level or len(c_prev_time) != self.num_level:
                                    raise Exception(f'Unmatched level number ({self.num_level}) with h_prev_time ({len(h_prev_time)}), c_prev_time ({len(c_prev_time)})')
                        else:
                            if k == 0:
                                if h_prev_time != [] or c_prev_time != []:
                                    raise Exception('First time stamp should have empty h_prev_time, c_prev_time')
                            else:
                                if len(h_prev_time) != self.num_level or len(c_prev_time) != self.num_level:
                                    raise Exception(f'Unmatched level number ({self.num_level}) with h_prev_time ({len(h_prev_time)}), c_prev_time ({len(c_prev_time)})')

                        # if the first layer, input is the [x], which has only one resolution(level)
                        if self.layer_index == 0:
                            if len(h_prev_layer) != 1:
                                raise Exception('Wrong number of inputs or layer index')

                        # upsample and downsample input if needed
                        for i in range(self.num_up):
                            h_us = self.up_sample(h_prev_layer[-1])
                            h_prev_layer.append(h_us)

                        for i in range(self.num_down):
                            h_ds = self.down_sample(h_prev_layer[0])
                            h_prev_layer.insert(0, h_ds)

                        # sanity check if enough levels
                        if len(h_prev_layer) != self.num_level:
                            raise Exception('Unmatched level number with input level number')

                        # concatenate different level input to construct LSTM input all_x
                        if self.num_level == 2:
                            x0 = torch.cat([h_prev_layer[0], self.down_sample(h_prev_layer[1])], axis=1)
                            x1 = torch.cat([self.up_sample(h_prev_layer[0]), h_prev_layer[1]], axis=1)
                            all_x = [x0, x1]
                        elif self.num_level == 3:
                            x0 = torch.cat([h_prev_layer[0], self.down_sample(h_prev_layer[1])], axis=1)
                            x1 = torch.cat([self.up_sample(h_prev_layer[0]), h_prev_layer[1], self.down_sample(h_prev_layer[2])], axis=1)
                            x2 = torch.cat([self.up_sample(h_prev_layer[1]), h_prev_layer[2]], axis=1)
                            all_x = [x0, x1, x2]
                        elif self.num_level == 4:
                            x0 = torch.cat([h_prev_layer[0], self.down_sample(h_prev_layer[1])], axis=1)
                            x1 = torch.cat([self.up_sample(h_prev_layer[0]), h_prev_layer[1], self.down_sample(h_prev_layer[2])], axis=1)
                            x2 = torch.cat([self.up_sample(h_prev_layer[1]), h_prev_layer[2], self.down_sample(h_prev_layer[3])], axis=1)
                            x3 = torch.cat([self.up_sample(h_prev_layer[2]), h_prev_layer[3]], axis=1)
                            all_x = [x0, x1, x2, x3]
                        elif self.num_level == 5:
                            x0 = torch.cat([h_prev_layer[0], self.down_sample(h_prev_layer[1]), self.double_down(h_prev_layer[2])], axis=1)
                            x1 = torch.cat([self.up_sample(h_prev_layer[0]), h_prev_layer[1], self.down_sample(h_prev_layer[2]), self.double_down(h_prev_layer[3])], axis=1)
                            x2 = torch.cat([self.double_up(h_prev_layer[0]), self.up_sample(h_prev_layer[1]), h_prev_layer[2], self.down_sample(h_prev_layer[3]), self.double_down(h_prev_layer[4])], axis=1)
                            x3 = torch.cat([self.double_up(h_prev_layer[1]), self.up_sample(h_prev_layer[2]), h_prev_layer[3], self.down_sample(h_prev_layer[4])], axis=1)
                            x4 = torch.cat([self.double_up(h_prev_layer[2]), self.up_sample(h_prev_layer[3]), h_prev_layer[4]], axis=1)
                            all_x = [x0, x1, x2, x3, x4]
                        else:
                            raise NotImplementedError('Memory layer currently only supports 2 to 5 levels')

                        # pass each level input into ConvLSTM
                        h_cur_layer = []
                        c_cur_layer = []
                        for i in range(self.num_level):
                            # input for current ConvLSTM
                            cur_x = all_x[i]

                            lstm_name = f'ConvLSTM_Layer{self.layer_index}_Level{i}'
                            ConvLSTM = getattr(self, lstm_name)
                            batch_size, _, height, width = cur_x.size()
                            init_h, init_c = ConvLSTM.init_hidden(batch_size=batch_size,
                                                                  hidden=self.hidden_channels,
                                                                  shape=(height, width))

                            # when first time stamp, use initialized h and c as h_prev_time and c_prev_time
                            if long_term_memory:
                                if t == 0:
                                    cur_h, cur_c = ConvLSTM(cur_x=cur_x,
                                                            prev_h=init_h,
                                                            prev_c=init_c)
                                else:
                                    cur_h, cur_c = ConvLSTM(cur_x=cur_x,
                                                            prev_h=h_prev_time[i],
                                                            prev_c=c_prev_time[i])
                            else:
                                if k == 0:
                                    cur_h, cur_c = ConvLSTM(cur_x=cur_x,
                                                            prev_h=init_h,
                                                            prev_c=init_c)
                                else:
                                    cur_h, cur_c = ConvLSTM(cur_x=cur_x,
                                                            prev_h=h_prev_time[i],
                                                            prev_c=c_prev_time[i])

                            h_cur_layer.append(cur_h)
                            c_cur_layer.append(cur_c)

                        return h_cur_layer, c_cur_layer

                # initialize memory layers
                # layer 0, 2 levels (one more down sampling)
                layer_name = f'MemoryLayer_0'
                self.num_layer += 1
                memory_layer = MemoryLayer(layer_index=0,
                                            num_level=2,
                                            num_up=0,
                                            num_down=1,
                                            input_channels=self.input_channels,
                                            hidden_channels=all_hidden_channels[0],
                                            kernel_size=3)
                setattr(self, layer_name, memory_layer)

                # layer 1, still 2 levels
                layer_name = f'MemoryLayer_1'
                self.num_layer += 1
                memory_layer = MemoryLayer(layer_index=1,
                                            num_level=2,
                                            num_up=0,
                                            num_down=0,
                                            input_channels=all_hidden_channels[0],
                                            hidden_channels=all_hidden_channels[1],
                                            kernel_size=3)
                setattr(self, layer_name, memory_layer)

                # layer 2, 3 levels (one more down sampling)
                layer_name = f'MemoryLayer_2'
                self.num_layer += 1
                memory_layer = MemoryLayer(layer_index=2,
                                            num_level=3,
                                            num_up=0,
                                            num_down=1,
                                            input_channels=all_hidden_channels[1],
                                            hidden_channels=all_hidden_channels[2],
                                            kernel_size=3)
                setattr(self, layer_name, memory_layer)

                # layer 3, still 3 levels
                layer_name = f'MemoryLayer_3'
                self.num_layer += 1
                memory_layer = MemoryLayer(layer_index=3,
                                            num_level=3,
                                            num_up=0,
                                            num_down=0,
                                            input_channels=all_hidden_channels[2],
                                            hidden_channels=all_hidden_channels[3],
                                            kernel_size=3)
                setattr(self, layer_name, memory_layer)

                # layer 4, 4 levels (one more down sampling)
                # # increase one level by downsampling
                layer_name = f'MemoryLayer_4'
                self.num_layer += 1
                memory_layer = MemoryLayer(layer_index=4,
                                           num_level=4,
                                           num_up=0,
                                           num_down=1,
                                           input_channels=all_hidden_channels[3],
                                           hidden_channels=all_hidden_channels[4],
                                           kernel_size=3)
                setattr(self, layer_name, memory_layer)

                # layer 5, still 4 levels
                # # increase one level by downsampling
                layer_name = f'MemoryLayer_5'
                self.num_layer += 1
                memory_layer = MemoryLayer(layer_index=5,
                                           num_level=5,
                                           num_up=0,
                                           num_down=1,
                                           input_channels=all_hidden_channels[4],
                                           hidden_channels=all_hidden_channels[5],
                                           kernel_size=3)
                setattr(self, layer_name, memory_layer)


            def forward(self, t, k, x, h_prev_time, c_prev_time, long_term_memory):

                if t == 0 and k == 0:
                    if h_prev_time != [] or c_prev_time != []:
                        raise Exception(f'For t = {t}, h_prev_time and c_prev_time should have been set to [].')

                # current time stamp's all layers' state outputs
                h_cur_time = []
                c_cur_time = []
                for i in range(self.num_layer):
                    layer_name = f'MemoryLayer_{i}'
                    memory_layer = getattr(self, layer_name)
                    # h_final and c_final are lists with h and c for each level
                    if i == 0:
                        # first layer takes the input as prev_layer h
                        h_prev_layer = [x]

                    # long-term memory mode requres only initialization at the very beginning
                    if long_term_memory:
                        if t == 0 and k == 0:
                            h_cur_layer, c_cur_layer = memory_layer(t, k, h_prev_layer, [], [], long_term_memory)
                        else:
                            h_cur_layer, c_cur_layer = memory_layer(t, k, h_prev_layer, h_prev_time[i], c_prev_time[i], long_term_memory)
                    else:
                        if k == 0:
                            h_cur_layer, c_cur_layer = memory_layer(t, k, h_prev_layer, [], [], long_term_memory)
                        else:
                            h_cur_layer, c_cur_layer = memory_layer(t, k, h_prev_layer, h_prev_time[i], c_prev_time[i], long_term_memory)

                    h_prev_layer = h_cur_layer.copy()

                    # all_cur_h and all_cur_c contains all the layer outputs of this time stamp
                    h_cur_time.append(h_cur_layer)
                    c_cur_time.append(c_cur_layer)

                return h_cur_time, c_cur_time

        # Estimate flow in a coarse-to-fine fashion
        class EstimateFlow(torch.nn.Module):
            # in_channels is the hidden states from memory network
            # feat_channels is the feature_net output channel
            # out_channels is the target dim
            def __init__(self, num_levels, in_channels, feat_channels, out_channels):
                super(EstimateFlow, self).__init__()

                self.num_levels = num_levels

                self.in_channels = in_channels
                self.feat_channels = feat_channels
                self.out_channels = out_channels

                # Flow estimation CNN of a certain level
                class LevelFlow(torch.nn.Module):
                    def __init__(self, level, in_channels, feat_channels, out_channels):
                        super(LevelFlow, self).__init__()

                        self.level = level
                        self.in_channels = in_channels
                        self.feat_channels = feat_channels
                        self.out_channels = out_channels

                        # flow upsampling
                        self.upsample_flow = torch.nn.ConvTranspose2d(in_channels=self.out_channels,
                                                                      out_channels=self.out_channels,
                                                                      kernel_size=4,
                                                                      stride=2,
                                                                      padding=1,
                                                                      bias=False,
                                                                      groups=2)

                        # feature net extract more information from catenated input (hidden state, previous level flow)
                        if level == 0:
                            self.feature_net = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=self.in_channels,
                                                out_channels=self.feat_channels,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0),
                                torch.nn.BatchNorm2d(num_features=self.feat_channels),
                                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                            )
                        else:
                            self.feature_net = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=self.in_channels+self.out_channels,
                                                out_channels=self.feat_channels,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0),
                                torch.nn.BatchNorm2d(num_features=self.feat_channels),
                                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                            )

                        # estimate cur-level flow
                        self.flow_prediction_cnn = torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=self.feat_channels,
                                            out_channels=64,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1),
                            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                            torch.nn.Conv2d(in_channels=64,
                                            out_channels=32,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1),
                            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                            torch.nn.Conv2d(in_channels=32,
                                            out_channels=self.out_channels,
                                            kernel_size=[ 5, 5, 3, 3, 3 ][level],
                                            stride=1,
                                            padding=[ 2, 2, 1, 1, 1 ][level])
                        )


                    def forward(self, x, prev_pred_flow):
                        # when not the first level, concatenate
                        if self.level != 0:
                            prev_pred_flow = self.upsample_flow(prev_pred_flow)
                            x = torch.cat([prev_pred_flow, x], axis=1)

                        # feature net
                        x = self.feature_net(x)

                        # put feature net result into flow_prediction_cnn
                        delta_um = self.flow_prediction_cnn(x)

                        if prev_pred_flow != None:
                            return prev_pred_flow + delta_um
                        else:
                            return delta_um

                for i in range(self.num_levels):
                    name = f'LevelFlow_{i}'
                    level_flow = LevelFlow(level=i,
                                           in_channels=self.in_channels,
                                           feat_channels=self.feat_channels,
                                           out_channels=self.out_channels)

                    setattr(self, name, level_flow)

                # down sample flow
                # (256-1+0)/2 + 1 = 128
                self.downsample_flow = torch.nn.Conv2d(in_channels=self.out_channels,
                                                        out_channels=self.out_channels,
                                                        kernel_size=1,
                                                        stride=2,
                                                        padding=0)


            def forward(self, x):
                prev_pred_flow = None
                for i in range(self.num_levels):
                    name = f'LevelFlow_{i}'
                    level_flow = getattr(self, name)

                    pred_flow = level_flow(x[i], prev_pred_flow)

                    prev_pred_flow = pred_flow

                # in the end, 256x256 flow was obtained needs to downsample
                # pred_flow = self.downsample_flow(pred_flow)

                return pred_flow

        # memory network
        all_hidden_channels = [16, 16, 32, 32, 64, 128]
        self.memory_network = MemoryNetwork(self.num_channels, all_hidden_channels)

        # flow estimation (don't use the last level when multi-frame)
        # override when using image-pair
        # if self.time_span == 2:
        #     self.estimate_flow = EstimateFlow(num_levels=5,
        #                                   in_channels=all_hidden_channels[-1],
        #                                   feat_channels=256,
        #                                   out_channels=self.target_dim)
        # else:
        self.estimate_flow = EstimateFlow(num_levels=4,
                                        in_channels=all_hidden_channels[-1]+all_hidden_channels[-1]//2,
                                        feat_channels=256,
                                        out_channels=self.target_dim)




    def forward(self, t, x, h_prev_time, c_prev_time, long_term_memory):
        # split x into time_span number of pieces in channel dimension
        x = torch.split(tensor=x,
                        split_size_or_sections=self.num_channels,
                        dim=1)

        # suppose first chunk is [0, 8], then next chunk is [1, 9]
        # so when processing the second chunk, we want the hidden state memory stage
        # after 0 in the first chunk
        h_cur_chunk = None
        c_cur_chunk = None

        for k in range(self.time_span):
            cur_x = x[k]
            h_cur_time, c_cur_time = self.memory_network(t, k, cur_x, h_prev_time, c_prev_time, long_term_memory)

            h_prev_time = h_cur_time
            c_prev_time = c_cur_time

            if k == 0:
                h_cur_chunk = h_cur_time.copy()
                c_cur_chunk = c_cur_time.copy()

        # final time stamp's h and c are used for flow prediction
        final_h = h_cur_time[-1]

        # append positional encoding to all the final hidden states
        def PE_2d(batch_size, d_model, height, width):
            """
            :param batch_size: tensor batch size
            :param d_model: dimension of the model
            :param height: height of the positions
            :param width: width of the positions
            :return: d_model*height*width position matrix
            """
            if d_model % 4 != 0:
                raise ValueError(f'Cannot use sin/cos positional encoding with odd dimension {d_model}')

            pe = torch.zeros(batch_size, d_model, height, width)
            # Each dimension use half of d_model
            d_model = int(d_model / 2)
            div_term = torch.exp(torch.arange(0., d_model, 2) *
                                -(math.log(10000.0) / d_model))
            pos_w = torch.arange(0., width).unsqueeze(1)
            pos_h = torch.arange(0., height).unsqueeze(1)
            pe[:, 0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
            pe[:, 1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
            pe[:, d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
            pe[:, d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

            return pe

        for i in range(len(final_h)):
            cur_h = final_h[i]
            batch_size = cur_h.shape[0]
            d_model = cur_h.shape[1] // 2
            height = cur_h.shape[2]
            width = cur_h.shape[3]

            cur_h_pe = PE_2d(batch_size, d_model, height, width).to(self.device)
            final_h[i] = torch.cat((final_h[i], cur_h_pe), dim=1)

        # flow estimation
        pred_flow = self.estimate_flow(final_h)

        # return pred_flow, h_cur_time, c_cur_time
        return pred_flow, h_cur_chunk, c_cur_chunk


# Memory-PIVnet that outputs same resolution outputs as inputs (no input neighboring padding)
class Memory_PIVnet_No_Neighbor(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Memory_PIVnet_No_Neighbor, self).__init__()

        # date channel
        self.num_channels = kwargs['num_channels']
        # number of time frames (including the current frame)
        self.time_span = kwargs['time_span']
        # target (flow) dimension
        self.target_dim = kwargs['target_dim']

        # Multigrid Memory Network
        class MemoryNetwork(torch.nn.Module):

            # input_channels: number of input data channels
            # all_hidden_channels: list of channel number for every memory layer
            def __init__(self, input_channels, all_hidden_channels):
                super(MemoryNetwork, self).__init__()

                self.num_layer = 0
                self.input_channels = input_channels
                self.all_hidden_channels = all_hidden_channels

                # Multigrid Memory layer with batch normalization
                class MemoryLayer(torch.nn.Module):
                    # layer_index: the memory layer index in the memory network
                    # num_level: levels of different resolutions
                    # num_up: number of upsampling needed
                    # num_down: number of downsampling needed
                    # input_channels: number of input data channels
                    # kernel_size: conv kernel for conv lstm
                    def __init__(self, layer_index, num_level, num_up, num_down, input_channels, hidden_channels, kernel_size):
                        super(MemoryLayer, self).__init__()

                        # number of different resolutions
                        self.layer_index = layer_index
                        self.num_level = num_level
                        self.num_up = num_up
                        self.num_down = num_down
                        self.input_channels = input_channels
                        self.hidden_channels = hidden_channels
                        self.kernel_size = kernel_size

                        # ConvLSTM cell
                        class ConvLSTMCell(torch.nn.Module):
                            def __init__(self, input_channels, hidden_channels, kernel_size):
                                super(ConvLSTMCell, self).__init__()

                                assert hidden_channels % 2 == 0

                                self.input_channels = input_channels
                                self.hidden_channels = hidden_channels
                                self.kernel_size = kernel_size

                                # To ensure that the states have the same number of rows
                                # and same number of columns as the inputs, padding is required before convolution
                                self.padding = (kernel_size - 1) // 2

                                # input gate
                                self.Wxi = torch.nn.Conv2d(in_channels=self.input_channels,
                                                            out_channels=self.hidden_channels,
                                                            kernel_size=self.kernel_size,
                                                            stride=1,
                                                            padding=self.padding,
                                                            bias=True)
                                self.Whi = torch.nn.Conv2d(in_channels=self.hidden_channels,
                                                            out_channels=self.hidden_channels,
                                                            kernel_size=self.kernel_size,
                                                            stride=1,
                                                            padding=self.padding,
                                                            bias=False)

                                self.WxiBN = torch.nn.Sequential(torch.nn.Conv2d(in_channels=self.input_channels,
                                                                                 out_channels=self.hidden_channels,
                                                                                 kernel_size=self.kernel_size,
                                                                                 stride=1,
                                                                                 padding=self.padding,
                                                                                 bias=True),
                                                                 torch.nn.BatchNorm2d(num_features=self.hidden_channels))
                                self.WhiBN = torch.nn.Sequential(torch.nn.Conv2d(in_channels=self.hidden_channels,
                                                                                 out_channels=self.hidden_channels,
                                                                                 kernel_size=self.kernel_size,
                                                                                 stride=1,
                                                                                 padding=self.padding,
                                                                                 bias=False),
                                                                 torch.nn.BatchNorm2d(num_features=self.hidden_channels))

                                # forget gate
                                self.Wxf = torch.nn.Conv2d(in_channels=self.input_channels,
                                                            out_channels=self.hidden_channels,
                                                            kernel_size=self.kernel_size,
                                                            stride=1,
                                                            padding=self.padding,
                                                            bias=True)
                                self.Whf = torch.nn.Conv2d(in_channels=self.hidden_channels,
                                                            out_channels=self.hidden_channels,
                                                            kernel_size=self.kernel_size,
                                                            stride=1,
                                                            padding=self.padding,
                                                            bias=False)

                                self.WxfBN = torch.nn.Sequential(torch.nn.Conv2d(in_channels=self.input_channels,
                                                                                out_channels=self.hidden_channels,
                                                                                kernel_size=self.kernel_size,
                                                                                stride=1,
                                                                                padding=self.padding,
                                                                                bias=True),
                                                                torch.nn.BatchNorm2d(num_features=self.hidden_channels))
                                self.WhfBN = torch.nn.Sequential(torch.nn.Conv2d(in_channels=self.hidden_channels,
                                                                                out_channels=self.hidden_channels,
                                                                                kernel_size=self.kernel_size,
                                                                                stride=1,
                                                                                padding=self.padding,
                                                                                bias=False),
                                                                torch.nn.BatchNorm2d(num_features=self.hidden_channels))

                                # candidate
                                self.Wxc = torch.nn.Conv2d(in_channels=self.input_channels,
                                                            out_channels=self.hidden_channels,
                                                            kernel_size=self.kernel_size,
                                                            stride=1,
                                                            padding=self.padding,
                                                            bias=True)
                                self.Whc = torch.nn.Conv2d(in_channels=self.hidden_channels,
                                                            out_channels=self.hidden_channels,
                                                            kernel_size=self.kernel_size,
                                                            stride=1,
                                                            padding=self.padding,
                                                            bias=False)

                                self.WxcBN = torch.nn.Sequential(torch.nn.Conv2d(in_channels=self.input_channels,
                                                                                out_channels=self.hidden_channels,
                                                                                kernel_size=self.kernel_size,
                                                                                stride=1,
                                                                                padding=self.padding,
                                                                                bias=True),
                                                                torch.nn.BatchNorm2d(num_features=self.hidden_channels))
                                self.WhcBN = torch.nn.Sequential(torch.nn.Conv2d(in_channels=self.hidden_channels,
                                                                                out_channels=self.hidden_channels,
                                                                                kernel_size=self.kernel_size,
                                                                                stride=1,
                                                                                padding=self.padding,
                                                                                bias=False),
                                                                torch.nn.BatchNorm2d(num_features=self.hidden_channels))

                                # output
                                self.Wxo = torch.nn.Conv2d(in_channels=self.input_channels,
                                                            out_channels=self.hidden_channels,
                                                            kernel_size=self.kernel_size,
                                                            stride=1,
                                                            padding=self.padding,
                                                            bias=True)
                                self.Who = torch.nn.Conv2d(in_channels=self.hidden_channels,
                                                            out_channels=self.hidden_channels,
                                                            kernel_size=self.kernel_size,
                                                            stride=1,
                                                            padding=self.padding,
                                                            bias=False)

                                self.WxoBN = torch.nn.Sequential(torch.nn.Conv2d(in_channels=self.input_channels,
                                                                                out_channels=self.hidden_channels,
                                                                                kernel_size=self.kernel_size,
                                                                                stride=1,
                                                                                padding=self.padding,
                                                                                bias=True),
                                                                torch.nn.BatchNorm2d(num_features=self.hidden_channels))
                                self.WhoBN = torch.nn.Sequential(torch.nn.Conv2d(in_channels=self.hidden_channels,
                                                                                out_channels=self.hidden_channels,
                                                                                kernel_size=self.kernel_size,
                                                                                stride=1,
                                                                                padding=self.padding,
                                                                                bias=False),
                                                                torch.nn.BatchNorm2d(num_features=self.hidden_channels))

                                self.Wci = None
                                self.Wcf = None
                                self.Wco = None

                                # batch normalization layer for the final hidden state output
                                self.BN = torch.nn.BatchNorm2d(num_features=self.hidden_channels)

                            # prev_h is the previous timestamp hidden state
                            # prev_c is the previous timestamp candidate value
                            def forward(self, cur_x, prev_h, prev_c):
                                # current input gate result
                                # i_t = sigmoid(Wxi * x_t + Whi * h_{t-1} + Wci x c_{t-1} + b_i)
                                cur_i = torch.sigmoid(self.WxiBN(cur_x) + self.WhiBN(prev_h) + self.Wci * prev_c)

                                # current forget gate result
                                # f_t = sigmoid(Wxf * x_t + Whf * h_{t-1} + Wcf x c_{t-1} + b_f)
                                cur_f = torch.sigmoid(self.WxfBN(cur_x) + self.WhfBN(prev_h) + self.Wcf * prev_c)

                                # current candidate value (cell state)
                                # c_t = f_t x c_{t-1} + i_t x tanh(Wxc * x_t + Whc * h_{t-1} + b_c)
                                cur_c = cur_f * prev_c + cur_i * torch.tanh(self.WxcBN(cur_x) + self.WhcBN(prev_h))

                                # current cell output
                                # o_t = sigmoid(Wxo * x_t + Who * h_{t-1} + Wco x c_t + b_o)
                                cur_o = torch.sigmoid(self.WxoBN(cur_x) + self.WhoBN(prev_h) + self.Wco * cur_c)

                                # current hidden state output
                                # h_t = o_t x tanh(c_t)
                                cur_h = cur_o * torch.tanh(self.BN(cur_c))

                                # return the hidden and cell state
                                return cur_h, cur_c

                            def init_hidden(self, batch_size, hidden, shape):
                                if self.Wci is None:
                                    self.Wci = torch.autograd.Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
                                    self.Wcf = torch.autograd.Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
                                    self.Wco = torch.autograd.Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
                                else:
                                    assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
                                    assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'

                                return (torch.autograd.Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                                        torch.autograd.Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())

                        # downsample and upsample
                        self.down_sample = torch.nn.AvgPool2d(kernel_size=2,
                                                              stride=2,
                                                              padding=0)

                        self.double_down = torch.nn.Sequential(
                            torch.nn.AvgPool2d(kernel_size=2,
                                               stride=2,
                                               padding=0),
                            torch.nn.AvgPool2d(kernel_size=2,
                                               stride=2,
                                               padding=0)
                        )

                        # deconv used to upsample to double
                        self.up_sample = torch.nn.ConvTranspose2d(in_channels=self.input_channels,
                                                                  out_channels=self.input_channels,
                                                                  kernel_size=2,
                                                                  stride=2)

                        self.double_up = torch.nn.Sequential(
                            torch.nn.ConvTranspose2d(in_channels=self.input_channels,
                                                    out_channels=self.input_channels,
                                                    kernel_size=2,
                                                    stride=2),
                            torch.nn.ConvTranspose2d(in_channels=self.input_channels,
                                                    out_channels=self.input_channels,
                                                    kernel_size=2,
                                                    stride=2)
                        )

                        # a ConvLSTM on each level within the layer
                        for i in range(num_level):
                            lstm_name = f'ConvLSTM_Layer{self.layer_index}_Level{i}'
                            # input channels number is based on levels
                            if num_level == 2:
                                if i == 0:
                                    lstm_input_channel = 2*self.input_channels
                                elif i == 1:
                                    lstm_input_channel = 2*self.input_channels
                            elif num_level == 3:
                                if i == 0:
                                    lstm_input_channel = 2*self.input_channels
                                elif i == 1:
                                    lstm_input_channel = 3*self.input_channels
                                elif i == 2:
                                    lstm_input_channel = 2*self.input_channels
                            elif num_level == 4:
                                if i == 0:
                                    lstm_input_channel = 2*self.input_channels
                                elif i == 1:
                                    lstm_input_channel = 3*self.input_channels
                                elif i == 2:
                                    lstm_input_channel = 3*self.input_channels
                                elif i == 3:
                                    lstm_input_channel = 2*self.input_channels
                            elif num_level == 5:
                                if i == 0:
                                    lstm_input_channel = 3*self.input_channels
                                elif i == 1:
                                    lstm_input_channel = 4*self.input_channels
                                elif i == 2:
                                    lstm_input_channel = 5*self.input_channels
                                elif i == 3:
                                    lstm_input_channel = 4*self.input_channels
                                elif i == 4:
                                    lstm_input_channel = 3*self.input_channels

                            lstm = ConvLSTMCell(input_channels=lstm_input_channel,
                                                hidden_channels=self.hidden_channels,
                                                kernel_size=self.kernel_size)

                            setattr(self, lstm_name, lstm)

                    # h_prev_layer: h from the previous memory layer of the CURRENT time stamp, used as LSTM input
                    # h_prev_time/c_prev_time: h and c from the same memory layer of the PREVIOUS time stamp
                    def forward(self, t, h_prev_layer, h_prev_time, c_prev_time):
                        # time stamp sanity check
                        if t == 0:
                            if h_prev_time != [] or c_prev_time != []:
                                raise Exception('First time stamp should have empty h_prev_time, c_prev_time')
                        else:
                            if len(h_prev_time) != self.num_level or len(c_prev_time) != self.num_level:
                                raise Exception(f'Unmatched level number ({self.num_level}) with h_prev_time ({len(h_prev_time)}), c_prev_time ({len(c_prev_time)})')

                        # if the first layer, input is the [x], which has only one resolution(level)
                        if self.layer_index == 0:
                            if len(h_prev_layer) != 1:
                                raise Exception('Wrong number of inputs or layer index')

                        # upsample and downsample input if needed
                        for i in range(self.num_up):
                            h_us = self.up_sample(h_prev_layer[-1])
                            h_prev_layer.append(h_us)

                        for i in range(self.num_down):
                            h_ds = self.down_sample(h_prev_layer[0])
                            h_prev_layer.insert(0, h_ds)

                        # sanity check if enough levels
                        if len(h_prev_layer) != self.num_level:
                            raise Exception('Unmatched level number with input level number')

                        # concatenate different level input to construct LSTM input all_x
                        if self.num_level == 2:
                            x0 = torch.cat([h_prev_layer[0], self.down_sample(h_prev_layer[1])], axis=1)
                            x1 = torch.cat([self.up_sample(h_prev_layer[0]), h_prev_layer[1]], axis=1)
                            all_x = [x0, x1]
                        elif self.num_level == 3:
                            x0 = torch.cat([h_prev_layer[0], self.down_sample(h_prev_layer[1])], axis=1)
                            x1 = torch.cat([self.up_sample(h_prev_layer[0]), h_prev_layer[1], self.down_sample(h_prev_layer[2])], axis=1)
                            x2 = torch.cat([self.up_sample(h_prev_layer[1]), h_prev_layer[2]], axis=1)
                            all_x = [x0, x1, x2]
                        elif self.num_level == 4:
                            x0 = torch.cat([h_prev_layer[0], self.down_sample(h_prev_layer[1])], axis=1)
                            x1 = torch.cat([self.up_sample(h_prev_layer[0]), h_prev_layer[1], self.down_sample(h_prev_layer[2])], axis=1)
                            x2 = torch.cat([self.up_sample(h_prev_layer[1]), h_prev_layer[2], self.down_sample(h_prev_layer[3])], axis=1)
                            x3 = torch.cat([self.up_sample(h_prev_layer[2]), h_prev_layer[3]], axis=1)
                            all_x = [x0, x1, x2, x3]
                        elif self.num_level == 5:
                            x0 = torch.cat([h_prev_layer[0], self.down_sample(h_prev_layer[1]), self.double_down(h_prev_layer[2])], axis=1)
                            x1 = torch.cat([self.up_sample(h_prev_layer[0]), h_prev_layer[1], self.down_sample(h_prev_layer[2]), self.double_down(h_prev_layer[3])], axis=1)
                            x2 = torch.cat([self.double_up(h_prev_layer[0]), self.up_sample(h_prev_layer[1]), h_prev_layer[2], self.down_sample(h_prev_layer[3]), self.double_down(h_prev_layer[4])], axis=1)
                            x3 = torch.cat([self.double_up(h_prev_layer[1]), self.up_sample(h_prev_layer[2]), h_prev_layer[3], self.down_sample(h_prev_layer[4])], axis=1)
                            x4 = torch.cat([self.double_up(h_prev_layer[2]), self.up_sample(h_prev_layer[3]), h_prev_layer[4]], axis=1)
                            all_x = [x0, x1, x2, x3, x4]
                        else:
                            raise NotImplementedError('Memory layer currently only supports 2 to 5 levels')

                        # pass each level input into ConvLSTM
                        h_cur_layer = []
                        c_cur_layer = []
                        for i in range(self.num_level):
                            # input for current ConvLSTM
                            cur_x = all_x[i]

                            lstm_name = f'ConvLSTM_Layer{self.layer_index}_Level{i}'
                            ConvLSTM = getattr(self, lstm_name)
                            batch_size, _, height, width = cur_x.size()
                            init_h, init_c = ConvLSTM.init_hidden(batch_size=batch_size,
                                                                  hidden=self.hidden_channels,
                                                                  shape=(height, width))

                            # when first time stamp, use initialized h and c as h_prev_time and c_prev_time
                            if t == 0:
                                cur_h, cur_c = ConvLSTM(cur_x=cur_x,
                                                        prev_h=init_h,
                                                        prev_c=init_c)
                            else:
                                cur_h, cur_c = ConvLSTM(cur_x=cur_x,
                                                        prev_h=h_prev_time[i],
                                                        prev_c=c_prev_time[i])

                            h_cur_layer.append(cur_h)
                            c_cur_layer.append(cur_c)

                        return h_cur_layer, c_cur_layer

                # initialize memory layers
                # layer 0, 2 levels (one more down sampling)
                layer_name = f'MemoryLayer_0'
                self.num_layer += 1
                memory_layer = MemoryLayer(layer_index=0,
                                            num_level=2,
                                            num_up=0,
                                            num_down=1,
                                            input_channels=self.input_channels,
                                            hidden_channels=all_hidden_channels[0],
                                            kernel_size=3)
                setattr(self, layer_name, memory_layer)

                # layer 1, still 2 levels
                layer_name = f'MemoryLayer_1'
                self.num_layer += 1
                memory_layer = MemoryLayer(layer_index=1,
                                            num_level=2,
                                            num_up=0,
                                            num_down=0,
                                            input_channels=all_hidden_channels[0],
                                            hidden_channels=all_hidden_channels[1],
                                            kernel_size=3)
                setattr(self, layer_name, memory_layer)

                # layer 2, 3 levels (one more down sampling)
                layer_name = f'MemoryLayer_2'
                self.num_layer += 1
                memory_layer = MemoryLayer(layer_index=2,
                                            num_level=3,
                                            num_up=0,
                                            num_down=1,
                                            input_channels=all_hidden_channels[1],
                                            hidden_channels=all_hidden_channels[2],
                                            kernel_size=3)
                setattr(self, layer_name, memory_layer)

                # layer 3, still 3 levels
                layer_name = f'MemoryLayer_3'
                self.num_layer += 1
                memory_layer = MemoryLayer(layer_index=3,
                                            num_level=3,
                                            num_up=0,
                                            num_down=0,
                                            input_channels=all_hidden_channels[2],
                                            hidden_channels=all_hidden_channels[3],
                                            kernel_size=3)
                setattr(self, layer_name, memory_layer)

                # layer 4, 4 levels (one more down sampling)
                # # increase one level by downsampling
                layer_name = f'MemoryLayer_4'
                self.num_layer += 1
                memory_layer = MemoryLayer(layer_index=4,
                                           num_level=4,
                                           num_up=0,
                                           num_down=1,
                                           input_channels=all_hidden_channels[3],
                                           hidden_channels=all_hidden_channels[4],
                                           kernel_size=3)
                setattr(self, layer_name, memory_layer)

                # layer 5, still 4 levels
                # # increase one level by downsampling
                layer_name = f'MemoryLayer_5'
                self.num_layer += 1
                memory_layer = MemoryLayer(layer_index=5,
                                           num_level=5,
                                           num_up=0,
                                           num_down=1,
                                           input_channels=all_hidden_channels[4],
                                           hidden_channels=all_hidden_channels[5],
                                           kernel_size=3)
                setattr(self, layer_name, memory_layer)


            def forward(self, t, x, h_prev_time, c_prev_time):

                if t == 0:
                    if h_prev_time != [] or c_prev_time != []:
                        raise Exception('For the first time stamp, h_prev_time and c_prev_time should have been set to None.')

                # current time stamp's all layers' state outputs
                h_cur_time = []
                c_cur_time = []
                for i in range(self.num_layer):
                    layer_name = f'MemoryLayer_{i}'
                    memory_layer = getattr(self, layer_name)
                    # h_final and c_final are lists with h and c for each level
                    if i == 0:
                        # first layer takes the input as prev_layer h
                        h_prev_layer = [x]

                    if t == 0:
                        h_cur_layer, c_cur_layer = memory_layer(t, h_prev_layer, [], [])
                    else:
                        h_cur_layer, c_cur_layer = memory_layer(t, h_prev_layer, h_prev_time[i], c_prev_time[i])

                    h_prev_layer = h_cur_layer.copy()

                    # all_cur_h and all_cur_c contains all the layer outputs of this time stamp
                    h_cur_time.append(h_cur_layer)
                    c_cur_time.append(c_cur_layer)

                return h_cur_time, c_cur_time

        # Estimate flow in a coarse-to-fine fashion
        class EstimateFlow(torch.nn.Module):
            # in_channels is the hidden states from memory network
            # feat_channels is the feature_net output channel
            # out_channels is the target dim
            def __init__(self, num_levels, in_channels, feat_channels, out_channels):
                super(EstimateFlow, self).__init__()

                self.num_levels = num_levels

                self.in_channels = in_channels
                self.feat_channels = feat_channels
                self.out_channels = out_channels

                # Flow estimation CNN of a certain level
                class LevelFlow(torch.nn.Module):
                    def __init__(self, level, in_channels, feat_channels, out_channels):
                        super(LevelFlow, self).__init__()

                        self.level = level
                        self.in_channels = in_channels
                        self.feat_channels = feat_channels
                        self.out_channels = out_channels

                        # flow upsampling
                        self.upsample_flow = torch.nn.ConvTranspose2d(in_channels=self.out_channels,
                                                                      out_channels=self.out_channels,
                                                                      kernel_size=4,
                                                                      stride=2,
                                                                      padding=1,
                                                                      bias=False,
                                                                      groups=2)

                        # feature net extract more information from catenated input (hidden state, previous level flow)
                        if level == 0:
                            self.feature_net = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=self.in_channels,
                                                out_channels=self.feat_channels,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0),
                                torch.nn.BatchNorm2d(num_features=self.feat_channels),
                                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                            )
                        else:
                            self.feature_net = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=self.in_channels+self.out_channels,
                                                out_channels=self.feat_channels,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0),
                                torch.nn.BatchNorm2d(num_features=self.feat_channels),
                                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                            )

                        # estimate cur-level flow
                        self.flow_prediction_cnn = torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=self.feat_channels,
                                            out_channels=64,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1),
                            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                            torch.nn.Conv2d(in_channels=64,
                                            out_channels=32,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1),
                            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                            torch.nn.Conv2d(in_channels=32,
                                            out_channels=self.out_channels,
                                            kernel_size=[ 5, 5, 3, 3, 3 ][level],
                                            stride=1,
                                            padding=[ 2, 2, 1, 1, 1 ][level])
                        )


                    def forward(self, x, prev_pred_flow):
                        # when not the first level, concatenate
                        if self.level != 0:
                            prev_pred_flow = self.upsample_flow(prev_pred_flow)
                            x = torch.cat([prev_pred_flow, x], axis=1)

                        # feature net
                        x = self.feature_net(x)

                        # put feature net result into flow_prediction_cnn
                        delta_um = self.flow_prediction_cnn(x)

                        if prev_pred_flow != None:
                            return prev_pred_flow + delta_um
                        else:
                            return delta_um

                for i in range(self.num_levels):
                    name = f'LevelFlow_{i}'
                    level_flow = LevelFlow(level=i,
                                           in_channels=self.in_channels,
                                           feat_channels=self.feat_channels,
                                           out_channels=self.out_channels)

                    setattr(self, name, level_flow)

                # down sample flow
                # (256-1+0)/2 + 1 = 128
                self.downsample_flow = torch.nn.Conv2d(in_channels=self.out_channels,
                                                        out_channels=self.out_channels,
                                                        kernel_size=1,
                                                        stride=2,
                                                        padding=0)


            def forward(self, x):
                prev_pred_flow = None
                for i in range(self.num_levels):
                    name = f'LevelFlow_{i}'
                    level_flow = getattr(self, name)

                    pred_flow = level_flow(x[i], prev_pred_flow)

                    prev_pred_flow = pred_flow.clone()

                # in the end, 256x256 flow was obtained needs to downsample
                # pred_flow = self.downsample_flow(pred_flow)

                return pred_flow

        # memory network
        all_hidden_channels = [16, 16, 32, 32, 64, 128]
        self.memory_network = MemoryNetwork(self.num_channels, all_hidden_channels)


        self.estimate_flow = EstimateFlow(num_levels=5,
                                        in_channels=all_hidden_channels[-1],
                                        feat_channels=256,
                                        out_channels=self.target_dim)


    def forward(self, x):
        # split x into time_span number of pieces in channel dimension
        x = torch.split(tensor=x,
                        split_size_or_sections=self.num_channels,
                        dim=1)

        # initialize
        h_prev_time = []
        c_prev_time = []
        for t in range(self.time_span):
            cur_x = x[t]
            h_cur_time, c_cur_time = self.memory_network(t, cur_x, h_prev_time, c_prev_time)

            h_prev_time = h_cur_time.copy()
            c_prev_time = c_cur_time.copy()

        # final time stamp's h and c are used for flow prediction
        final_h = h_cur_time[-1]

        # flow estimation
        pred_flow = self.estimate_flow(final_h)

        # print("\tIn Model: input size", x.shape,
        #       "output size", pred_flow.shape)

        return pred_flow

