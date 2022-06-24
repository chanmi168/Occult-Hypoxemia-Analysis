import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
import numpy as np

import scipy
from scipy.fftpack import fft, ifft

import sys

import sys
sys.path.append('../') # add this line so Data and data are visible in this file
from setting import *

class MyAvgPool1dPadSame(nn.Module):
    """
    extend nn.AvgPool1d to support SAME padding
    """
    def __init__(self, kernel_size, stride):
        super(MyAvgPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.avg_pool = torch.nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.stride)
    def forward(self, x):
        net = x
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left

        net = F.pad(net, (pad_left, pad_right), "reflect")

        net = self.avg_pool(net)

        return net
    
class InceptionBlock(nn.Module):

    def __init__(self,training_params, stride=2, input_channel=None, output_channel=None, is_conv=True, outsize=None):
        """
        Inputs:
            c_in - Number of input feature maps from the previous layers
            c_red - Dictionary with keys "3x3" and "5x5" specifying the output of the dimensionality reducing 1x1 convolutions
            c_out - Dictionary with keys "1x1", "3x3", "5x5", and "max"
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()
        
        input_dim = training_params['data_dimensions'][1]
        if input_channel is None:
            input_channel = training_params['data_dimensions'][0]
        if output_channel is None:
            output_channel = training_params['data_dimensions'][0]
            
        self.input_channel = input_channel
        self.output_channel = output_channel
        
        k1 = training_params['kernels']['k1']
        k2 = training_params['kernels']['k2']
        k3 = training_params['kernels']['k3']
        
        # 1x1 convolution branch
        self.conv_k1x1 = nn.Sequential(
            MyAvgPool1dPadSame(kernel_size=k1, stride=stride) if is_conv else torch.nn.Upsample(size=(outsize), mode='linear', align_corners=True),
            nn.Conv1d(self.input_channel, self.output_channel, kernel_size=k1, stride=1, padding=math.floor(k1/2)),
            nn.BatchNorm1d(self.output_channel),
            nn.ReLU(),
            # MyAvgPool1dPadSame(kernel_size=k1, stride=stride) if is_conv else torch.nn.Upsample(size=(outsize), mode='linear', align_corners=True)
        )
        # 3x3 convolution branch
        self.conv_k2x1 = nn.Sequential(
            MyAvgPool1dPadSame(kernel_size=k2, stride=stride) if is_conv else torch.nn.Upsample(size=(outsize), mode='linear', align_corners=True),
            nn.Conv1d(self.input_channel, self.output_channel, kernel_size=k2, stride=1, padding=math.floor(k2/2)),
            nn.BatchNorm1d(self.output_channel),
            nn.ReLU(),
            # MyAvgPool1dPadSame(kernel_size=k2, stride=stride),
        )
# up = nn.Upsample(size=(24, 24))

        # 5x5 convolution branch
        self.conv_k3x1 = nn.Sequential(
            MyAvgPool1dPadSame(kernel_size=k3, stride=stride) if is_conv else torch.nn.Upsample(size=(outsize), mode='linear', align_corners=True),
            nn.Conv1d(self.input_channel, self.output_channel, kernel_size=k3, stride=1, padding=math.floor(k3/2)),
            nn.BatchNorm1d(self.output_channel),
            nn.ReLU(),
            # MyAvgPool1dPadSame(kernel_size=k3, stride=stride),
        )
        
        self.ch_pooling = torch.nn.Conv1d(self.output_channel*3, self.output_channel, 1) # out_channels->1 channel, kernel size = 1

    def forward(self, x):
        x_k1x1 = self.conv_k1x1(x)
        x_k2x1 = self.conv_k2x1(x)
        x_k3x1 = self.conv_k3x1(x)

        x_out = torch.cat([x_k1x1, x_k2x1, x_k3x1], dim=1)
        x_out = self.ch_pooling(x_out)
        
        return x_out
    
    

    

class Encoder(nn.Module):
    def __init__(self, training_params=None, input_channel=None):
        super(Encoder, self).__init__()
        
        input_dim = training_params['data_dimensions'][1]
        if input_channel is None:
            input_channel = training_params['data_dimensions'][0]
            
        channel_n = training_params['channel_n']
        stride = training_params['stride']
        
        self.model_name = training_params['model_name']
        self.verbose = training_params['verbose']
        
        self.n_block = training_params['n_block']
        in_ch = input_channel
        out_ch = channel_n
        self.basicblock_list = nn.ModuleList()
        
        self.input_channels = []
        self.output_channels = []
        for i_block in range(self.n_block):
            self.basicblock_list.append(InceptionBlock(training_params, stride=stride, input_channel=in_ch, output_channel=out_ch))
            self.input_channels.append(in_ch)
            in_ch = out_ch
            self.output_channels.append(out_ch)
            out_ch = out_ch*2

        last_layer_dim = input_dim
    
        encoder_layer_dims = []
        encoder_layer_dims.append(last_layer_dim)
        for i_block in range(self.n_block):
            # last_layer_dim = round(last_layer_dim/stride)
            last_layer_dim = int(np.ceil(last_layer_dim/stride))
            encoder_layer_dims.append(last_layer_dim)
            
        self.encoder_layer_dims = encoder_layer_dims
        self.last_layer_dim = last_layer_dim
        
        self.ch_pooling = torch.nn.Conv1d(self.output_channels[-1], 1, 1)
        self.input_channels.append(self.output_channels[-1])
        self.output_channels.append(1)
    
        # VAE: transform hidden features to mean and variance
        self.relu = torch.nn.ReLU()
        self.encode_mean = torch.nn.Linear(last_layer_dim*1, 25)
        self.encode_logvar = torch.nn.Linear(last_layer_dim*1, 25)

        self.feature_out_dim = self.last_layer_dim * self.output_channels[-1]

        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('feature_out_dim :', self.feature_out_dim)
        

    def forward(self, x):   
        
        out = x.float()

        for i_block in range(len(self.basicblock_list)):
            net = self.basicblock_list[i_block]
            out = net(out)
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, out dim: {3}'.format(i_block, net.input_channel, net.output_channel, out.size()))
            # print(i_block, out.size())

        out = self.ch_pooling(out)

        debug = False
        if debug == True:
            # size of x is  torch.Size([2, 1, 1000])
            print('-----------------------------')
            print('size of x is ', x.size())
#             print('size of out1 is ', out1.size())
            print('size of out is ', out.size())
            print('-----------------------------')
            sys.exit()

#         print(out.size())
        return out



class Decoder(torch.nn.Module):    
    def __init__(self, training_params=None, encoder_layer_dims=None, encoder_channels=None, input_channel=None):
        super(Decoder, self).__init__()
        
        if encoder_layer_dims is None:
            input_dim = training_params['data_dimensions'][1]
        if input_channel is None:
            input_channel = training_params['data_dimensions'][0]

        input_dim = encoder_layer_dims[-1]
        channel_n = training_params['channel_n']
        stride = training_params['stride']
        
        self.encoder_channels = encoder_channels
        self.verbose = training_params['verbose']

        self.n_block = training_params['n_block']
        in_ch = input_channel

        self.basicblock_list = nn.ModuleList()

        for i_block in range(self.n_block):
            outsize = encoder_layer_dims[self.n_block-i_block-1] # ignore the last dimension
            in_ch = self.encoder_channels[self.n_block-i_block-1] # ignore the last dimension
            if i_block==self.n_block-1:
                out_ch = 1
            else:
                out_ch = in_ch // 2

            self.basicblock_list.append(InceptionBlock(training_params, stride=stride, input_channel=in_ch, output_channel=out_ch, is_conv=False, outsize=outsize))
    
        # self.basicblock_list.append(torch.nn.Conv1d(out_ch, 1, 1)) # out_channels->1 channel, kernel size = 1

        self.ch_pooling = torch.nn.Conv1d(out_ch, 1, 1)
        
        self.channel_n = int(channel_n)
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print('feature_out_dim   :', self.feature_out_dim)

    def forward(self, x):   

        out = x.float()

        for i_block in range(len(self.basicblock_list)):
            net = self.basicblock_list[i_block]
            out = net(out)
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, out dim: {3}'.format(i_block, net.input_channel, net.output_channel, out.size()))

        out = self.ch_pooling(out)
        
        out = out.reshape(out.size(0), -1) # flatten the channel dimension -> (N_instances, N_channel x N_feature)
        # out = Flatten(out)

        debug = False
        if debug == True:
            # size of x is  torch.Size([2, 1, 1000])
            # size of out1 is  torch.Size([2, 4, 500])
            print('-----------------------------')
            print('size of x is ', x.size())
            print('size of out is ', out.size())
            print('-----------------------------')
            sys.exit()

        return out

extractor_dict = {
    # 'FeatureExtractor_CNNlight': FeatureExtractor_CNNlight
    'Encoder': Encoder
}