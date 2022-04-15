import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

import sys

class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left

        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = self.max_pool(net)

        return net
    
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, use_sc, is_first_block=False):
        super(BasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do
        self.use_sc = use_sc

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        if self.use_sc:
            out += identity

        return out
    
    

    
class ResNet1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, use_sc=True, verbose=False):
        super(ResNet1D, self).__init__()
        
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.use_sc = use_sc

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                use_sc = self.use_sc,
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels, n_classes)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        out = x
        
        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print('final pooling', out.shape)
        # out = self.do(out)
        out = self.dense(out)
        if self.verbose:
            print('dense', out.shape)
        # out = self.softmax(out)

        if self.verbose:
            print('softmax', out.shape)
        
        return out    
    
    
# # HIDDEN_DIM = 50
# KERNEL_SIZE = 5
# P_DROPOUT = 0.5
# bidirectional = False
# num_layers = 2

# class lstmnet(nn.Module):
#     r"""lstmnet is a simple recurrent neural network that contains one 
#     hidden layer of 64 nodes with 3 time steps by default. It expects 
#     an input of 3D tensor with a dimension (batch, time_step, 
#     input_size). The output will be a 2D tensor with a dimension (N, 2).
#     Args:
#         - None. The variables used for each sub-layer are hard-coded
#     Shape:
#         - Input: :math:`(batch, time_step, input_size)`
#         - Output: :math:`(batch, output_size)`
#     Examples::
#         >>> m = lstmnet()
#         >>> batchSize = 16
#         >>> featDim = 10
#         >>> timeStep = 3
#         >>> input = torch.randn(batchSize, timeStep, featDim)
#         >>> output = m(input)
#         >>> output.size()
#             (16, 2)
#     """
#     def __init__(self, inputDim=10, hiddenDim=8, outputDim=2):
#         super(lstmnet, self).__init__()
#         self.outputDim = outputDim

#         self.lstm = nn.LSTM(         # if use nn.RNN(), it hardly learns
#             input_size=inputDim,
#             hidden_size=hiddenDim,         # rnn hidden unit
#             num_layers=num_layers,           # number of rnn layer
#             batch_first=True,       # input & output will has batch size as 1st dimension. e.g. (batch, time_step, input_size)
#             bidirectional=bidirectional,
#             dropout=P_DROPOUT
#         )
        
#         self.fc1 = nn.Linear(hiddenDim, 2)
# #         self.fc1 = nn.Linear(hiddenDim*2, 2)
# #         self.fc1 = nn.Linear(hiddenDim*2, 32)
# #         self.fc2 = nn.Linear(32, outputDim)
# #         self.relu = nn.ReLU(inplace=False)
# #         self.lsm = nn.LogSoftmax(dim=1)
# #         self.sm = nn.Softmax(dim=1)
        
#     def forward(self, x):
#         # x shape (batch, time_step, input_size)
#         # r_out shape (batch, time_step, lstm_output_size)
#         # h_n shape (n_layers, batch, hidden_size)
#         # h_c shape (n_layers, batch, hidden_size)
#         # out shape (batch, output_size)
            
#         # None represents zero initial hidden state, so don't have to implement initHidden
#         r_out, (h_n, h_c) = self.lstm(x, None)
# #         print(r_out.size())
#         out_fc1 = self.fc1(r_out)
# #         out_fc2 = self.fc2(out_fc1)
# #         out = self.sm(out_fc2)
# #         print(out)

#         debug = False
#         if debug == True:
# #             size of x is  torch.Size([16, 6, 150])
# #             size of r_out is  torch.Size([16, 6, 128])
# #             size of h_n is  torch.Size([4, 16, 64])
# #             size of h_c is  torch.Size([4, 16, 64])
# #             size of out_fc1 is  torch.Size([16, 6, 32])
# #             size of out_fc2 is  torch.Size([16, 6, 2])
#             print('-----------------------------')
#             print('size of x is ', x.size())
#             print('size of r_out is ', r_out.size())
#             print('size of h_n is ', h_n.size())
#             print('size of h_c is ', h_c.size())
#             print('size of out_fc1 is ', out_fc1.size())
#             print('size of out_fc2 is ', out_fc2.size())
# #             print('size of out is ', out.size())
#             print('-----------------------------')
#             sys.exit()

# #         return out_fc2
#         return out_fc1
    
    
    


# # DON'T DELETE
# # Convolutional neural network (two convolutional layers)
# class FeatureExtractor(nn.Module):
#     def __init__(self, input_dim=50, input_channel=1, channel_n=4):
#         super(FeatureExtractor, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv1d(input_channel, channel_n, kernel_size=KERNEL_SIZE, stride=1, padding=2),
#             nn.BatchNorm1d(channel_n),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2))
#         self.layer2 = nn.Sequential(
#             nn.Conv1d(channel_n, channel_n, kernel_size=KERNEL_SIZE, stride=1, padding=2),
#             nn.BatchNorm1d(channel_n),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2))
#         self.layer3 = nn.Sequential(
#             nn.Conv1d(channel_n, channel_n, kernel_size=KERNEL_SIZE, stride=1, padding=2),
#             nn.BatchNorm1d(channel_n),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2))

#         cnn_layer1_dim = (input_dim+2*2-1*(KERNEL_SIZE-1)-1)+1
#         pool_layer1_dim = math.floor((cnn_layer1_dim-1*(2-1)-1)/2+1)

#         cnn_layer2_dim = (pool_layer1_dim+2*2-1*(KERNEL_SIZE-1)-1)+1
#         pool_layer2_dim = math.floor((cnn_layer2_dim-1*(2-1)-1)/2+1)

#         cnn_layer3_dim = (pool_layer2_dim+2*2-1*(KERNEL_SIZE-1)-1)+1
#         pool_layer3_dim = math.floor((cnn_layer3_dim-1*(2-1)-1)/2+1)

#         self.feature_out_dim = int(pool_layer3_dim*channel_n)
#         #       self.feature_out_dim = int(pool_layer2_dim*channel_n)
#         pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         #       print('FeatureExtractor_total_params:', pytorch_total_params)
#         #       print('feature_out_dim:', self.feature_out_dim)

#     def forward(self, x):
# #         print(x.size())
#         x = x[:,None,:]
# #         print(x.size())

# #         print(x.size())
        
#         out1 = self.layer1(x.float())
#         out2 = self.layer2(out1)
#         out3 = self.layer3(out2)
# #         print(out3.size())

#         out = out3.reshape(out3.size(0), -1)
        
        

#         debug = False
#         if debug == True:
# #             size of x is  torch.Size([16, 6, 150])
# #             size of r_out is  torch.Size([16, 6, 128])
# #             size of h_n is  torch.Size([4, 16, 64])
# #             size of h_c is  torch.Size([4, 16, 64])
# #             size of out_fc1 is  torch.Size([16, 6, 32])
# #             size of out_fc2 is  torch.Size([16, 6, 2])
#             print('-----------------------------')
#             print('size of x is ', x.size())
#             print('size of out1 is ', out1.size())
#             print('size of out2 is ', out2.size())
#             print('size of out3 is ', out3.size())
#             print('size of out is ', out.size())
#             print('-----------------------------')
#             sys.exit()

# #         print(out.size())
#         return out

# class cnnlstmnet(nn.Module):
#     r"""cnnlstmnet is a simple recurrent neural network that contains one 
#     hidden layer of 64 nodes with 3 time steps by default. It expects 
#     an input of 3D tensor with a dimension (batch, time_step, 
#     input_size). The output will be a 2D tensor with a dimension (N, 2).
#     Args:
#         - None. The variables used for each sub-layer are hard-coded
#     Shape:
#         - Input: :math:`(batch, time_step, input_size)`
#         - Output: :math:`(batch, output_size)`
#     Examples::
#         >>> m = lstmnet()
#         >>> batchSize = 16
#         >>> featDim = 10
#         >>> timeStep = 3
#         >>> input = torch.randn(batchSize, timeStep, featDim)
#         >>> output = m(input)
#         >>> output.size()
#             (16, 2)
#     """
#     def __init__(self, device, inputDim=10, hiddenDim=8, input_channel=1, stepN=6, outputDim=2):
#         super(cnnlstmnet, self).__init__()
#         self.outputDim = outputDim
#         self.step_n = stepN
#         self.device = device

        
#         self.feature_extractor = FeatureExtractor(input_dim=inputDim, input_channel=input_channel).to(device).float()
#         feature_out_dim = self.feature_extractor.feature_out_dim
#         self.feature_out_dim = feature_out_dim

#         self.lstm = nn.LSTM(         # if use nn.RNN(), it hardly learns
#             input_size=feature_out_dim,
#             hidden_size=hiddenDim,         # rnn hidden unit
#             num_layers=num_layers,           # number of rnn layer
#             batch_first=True,       # input & output will has batch size as 1st dimension. e.g. (batch, time_step, input_size)
#             bidirectional=bidirectional,
#             dropout=P_DROPOUT
#         )
        
#         self.fc1 = nn.Linear(hiddenDim, outputDim)
# #         self.fc1 = nn.Linear(hiddenDim*2, outputDim)
# #         self.fc1 = nn.Linear(hiddenDim*2, 32)
# #         self.fc2 = nn.Linear(32, outputDim)
        
#     def forward(self, x):
#         # x shape (batch, time_step, input_size)
#         # r_out shape (batch, time_step, lstm_output_size)
#         # h_n shape (n_layers, batch, hidden_size)
#         # h_c shape (n_layers, batch, hidden_size)
#         # out shape (batch, output_size)
            
# #         cnn_out_seq = torch.zeros_like(x)
#         cnn_out_seq = torch.ones((x.size()[0], x.size()[1], self.feature_out_dim), dtype=torch.float).to(self.device)

        
# #         print(cnn_out_seq.size())
#         for t in range(self.step_n):
#           # Input: (N, C_in, L_in)
#           # Output: (N, L_out=self.feature_out_dim*C_out)
#           # print('show size')
#           # print(feature_out_seq.size(), x_seq[t,:,:,:].size(), self.feature_extractor(x_seq[t,:,:,:]).size())
#           # sys.exit()
        
# #             print(self.feature_extractor(x[:,t,:]).size())
# #             print(x[:,t,:].size(), x.size())

#             cnn_out_seq[:,t,:] = self.feature_extractor(x[:,t,:])
    
    
        
#         # None represents zero initial hidden state, so don't have to implement initHidden

#         r_out, (h_n, h_c) = self.lstm(cnn_out_seq, None)
#         out_fc1 = self.fc1(r_out)
# #         out_fc2 = self.fc2(out_fc1)
# #         out = self.sm(out_fc2)
# #         print(out)

#         debug = False
#         if debug == True:
# #             size of x is  torch.Size([16, 6, 150])
# #             size of r_out is  torch.Size([16, 6, 128])
# #             size of h_n is  torch.Size([4, 16, 64])
# #             size of h_c is  torch.Size([4, 16, 64])
# #             size of out_fc1 is  torch.Size([16, 6, 32])
# #             size of out_fc2 is  torch.Size([16, 6, 2])
#             print('-----------------------------')
#             print('size of x is ', x.size())
#             print('size of cnn_out_seq is ', cnn_out_seq.size())
#             print('size of r_out is ', r_out.size())
#             print('size of h_n is ', h_n.size())
#             print('size of h_c is ', h_c.size())
#             print('size of out_fc1 is ', out_fc1.size())
#             print('size of out_fc2 is ', out_fc2.size())
# #             print('size of out is ', out.size())
#             print('-----------------------------')
#             sys.exit()

# #         return out_fc2
#         return out_fc1








# # KERNEL_SIZE = 5


# class FeatureExtractor_CNN(nn.Module):
#     def __init__(self, input_dim=50, input_channel=1, channel_n=4):
#         super(FeatureExtractor_CNN, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv1d(input_channel, channel_n, kernel_size=KERNEL_SIZE, stride=1, padding=2),
#             nn.BatchNorm1d(channel_n),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2))
#         self.layer2 = nn.Sequential(
#             nn.Conv1d(channel_n, channel_n, kernel_size=KERNEL_SIZE, stride=1, padding=2),
#             nn.BatchNorm1d(channel_n),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2))
#         self.layer3 = nn.Sequential(
#             nn.Conv1d(channel_n, channel_n, kernel_size=KERNEL_SIZE, stride=1, padding=2),
#             nn.BatchNorm1d(channel_n),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2))
#         self.layer4 = nn.Sequential(
#             nn.Conv1d(channel_n, channel_n, kernel_size=KERNEL_SIZE, stride=1, padding=2),
#             nn.BatchNorm1d(channel_n),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2))
#         self.layer5 = nn.Sequential(
#             nn.Conv1d(channel_n, channel_n, kernel_size=KERNEL_SIZE, stride=1, padding=2),
#             nn.BatchNorm1d(channel_n),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2))

#         cnn_layer1_dim = (input_dim+2*2-1*(KERNEL_SIZE-1)-1)+1
#         pool_layer1_dim = math.floor((cnn_layer1_dim-1*(2-1)-1)/2+1)

#         cnn_layer2_dim = (pool_layer1_dim+2*2-1*(KERNEL_SIZE-1)-1)+1
#         pool_layer2_dim = math.floor((cnn_layer2_dim-1*(2-1)-1)/2+1)

#         cnn_layer3_dim = (pool_layer2_dim+2*2-1*(KERNEL_SIZE-1)-1)+1
#         pool_layer3_dim = math.floor((cnn_layer3_dim-1*(2-1)-1)/2+1)

#         cnn_layer4_dim = (pool_layer3_dim+2*2-1*(KERNEL_SIZE-1)-1)+1
#         pool_layer4_dim = math.floor((cnn_layer4_dim-1*(2-1)-1)/2+1)

#         cnn_layer5_dim = (pool_layer4_dim+2*2-1*(KERNEL_SIZE-1)-1)+1
#         pool_layer5_dim = math.floor((cnn_layer5_dim-1*(2-1)-1)/2+1)
        
#         self.feature_out_dim = int(pool_layer5_dim*channel_n)
#         #       self.feature_out_dim = int(pool_layer2_dim*channel_n)
#         pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         #       print('FeatureExtractor_total_params:', pytorch_total_params)
#         #       print('feature_out_dim:', self.feature_out_dim)

#     def forward(self, x):
# #         print(x.size())
# #         x = x[:,None,:]
# #         print(x.size())

# #         print(x.size())
        
#         out1 = self.layer1(x.float())
#         out2 = self.layer2(out1)
#         out3 = self.layer3(out2)
#         out4 = self.layer4(out3)
#         out5 = self.layer5(out4)
# #         print(out3.size())

#         out = out5.reshape(out5.size(0), -1)
        
        

#         debug = False
#         if debug == True:
# #             size of x is  torch.Size([16, 6, 150])
# #             size of r_out is  torch.Size([16, 6, 128])
# #             size of h_n is  torch.Size([4, 16, 64])
# #             size of h_c is  torch.Size([4, 16, 64])
# #             size of out_fc1 is  torch.Size([16, 6, 32])
# #             size of out_fc2 is  torch.Size([16, 6, 2])
#             print('-----------------------------')
#             print('size of x is ', x.size())
#             print('size of out1 is ', out1.size())
#             print('size of out2 is ', out2.size())
#             print('size of out3 is ', out3.size())
#             print('size of out is ', out.size())
#             print('-----------------------------')
#             sys.exit()

# #         print(out.size())
#         return out


# class GenderClassifier(nn.Module):
#     def __init__(self, num_classes=10, input_dim=50):
#         super(GenderClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, num_classes)

#     def forward(self, x):
#         out = self.fc1(x)
#         return out

# class cnnnet(nn.Module):
#     def __init__(self, inputDim=1000, input_channel=1, class_N=2):
#         super(cnnnet, self).__init__()
# #         self.feature_extractor = FeatureExtractor(input_dim=input_dim, channel_n=channel_n)
#         self.feature_extractor = FeatureExtractor_CNN(input_dim=inputDim, input_channel=input_channel)

#         feature_out_dim = self.feature_extractor.feature_out_dim

#         self.gender_classfier = GenderClassifier(num_classes=class_N, input_dim=feature_out_dim)

#         pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
# #     print('DannModel_total_params:', pytorch_total_params)

#     def forward(self, x):
#         feature_out = self.feature_extractor(x)
#         classifier_output = self.gender_classfier(feature_out)
# #         domain_output = self.domain_classifier(feature_out, 1)
#         return classifier_output
