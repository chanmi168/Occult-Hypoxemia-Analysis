import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

import sys


# KERNEL_SIZE = 5


class FeatureExtractor_CNN(nn.Module):
    def __init__(self, training_params=None):
        super(FeatureExtractor_CNN, self).__init__()
        
        input_dim = training_params['data_dimensions'][1]
        input_channel = training_params['data_dimensions'][0]
        channel_n = training_params['channel_n']
        kernel_size = training_params['kernel_size']
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channel, channel_n, kernel_size=kernel_size, stride=1, padding=2),
            nn.BatchNorm1d(channel_n),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=2),
            nn.BatchNorm1d(channel_n),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=2),
            nn.BatchNorm1d(channel_n),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv1d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=2),
            nn.BatchNorm1d(channel_n),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
#         self.layer5 = nn.Sequential(
#             nn.Conv1d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=2),
#             nn.BatchNorm1d(channel_n),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2))
#         self.layer6 = nn.Sequential(
#             nn.Conv1d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=2),
#             nn.BatchNorm1d(channel_n),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2))
#         self.layer7 = nn.Sequential(
#             nn.Conv1d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=2),
#             nn.BatchNorm1d(channel_n),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2))

        cnn_layer1_dim = (input_dim+2*2-1*(kernel_size-1)-1)+1
        pool_layer1_dim = math.floor((cnn_layer1_dim-1*(2-1)-1)/2+1)

        cnn_layer2_dim = (pool_layer1_dim+2*2-1*(kernel_size-1)-1)+1
        pool_layer2_dim = math.floor((cnn_layer2_dim-1*(2-1)-1)/2+1)

        cnn_layer3_dim = (pool_layer2_dim+2*2-1*(kernel_size-1)-1)+1
        pool_layer3_dim = math.floor((cnn_layer3_dim-1*(2-1)-1)/2+1)

        cnn_layer4_dim = (pool_layer3_dim+2*2-1*(kernel_size-1)-1)+1
        pool_layer4_dim = math.floor((cnn_layer4_dim-1*(2-1)-1)/2+1)

#         cnn_layer5_dim = (pool_layer4_dim+2*2-1*(kernel_size-1)-1)+1
#         pool_layer5_dim = math.floor((cnn_layer5_dim-1*(2-1)-1)/2+1)
        
#         cnn_layer6_dim = (pool_layer5_dim+2*2-1*(kernel_size-1)-1)+1
#         pool_layer6_dim = math.floor((cnn_layer6_dim-1*(2-1)-1)/2+1)
        
#         cnn_layer7_dim = (pool_layer6_dim+2*2-1*(kernel_size-1)-1)+1
#         pool_layer7_dim = math.floor((cnn_layer7_dim-1*(2-1)-1)/2+1)
        
#         self.feature_out_dim = int(pool_layer5_dim*channel_n)
        self.feature_out_dim = int(pool_layer4_dim*channel_n)
        self.channel_n = int(channel_n)
        #       self.feature_out_dim = int(pool_layer2_dim*channel_n)
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        #       print('FeatureExtractor_total_params:', pytorch_total_params)
        print('feature_out_dim:', self.feature_out_dim)

    def forward(self, x):
#         print(x.size())
#         x = x[:,None,:]
#         print(x.size())

#         print(x.size())
        
        out = self.layer1(x.float())
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
#         out = self.layer5(out)
#         out = self.layer6(out)
#         out = self.layer7(out)
#         print(out3.size())

        out = out.reshape(out.size(0), -1)
        
        

        debug = False
        if debug == True:
            # size of x is  torch.Size([2, 1, 1000])
            # size of out1 is  torch.Size([2, 4, 500])
            # size of out2 is  torch.Size([2, 4, 250])
            # size of out3 is  torch.Size([2, 4, 125])
            # size of out4 is  torch.Size([2, 4, 62])
            # size of out5 is  torch.Size([2, 4, 31])
            # size of out is  torch.Size([2, 124])
            print('-----------------------------')
            print('size of x is ', x.size())
            print('size of out1 is ', out1.size())
            print('size of out2 is ', out2.size())
            print('size of out3 is ', out3.size())
            print('size of out4 is ', out4.size())
            print('size of out5 is ', out5.size())
            print('size of out is ', out.size())
            print('-----------------------------')
            sys.exit()

#         print(out.size())
        return out


# class GenderClassifier(nn.Module):
#     def __init__(self, num_classes=10, input_dim=50):
#         super(GenderClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, num_classes)

#     def forward(self, x):
#         out = self.fc1(x)
#         return out
    
    
class RespiratoryRegression(nn.Module):
    def __init__(self, num_classes=10, input_dim=50, channel_n=64):
        super(RespiratoryRegression, self).__init__()
        self.bn = nn.BatchNorm1d(channel_n)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, num_classes)
#         self.fc1 = nn.Linear(input_dim, 50)
#         self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):  
        out = self.fc1(x)
#         out = F.relu(out)
#         out = self.fc2(out)
        
        return out
        
class EERegression(nn.Module):
    def __init__(self, num_classes=10, input_dim=50, channel_n=64):
        super(EERegression, self).__init__()
        self.bn = nn.BatchNorm1d(channel_n)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, num_classes)
#         self.fc1 = nn.Linear(input_dim, 50)
#         self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):  
        out = self.fc1(x)        
        return out
    
    
class RRRegression(nn.Module):
    def __init__(self, num_classes=10, input_dim=50, channel_n=64):
        super(RRRegression, self).__init__()
        self.bn = nn.BatchNorm1d(channel_n)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, num_classes)
#         self.fc1 = nn.Linear(input_dim, 50)
#         self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):  
        out = self.fc1(x)        
        return out
    
    

# #         out = self.bn(x)
# #         out = self.relu(out)
#         out = out.mean(-1)
# #         if self.verbose:
# #             print('final pooling', out.shape)
#         # out = self.do(out)
#         out = self.fc1(out)

class cnnnet(nn.Module):
#     def __init__(self, inputDim=1000, input_channel=1, class_N=2, training_params=None):
    def __init__(self, class_N=1, training_params=None):
        super(cnnnet, self).__init__()
        
        input_dim = training_params['data_dimensions'][1]
        input_channel = training_params['data_dimensions'][0]
        channel_n = training_params['channel_n']
        kernel_size = training_params['kernel_size']
        self.tasks = training_params['tasks']
#         self.feature_extractor = FeatureExtractor(input_dim=input_dim, channel_n=channel_n)
        self.feature_extractor = FeatureExtractor_CNN(training_params=training_params)
#         self.feature_extractor = FeatureExtractor_CNN(input_dim=inputDim, input_channel=input_channel, training_params=training_params)

        feature_out_dim = self.feature_extractor.feature_out_dim
        channel_n = self.feature_extractor.channel_n

#         self.gender_classfier = GenderClassifier(num_classes=class_N, input_dim=feature_out_dim)
#         self.respiratory_regressor = RespiratoryRegression(num_classes=class_N, input_dim=feature_out_dim)
        self.regressors = {}
    
        self.regressors = nn.ModuleDict(
            [[task, RespiratoryRegression(num_classes=class_N, input_dim=feature_out_dim)] for task in self.tasks]
        )
            
            
            
            
            
#             [
            
#                 ['lrelu', RespiratoryRegression(num_classes=class_N, input_dim=feature_out_dim)],
#                 ['prelu', nn.PReLU()]
#         ])
    
#         regresso = [task: RespiratoryRegression(num_classes=class_N, input_dim=feature_out_dim) for task in self.tasks]


        
#         for task in training_params['tasks']:
#             self.regressors[task] = RespiratoryRegression(num_classes=class_N, input_dim=feature_out_dim)
#         self.EE_regressor = EERegression(num_classes=class_N, input_dim=feature_out_dim)
#         self.RR_regressor = RRRegression(num_classes=class_N, input_dim=feature_out_dim)

        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#     print('DannModel_total_params:', pytorch_total_params)

    def forward(self, x):
        feature_out = self.feature_extractor(x)
#         print('feature_out size:', feature_out.size())
        output = {}
        for task in self.tasks:
#             next(self.regressors[task].parameters()).is_cuda 
            output[task] = self.regressors[task](feature_out)
            
            
            
#         EE_output = self.EE_regressor(feature_out)
#         RR_output = self.RR_regressor(feature_out)
# #         print('regression_output:', regression_output.size())
# #         return regression_output
#         output = {
#             'RR_cosmed': RR_output,
#             'EE_cosmed': EE_output
#         }
#         output = [RR_output, EE_output]
        return output
