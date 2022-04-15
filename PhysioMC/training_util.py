import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from dataIO import *
from stage3_preprocess import *


# def train_resnet(model, dataloader, optimizer, criterion, epoch, training_params):
def train_resnet(model, dataloader, training_params):

    optimizer = training_params['optimizer']
    criterion = training_params['criterion']
    epoch = training_params['epoch']
    device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')

    dataset_size = len(dataloader.dataset)
    total_loss = 0

    model.train()
    for i, (data, label, meta) in enumerate(dataloader):
#         print('epoch', i)
        # 1. get data
        data = data.to(device).float()
#         label = label.to(device=device, dtype=torch.float)
        label = label.to(device=device, dtype=torch.float)

        # 2. infer by net
        out = model(data)
        
        # 3. loss function
#         print(out.squeeze().size(), label.size())
        losses = criterion(out, label)
        loss = losses['total']

        # 3. accumulate the loss
        total_loss += loss.data.detach().cpu().numpy()

        # 4. Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_loss = total_loss/dataset_size
    # # TODO: remove performance_dict
    performance_dict = {
        'total_loss': total_loss,
    }
    return performance_dict


# def eval_resnet(model, dataloader, optimizer, criterion, epoch, training_params):
def eval_resnet(model, dataloader, training_params):

    optimizer = training_params['optimizer']
    criterion = training_params['criterion']
    epoch = training_params['epoch']
    
    device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')

    dataset_size = len(dataloader.dataset)

    total_loss = 0
    
    model.eval()
#     print('\t\tswitch model to eval')

    for i, (data, label, meta) in enumerate(dataloader):
        # 1. get data        
        data = data.to(device).float()
        label = label.to(device=device, dtype=torch.float)

        # 2. infer by net
        out = model(data)
            
        # 3. loss function
#         loss = criterion(out, label)
        losses = criterion(out, label)
        loss = losses['total']

        # 3. accumulate the loss
        total_loss += loss.data.detach().cpu().numpy()
        

        
    total_loss = total_loss/dataset_size

    performance_dict = {'total_loss': total_loss,
                       }
    
    return performance_dict




# def pred_resnet(model, dataloader, criterion, epoch, training_params):
def pred_resnet(model, dataloader, training_params):

    optimizer = training_params['optimizer']
    criterion = training_params['criterion']
    epoch = training_params['epoch']
    
    device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')

    dataset_size = len(dataloader.dataset)

    total_loss = 0
    
#     out_arr = np.empty(0)
#     label_arr = np.empty(0)
    out_dict = {}
    label_dict= {}
    
    for task in training_params['tasks']:
        out_dict[task] = np.empty(0)
        label_dict[task] = np.empty(0)

    meta_arr = []
    
    model.eval()
#     print('\t\tswitch model to eval')

    for i, (data, label, meta) in enumerate(dataloader):
        # 1. get data
        data = data.to(device).float()
        label = label.to(device=device, dtype=torch.float)

        # 2. infer by net
        out = model(data)
            
        # 3. loss function
#         loss = criterion(out, label)
        losses = criterion(out, label)
        loss = losses['total']
        
        # 4. compute the class loss of features
        total_loss += loss.data.detach().cpu().numpy()

        
#         out = out.detach().cpu().numpy()
#         label = label.detach().cpu().numpy()
        
#         print(out_softmax.shape)
        # out_softmax[:,1] = prob of samples belonging to class 1
#         print(out_arr.shape)

        for task in training_params['tasks']:
#             print( out[task].detach().cpu().numpy().shape)
#             print(label[:,training_params['tasks'].index(task)].detach().cpu().numpy().shape)
            out_dict[task] = np.r_[out_dict[task], out[task].detach().cpu().numpy().squeeze()]
            label_dict[task] = np.r_[label_dict[task], label[:,training_params['tasks'].index(task)].detach().cpu().numpy().squeeze()]

            
            
            
#         out_arr = np.r_[out_arr, out]
# #         sys.exit()
#         label_arr = np.r_[label_arr, label]
    
#         print(meta.shape)
        meta_arr.append( meta.detach().cpu().numpy())


    meta_arr = np.concatenate(meta_arr,axis=0)
        
    # TODO: plot feature maps and filters
        
    total_loss = total_loss/dataset_size
    performance_dict = {'total_loss': total_loss,
                        'out_dict': out_dict,
                        'label_dict': label_dict,
                        'meta_arr': meta_arr,
                       }
    
    return performance_dict



