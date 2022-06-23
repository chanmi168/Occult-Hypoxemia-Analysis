import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.preprocessing import LabelEncoder

from dataIO import *
from filters import *


# def get_samples(data, label, inputdir, training_params):
def get_samples(training_params):
    
    data = training_params['data']
    label = training_params['label']
    meta = training_params['meta']
    
    CV = training_params['CV_config']['CV']

    # dataset_dict = training_params['dataset_dict']
    # i_split = dataset_dict['list_label'].index('split_name')
    i_split = training_params['meta_names'].index('split_name')

    le = training_params['split_name_encoder']
    label_split_names = le.inverse_transform(meta[:,i_split].astype(int))


    # TEST_set = label[:,i_split]=='TEST'
    # train_set = (label[:,i_split]!='TEST') & (label[:,i_split]!='TRAIN-CV{}'.format(CV))
    # val_set = (label[:,i_split]!='TEST') & (label[:,i_split]=='TRAIN-CV{}'.format(CV))

    TEST_set = label_split_names=='TEST'
    train_set = (label_split_names!='TEST') & (label_split_names!='TRAIN-CV{}'.format(CV))
    val_set = (label_split_names!='TEST') & (label_split_names=='TRAIN-CV{}'.format(CV))


    data_TEST = data[TEST_set,:,:]
    label_TEST = label[TEST_set,:]
    meta_TEST = meta[TEST_set,:]

    data_train = data[train_set,:,:]
    label_train = label[train_set,:]
    meta_train = meta[train_set,:]

    data_val = data[val_set,:,:]
    label_val = label[val_set,:]
    meta_val = meta[val_set,:]

    samples_dict = {
        'data_TEST': data_TEST,
        'label_TEST': label_TEST,
        'meta_TEST': meta_TEST,
        'data_train': data_train,
        'label_train': label_train,
        'meta_train': meta_train,
        'data_val': data_val,
        'label_val': label_val,
        'meta_val': meta_val,
    }
    
    
    # return data_TEST, label_TEST, data_train, label_train, data_val, label_val
    return samples_dict


def reduce_data_dim(data, training_params):
    # data has a dimension of (N_instances, N_feature/signals, N_samples)
    # data_reduced has a dimension of (N_instances, N_feature/signals, output_size)
    # downsample_factor = N_samples // output_size

    data_reduced = np.zeros((data.shape[0], data.shape[1], training_params['output_dim']))
    
    downsample_factor = data.shape[-1] // training_params['output_dim']

    for i_instance in range(data_reduced.shape[0]):
        for i_signal in range(data_reduced.shape[1]):
            sig = data[i_instance, i_signal, :]
            sig_smoothed = get_smooth(sig, N=int(training_params['FS_RESAMPLE_DL']*5))
            sig_smoothed = sig_smoothed[downsample_factor//2:][::downsample_factor]
            data_reduced[i_instance, i_signal, :] = sig_smoothed
    
    return data_reduced, downsample_factor


# class Emory_Dataset(Dataset):
#     def __init__(self, data, label, training_params, transform=None):
#         self.data = data
#         self.feature = np.zeros((data.shape[0], 0))
#         self.label = label.astype(float)
#         # self.meta = meta
        
#     def __getitem__(self, index):
#         data = self.data[index, :]
#         label = self.label[index, :]
#         feature = self.feature[index, :]
        
#         data = torch.from_numpy(data)
#         label = torch.from_numpy(label)
#         feature = torch.from_numpy(feature)
        
#         return data, feature, label

#     def __len__(self):
#         return self.data.shape[0]

class Emory_Dataset(Dataset):
    def __init__(self, samples_dict, training_params, mode='train', transform=None):
        self.data = samples_dict['data_'+mode]
        self.feature = np.zeros((self.data.shape[0], 0))
        self.label = samples_dict['label_'+mode].astype(float)
        self.meta = samples_dict['meta_'+mode].astype(float)
        
    def __getitem__(self, index):
        data = self.data[index, :]
        label = self.label[index, :]
        feature = self.feature[index, :]
        meta = self.meta[index, :]
        
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        feature = torch.from_numpy(feature)
        meta = torch.from_numpy(meta)
        
        return data, feature, label, meta

    def __len__(self):
        return self.data.shape[0]

def get_loaders(training_params):
# def get_loaders(inputdir, training_params):
#     data = data_loader('data', inputdir)[:,None,:] # make middle dimension (channel) one
#     label = data_loader('label', inputdir)
    
    

    # print(label)
    samples_dict = get_samples(training_params)
    # data_TEST, label_TEST, data_train, label_train, data_val, label_val = get_samples(data, label, training_params)
        
#     # zero mean unit variance
#     feature_mean = np.mean(feature_train, axis=0)
#     feature_std = np.std(feature_train, axis=0)

#     feature_train = (feature_train-feature_mean)/feature_std
#     feature_val = (feature_val-feature_mean)/feature_std

#     print(feature_mean, feature_std)

    train_dataset = Emory_Dataset(samples_dict, training_params, 'train')
    val_dataset = Emory_Dataset(samples_dict, training_params, 'val')
    TEST_dataset = Emory_Dataset(samples_dict, training_params, 'TEST')
    # train_dataset = Emory_Dataset(data_train, label_train, training_params)
    # val_dataset = Emory_Dataset(data_val, label_val, training_params)
    # TEST_dataset = Emory_Dataset(data_TEST, label_TEST, training_params)


    Emory_datasets = {
        'train': train_dataset, 'val': val_dataset, 'TEST': TEST_dataset
    }
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True, num_workers=10),
        # 'train_eval': DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=10),
        'TEST': DataLoader(TEST_dataset, batch_size=128, shuffle=False, num_workers=10),
        # 'feature_mean': feature_mean,
        # 'feature_std': feature_std,
    }
    

    dataset_sizes = {
        x: len(Emory_datasets[x]) for x in Emory_datasets.keys()
    }
    
    return dataloaders, dataset_sizes, training_params

