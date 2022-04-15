import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from dataIO import *

# from imgaug import augmenters as iaa
# import imgaug as ia
from stage3_preprocess import *

def get_samples(inputdir, set_name, training_params):
    
    inputdir_set = inputdir+set_name
    
    input_names = training_params['input_names']
    print('input_names are:', input_names)
    output_names = training_params['output_names']
    print('output_names are:', output_names)

    meta_names = training_params['meta_names']
    print('meta_names are:', meta_names)

    # input_names = ['ECG', 'accelZ']

    indices_sig = []
    for sig_name in input_names:
        i_sig = list_input.index(sig_name)
        indices_sig.append(i_sig)

    # output_names = ['VT_cosmed', 'task']

    indices_label = []
    for label_name in output_names:
        i_label = list_output.index(label_name)
        indices_label.append(i_label)

    indices_meta = []
    for meta_name in meta_names:
        i_meta = list_meta.index(meta_name)
        indices_meta.append(i_meta)
        
#     print(data_loader('label', inputdir_set).shape, indices_label)
    data = data_loader('data', inputdir_set)[:,indices_sig,:]
    label = data_loader('label', inputdir_set)[:, indices_label]
    meta = data_loader('meta', inputdir_set)[:, indices_meta]

    return data, label, meta

    
# def get_samples(inputdir, set_name):
    
#     inputdir_set = inputdir+set_name
    
#     input_names = training_params['input_names']
#     output_names = training_params['output_names']
    
    
#     data = data_loader('data', inputdir_set)[:,0,:][:,None,:] # get ECG only
#     label = data_loader('label', inputdir_set)[:, [0, -1]] # get RR and task
    
#     return data, label


class FS_Dataset(Dataset):
    def __init__(self, data, label, meta, training_params, transform=None):
        self.data = data
        self.label = label
        self.meta = meta
        
    def __getitem__(self, index):
        data = self.data[index,:,:]
#         label = np.asarray(self.label[index,0]).astype(float)
        label = self.label[index, :].astype(float)
        meta = self.meta[index,:]
        
#         print(label.shape, meta.shape)
        
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
#         meta = torch.from_numpy(meta)

        
        return data, label, meta

    def __len__(self):
#         return len(self.data)
        return self.data.shape[0]

def get_loaders(inputdir, training_params):

    data_train, label_train, meta_train = get_samples(inputdir, 'train/', training_params)
    data_val, label_val, meta_val = get_samples(inputdir, 'val/', training_params)
#     data_test, label_test = get_samples(inputdir, 'test/')

    train_dataset = FS_Dataset(data_train, label_train, meta_train, training_params)
    val_dataset = FS_Dataset(data_val, label_val, meta_val, training_params)
#         test_dataset = PTBXL_Dataset(data_test, label_test, training_params, transform=transforms)


    FS_datasets = {
        'train': train_dataset, 'val': val_dataset
    }
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True, num_workers=0),
        'train_eval': DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0),
    }
    

    dataset_sizes = {
        x: len(FS_datasets[x]) for x in FS_datasets.keys()
    }
    
    return dataloaders, dataset_sizes

