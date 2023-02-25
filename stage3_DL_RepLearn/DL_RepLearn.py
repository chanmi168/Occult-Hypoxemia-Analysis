#!/usr/bin/env python
# coding: utf-8

# # load data
# # design model
# # test model
# # TODO: continue to debug model block

# In[1]:


import pandas as pd
from datetime import datetime
import os
import pickle
from datetime import timedelta
from datetime import datetime
from datetime import date
from datetime import time
from dateutil.relativedelta import relativedelta
import argparse

import os
import math

import pandas as pd
import numpy as np
import os

import json
import pytz
import pprint

import sys
from scipy.io import loadmat

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
import matplotlib.ticker as plticker

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets, models

# %pip install torch-summary
from torchsummary import summary
torch.manual_seed(0)

import seaborn as sns
from matplotlib.cm import ScalarMappable

from sklearn.preprocessing import LabelEncoder


import sys
sys.path.append('../') # add this line so data are visible in this file
sys.path.append('../../') # add this line so data are visible in this file
sys.path.append('../PhysioMC/') # add this line so data are visible in this file

# from PatchWand import *
from filters import *
from setting import *
# from preprocessing import *
from ECG_module import *
from dataIO import *
from evaluate import *
from stage1_PPG_analysis import *
from plotting_tools import *
from stage4_regression import *


from DR_extension.training_util import *
from DR_extension.dataset_util import *
from DR_extension.evaluation_util import *
from DR_extension.models import *
from DR_extension.models_CNNlight import *

from importlib import reload
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# # m = nn.Softmax(dim=-1)
# input1 = torch.randn(64, 3, 10, 100)
# input2 = torch.randn(64, 3, 10, 100)

# attn = torch.matmul(input1, input2.transpose(2, 3))
# # fc = nn.Linear(25, 37, bias=False)

# print(attn.size())


# In[3]:


# fc(input).size()


# In[4]:


# output.sum(axis=-1), output.sum(axis=-1).shape


# In[ ]:





# In[5]:


print(torch.version.cuda)
print(torch.cuda.is_available())


# In[6]:


parser = argparse.ArgumentParser(description='feature_learning')
parser.add_argument('--input_folder', metavar='input_folder', help='input_folder',
                    default='../')
parser.add_argument('--output_folder', metavar='output_folder', help='output_folder',
                    default='../')
parser.add_argument('--training_params_file', metavar='training_params_file', help='training_params_file',
                    default='training_params_list.json')


# checklist 3: comment first line, uncomment second line
# args = parser.parse_args(['--input_folder', '../../data/stage1/waveform/', 
# args = parser.parse_args(['--input_folder', '../../data/stage3_DL_prepare/', 
#                           '--output_folder', '../../data/stage3_DL_RepLearn/',
#                           '--training_params_file', 'training_params_baseline.json',
#                           # '--training_params_file', 'training_params_dummy.json',
#                          ])
args = parser.parse_args()
print(args)


# In[7]:


# selected_id = 10


# In[8]:





# In[9]:


# inputdir = '../../data/stage1/waveform/'
inputdir = args.input_folder
outputdir = args.output_folder
training_params_file = args.training_params_file

# outputdir = '../../data/stage3_DL_RepLearn/'
if not os.path.exists(outputdir):
    os.makedirs(outputdir)


# In[10]:


def get_model_out_names(training_params):
    model_out_names = []

#     for output_name in training_params['output_names']:
    for output_name in training_params['output_names']:
        for input_name in training_params['input_names']:
            model_out_names.append(output_name+'-{}'.format(input_name))
    return model_out_names


# In[11]:


def encode_labels(label, training_params):

    dataset_dict = training_params['dataset_dict']

    race_encoder = LabelEncoder()
    i_race = dataset_dict['list_label'].index('Race String')
    label[:, i_race] = race_encoder.fit_transform(label[:, i_race])

    PAT_ID_encoder = LabelEncoder()
    i_PAT_ID = dataset_dict['list_label'].index('PAT_ID')
    label[:, i_PAT_ID] = PAT_ID_encoder.fit_transform(label[:, i_PAT_ID])

    split_name_encoder = LabelEncoder()
    i_split_name = dataset_dict['list_label'].index('split_name')
    label[:, i_split_name] = split_name_encoder.fit_transform(label[:, i_split_name])
    
    training_params['PAT_ID_encoder'] = PAT_ID_encoder
    training_params['race_encoder'] = race_encoder
    training_params['split_name_encoder'] = split_name_encoder

    return label, training_params


# In[12]:


trainer = train_dann
evaler = eval_dann
preder = pred_dann


# In[ ]:





# In[13]:


# TODO: move below to stage4_regression or dataset_util    
def store_data_meta_label(training_params):

    data = data_loader('data', training_params['inputdir'])[:,None,:] # make middle dimension (channel) one
    label_raw = data_loader('label', training_params['inputdir'])

    # choose a small subset first
    if training_params['tiny_dataset']:
        data = data[:5000,:,:]
        label_raw = label_raw[:5000,:]

    # encode the labels so they are not stored in strings but in int
    label_raw, training_params = encode_labels(label_raw, training_params)

    # get actual label and actual meta

    # select the relevant label (stored in output_names)

    list_label = dataset_dict['list_label']

    indices_label = []
    for label_name in training_params['output_names']:
        if 'reconstruction' in training_params['output_names']:
            continue
        i_label = list_label.index(label_name)
        indices_label.append(i_label)

    if training_params['output_names'][0]=='reconstruction':
        label = data
    else:
        label = label_raw[:, indices_label]

    # select the relevant meta (stored in output_names)
    indices_meta = []
    for meta_name in training_params['meta_names']:
        i_meta = list_label.index(meta_name)
        indices_meta.append(i_meta)

    meta = label_raw[:, indices_meta]

    # store them in training_params
    training_params['data'] = data
    training_params['label'] = label
    training_params['meta'] = meta

    return training_params


# In[14]:


with open(training_params_file) as json_file:
    training_params_list = json.load(json_file)

for training_params in [training_params_list[0]]:
    # include device in training_params
    device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_params['device'] = device
    
    training_params['sweep_name'] = training_params_file.split('_')[-1].split('.')[0]
    
    if 'training_mode' in training_params:
        training_mode = training_params['training_mode']
    else:
        training_params = 'subject_ind'

    training_params['CV_config'] = {
        'CV': 1,
    }

    training_params['FS_RESAMPLE_DL'] = 100
    
    dataset_dict = data_loader('dataset_dict', inputdir).item()
    training_params['dataset_dict'] = dataset_dict
    
    training_params['inputdir'] = inputdir
    training_params['outputdir'] = outputdir

    # load the data once only
    training_params = store_data_meta_label(training_params)


    dataloaders, dataset_sizes, training_params = get_loaders(training_params)
    # dataloaders, dataset_sizes, training_params = get_loaders(outputdir, training_params)
    print('data dimensions are:', dataloaders['val'].dataset.data.shape)
    print('feature dimensions are:', dataloaders['val'].dataset.feature.shape)
    print('meta dimensions are:', dataloaders['val'].dataset.meta.shape)
    print('label dimensions are:', dataloaders['val'].dataset.label.shape)

    data_dimensions = dataloaders['train'].dataset.__getitem__(0)[0].size()
    training_params['data_dimensions'] = list(data_dimensions) # should be (N_channel, N_samples)
    del dataloaders

#     sweep_name = training_params['sweep_name'] 
    
    
    training_params['featrue_extractor'] = extractor_dict[training_params['extractor_name']]
    # if training_params['model_name'] == 'FeatureExtractor_CNN':
    #     training_params['featrue_extractor'] = FeatureExtractor_CNN
    # elif training_params['model_name'] == 'ResNet1D':
    #     training_params['featrue_extractor'] = ResNet1D
    # elif training_params['model_name'] == 'FeatureExtractor_CNN2':
    #     training_params['featrue_extractor'] = FeatureExtractor_CNN2
    # elif training_params['model_name'] == 'FeatureExtractor_CNNlight':
    #     training_params['featrue_extractor'] = FeatureExtractor_CNNlight
    


    model_out_names = get_model_out_names(training_params)
    training_params['model_out_names'] = model_out_names
    


    training_params['FS_Extracted'] = training_params['FS_RESAMPLE_DL'] / (training_params['stride']**training_params['n_block'])

    
    
#     last_layer_dim = training_params['data_dimensions'][-1]
#     for n in range(training_params['n_block']):
#         last_layer_dim = round(last_layer_dim/training_params['stride'])

#     training_params['last_layer_dim'] = last_layer_dim
#     xf = np.linspace(0.0, 1.0/2.0*training_params['FS_Extracted'] , training_params['last_layer_dim']//2)*60    
#     mask = (xf>=label_range_dict['HR_DL'][0]) & (xf<=label_range_dict['HR_DL'][1])

#     training_params['xf'] = xf
#     training_params['xf_masked'] = xf[mask]
#     training_params['mask'] = mask

# # training_params = training_params_list[0]


# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:


# aaa['decoder1'] = nn.Linear(50,1)


# In[16]:


# aaa?


# In[17]:


# dataloaders, dataset_sizes, training_params = get_loaders(training_params)


# In[18]:


# dataloaders


# In[19]:


# dataset_dict['list_label']


# In[ ]:





# In[20]:


# training_params['race_encoder'].classes_


# In[21]:


# # df_demographic_processed = pd.read_csv(inputdir+'df_demographic_processed.csv.gz')  
# df_demographic_processed = pd.read_csv('../../data/stage1/waveform/'+'df_demographic_processed.csv.gz')  
# df_demographic_processed


# In[ ]:





# # test model

# In[22]:


def test_model_lstm(training_params):
    print('test_model_lstm')
    print('using model ', training_params['model_name'])

    # prepare model
    model = resp_multiverse(training_params=training_params)
    model = model.to(device).float()

    # prepare data
    dataloaders, dataset_sizes = get_loaders(inputdir, training_params)

    data = dataloaders['val'].dataset.data[:5,:,:]
    data = torch.from_numpy(data)

    feature = dataloaders['val'].dataset.feature[:5,:]
    feature = torch.from_numpy(feature)

    label = dataloaders['val'].dataset.label[:5,:]
    label = torch.from_numpy(label)

    data = data.to(device=device, dtype=torch.float)
    feature = feature.to(device=device, dtype=torch.float)
    label = label.to(device=device, dtype=torch.float)

    # model inference
    out = model(data, feature)

    # compute loss
    criterion = MultiTaskLoss(training_params)
    losses = criterion(out, label)

    # check losses
    print(losses)
    del model

    
def test_model(training_params):
    print('test_model')
    print('using model ', training_params['model_name'])

    model = resp_multiverse(training_params=training_params)
    summary(model, input_size=[tuple(training_params['data_dimensions']), (model.N_features,1)], device='cpu')
    print(model)
    del model
    
def test_model_dann(training_params):
    print('test_model_dann')
    print('using model ', training_params['model_name'])

    model = resp_DANN(training_params=training_params)
    print(model)

    summary(model, input_size=[tuple(training_params['data_dimensions']), (model.N_features,1)], device='cpu')
    del model

def test_PPG_compressor(training_params):
    print('test_PPG_compressor')
    print('using model ', training_params['model_name'])

    model = PPG_compressor(training_params=training_params)
    print(model)

    summary(model, input_size=[tuple(training_params['data_dimensions']), (model.N_features,1)], device='cpu')
    del model



debug_model = True
if debug_model==True:
    if 'LSTM' in training_params['model_name']:
        test_model_lstm(training_params)
    elif 'DANN' in training_params['model_name']:
        test_model_dann(training_params)
    elif 'PPG_VAEcompressor' in training_params['model_name']:
        test_PPG_compressor(training_params)


# In[ ]:





# In[ ]:





# In[ ]:





# ## make sure data can pass through the model

# In[23]:


check_data_flow = True

if check_data_flow:

    # dataloaders, dataset_sizes = get_loaders(inputdir, training_params)
    # dataloaders, dataset_sizes, training_params = get_loaders(outputdir, training_params)
    dataloaders, dataset_sizes, training_params = get_loaders(training_params)

    data_val = dataloaders['val'].dataset.data[:5,:,:]
    data_val = torch.from_numpy(data_val)
    data_val = data_val.to(device=device, dtype=torch.float)

    feature_val = dataloaders['val'].dataset.feature[:5,:]
    feature_val = torch.from_numpy(feature_val)
    feature_val = feature_val.to(device=device, dtype=torch.float)

    label_val = dataloaders['val'].dataset.label[:5,:]
    label_val = torch.from_numpy(label_val)
    label_val = label_val.to(device=device, dtype=torch.float)

    # model = resp_DANN(training_params=training_params)
    model = PPG_compressor(training_params=training_params)

    model = model.to(device).float()
    output, feature_out, mu, logvar = model(data_val, feature_val)

    # should be torch.Size([5, 1, 800]) torch.Size([5, 0]) torch.Size([5, 5]) torch.Size([5, 2]) torch.Size([5, 50])
    # print(data_train.size(), feature_train.size(), label_train.size(), output['Race String-PPG'].size(), feature_out['PPG'].size())
    
    # should be torch.Size([5, 1, 800]) torch.Size([5, 0]) torch.Size([5, 5]) torch.Size([5, 500]) torch.Size([5, 1, 100]), torch.Size([5, 25]), torch.Size([5, 25])
    print(data_val.size(), feature_val.size(), label_val.size(), output['reconstruction-PPG'].shape, feature_out['PPG'].size(), mu.size(), logvar.size())
    print('encoder_layer_dims:', model.encoders.PPG.encoder_layer_dims)
    print('output_channels:', model.encoders.PPG.output_channels)
    
    del model
    del dataloaders


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # activate wandb session

# In[24]:


if training_params['wandb']:
    wandb.login()
    os.environ["WANDB_DIR"] = os.path.abspath(outputdir)
    os.environ["WANDB_NOTEBOOK_NAME"] = 'PPG_compression'


# 
# # define outputdir

# In[25]:


def get_sweep_folder(training_params):
    n_block = training_params['n_block']
    inputs_combined = '+'.join([ i_name.split('_')[0] for i_name in training_params['input_names']])
    auxillary_weight = training_params['loss_weights']['auxillary_task']
    # adversarial_weight = training_params['adversarial_weight']
    channel_n = training_params['channel_n']

    list_act = '+'.join( [str(int) for int in training_params['activity_names']] )

    sweep_folder = '{}blocks-{}-weight{}-{}ch-act{}'.format(n_block, inputs_combined, auxillary_weight, channel_n, list_act)

    return sweep_folder


# In[ ]:





# In[26]:


def get_outputdirs(training_params):

    outputdir = training_params['outputdir']
    sweep_folder = get_sweep_folder(training_params)
    outputdir_sweep = outputdir+'{}/'.format(sweep_folder)

    outputdir_numeric = outputdir_sweep + 'numeric_results/'
    if outputdir_numeric is not None:
        if not os.path.exists(outputdir_numeric):
            os.makedirs(outputdir_numeric)

        if not os.path.exists(outputdir_numeric+'train/'):
            os.makedirs(outputdir_numeric+'train/')
            
        if not os.path.exists(outputdir_numeric+'val/'):
            os.makedirs(outputdir_numeric+'val/')
            
            

    outputdir_modelout = outputdir_sweep + 'model_output/'
    if outputdir_modelout is not None:
        if not os.path.exists(outputdir_modelout):
            os.makedirs(outputdir_modelout)

    outputdir_activation = outputdir_sweep + 'activation_layers/'
    if outputdir_activation is not None:
        if not os.path.exists(outputdir_activation):
            os.makedirs(outputdir_activation)

    outputdir_feature = outputdir_sweep + 'feature_visualization/'
    if outputdir_feature is not None:
        if not os.path.exists(outputdir_feature):
            os.makedirs(outputdir_feature)

    training_params['outputdir_sweep'] = outputdir_sweep
    training_params['outputdir_numeric'] = outputdir_numeric
    training_params['outputdir_modelout'] = outputdir_modelout
    training_params['outputdir_activation'] = outputdir_activation
    training_params['outputdir_feature'] = outputdir_feature

    return training_params


# In[ ]:





# In[27]:


def get_regressor_names(training_params):
    training_params['regressor_names'] = []
    main_task_name = training_params['output_names'][0]
    
    for output_name in training_params['output_names']:
        if output_name == main_task_name:
            training_params['regressor_names'].append(output_name)
        else:
            for input_name in training_params['input_names']:
                training_params['regressor_names'].append(output_name + '-' + input_name)
                
    return training_params


# In[28]:


# training_params['regressor_names']


# In[ ]:





# In[ ]:





# In[29]:


# training_params


# In[30]:


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_conf(df_outputlabel, training_params, weighted=True, fig_name=None, show_plot=False, outputdir=None, log_wandb=False):

    fig, ax = plt.subplots(1,1, figsize=(6, 5), dpi=100, facecolor='white')
    # task_name = task.split('_')[0]

    label = df_outputlabel['label']
    label_est =  df_outputlabel['label_est']
    
    cm = metrics.confusion_matrix(label, label_est)
    
    if weighted:
        cm = cm/cm.sum(axis=1)

    # fig, ax = plt.subplots(figsize=(8,5))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=training_params['race_encoder'].classes_ )

    disp = disp.plot(include_values=True, cmap='Blues', ax=ax, xticks_rotation='60')
    
    fig.tight_layout()
    
    plt.show()
    
    if fig_name is None:
        fig_name = 'cm'

    if log_wandb:
        wandb.log({fig_name: wandb.Image(fig)})

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + fig_name + '.png', facecolor=fig.get_facecolor())

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')
    # ax.set_title(title, fontsize=15)
    
    # return fig


# In[31]:


def plot_one_window(model, training_params, mode='train', fig_name=None, show_plot=False, outputdir=None, log_wandb=False):

    inputdir = training_params['inputdir']
    device = training_params['device']
    # dataloaders, dataset_sizes = get_loaders(inputdir, training_params)
    dataloaders, dataset_sizes, training_params = get_loaders(training_params)

    dataloader = dataloaders[mode]

    data = torch.from_numpy(dataloader.dataset.data)
    feature = torch.from_numpy(dataloader.dataset.feature)
    data = data.to(device).float()
    feature = feature.to(device).float()

    label = dataloader.dataset.label

    #     print(data.size(), feature.size(), label.shape)
    #     print(data, feature, label)

    # meta = dataloader.dataset.meta

    model.eval()
    #     _ = model(data, feature)
    model = model.to(device).float()
    output, feature_out, mu, logvar = model(data, feature)

    data = data.cpu().detach().numpy()
    output = output['reconstruction-PPG'].cpu().detach().numpy()
    feature_out = feature_out['PPG'].cpu().detach().numpy()

    N_ch = feature_out.shape[1]

    fig, axes = plt.subplots(N_ch+2,1,figsize=(5,(N_ch+2)), dpi=60) #   figsize=(width, height)



    i_sample = 0

    t_arr = np.arange(data.shape[-1])/FS_RESAMPLE_DL
    
#     print(feature_out.shape)
#     sys.exit()
    t_arr_feature = np.arange(feature_out.shape[-1]) / ( FS_RESAMPLE_DL / (training_params['stride']**(training_params['n_block']) ) )

    
    axes[0].plot(t_arr, data[i_sample,0,:])
    axes[0].set_ylabel('input')
    axes[0].set_xlim(t_arr.min(), t_arr.max()) # remove the weird white space at the beg and end of the plot


#     for i_ch in range(N_ch):
#         axes[i_ch+1].plot(t_arr_feature, feature_out[i_sample,i_ch,:])
#         axes[i_ch+1].set_ylabel('ch: {}'.format(i_ch))
        
# #         print(t_arr_feature.min(), t_arr_feature.max() )
# #         sys.exit()
#         axes[i_ch+1].set_xlim(t_arr_feature.min(), t_arr_feature.max()) # remove the weird white space at the beg and end of the plot

    axes[-1].plot(t_arr, output[i_sample,:])
    axes[-1].set_ylabel('output')
    axes[-1].set_xlim(t_arr.min(), t_arr.max()) # remove the weird white space at the beg and end of the plot

    for ax in axes:
        ax_no_top_right(ax)

    fig.tight_layout()
    
    if fig_name is None:
        fig_name = 'signals'

    if log_wandb:
        wandb.log({fig_name: wandb.Image(fig)})

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + fig_name + '.png', facecolor=fig.get_facecolor())

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')


# In[32]:


def plot_hidden(mu, logvar):
    # mu dim: (N_batch, N_dim)
    # logvar dim: (N_batch, N_dim)
    
    fig, axes = plt.subplots(N_ch+2,1,figsize=(5,(N_ch+2)), dpi=60) #   figsize=(width, height)


# In[ ]:





# # train_master is a function that train and eval a model using training_params (which stores one HP set)

# In[33]:


debug_auxillary = False

def train_master(training_params):

   # TODO: change all to training_params['xxx'] = get_xxx(training_params)
    training_params = get_outputdirs(training_params) # could be tricky since it changes several keys
    training_params = get_regressor_names(training_params) # may not need this in this task
    training_params['model_out_names'] = get_model_out_names(training_params)
    # training_params['modality_dict'] = get_modality_dict(training_params)
    
    # pprint.pprint(training_params)
    
    df_performance_train = {}
    df_performance_val = {}

    df_outputlabel_train = {}
    df_outputlabel_val = {}

    for task in training_params['model_out_names']:

        df_performance_train[task] = pd.DataFrame()
        df_performance_val[task] = pd.DataFrame()

        df_outputlabel_train[task] = pd.DataFrame()
        df_outputlabel_val[task] = pd.DataFrame()

        
    main_task = training_params['output_names'][0].split('-')[0]
    
    N_CV = training_params['split_name_encoder'].classes_.shape[0] - 1
    for i_CV in range(N_CV):
        
        if 'CV_max' in training_params:
            if i_CV >= training_params['CV_max']:
                continue

        training_params['CV_config']['CV'] = i_CV

        device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')
        print('using device', device)
        print('using model ', training_params['model_name'])

        model = PPG_compressor(training_params=training_params)
        model = model.to(device).float()

        optimizer = torch.optim.Adam(model.parameters(), lr=training_params['learning_rate'], weight_decay=0.01)
        criterion = VAELoss(training_params)
        
        training_params['criterion'] = criterion
        training_params['optimizer'] = optimizer
        training_params['inputdir'] = inputdir

        CV_dict = train_model(model, training_params, trainer, evaler, preder)

        plot_losses(CV_dict, outputdir=training_params['outputdir_sweep'], show_plot=False)

        for task in training_params['model_out_names']:
            if 'domain' in task:
                continue
        
            label_est_val = CV_dict['performance_dict_val']['out_dict'][task]
            label_val = CV_dict['performance_dict_val']['label_dict'][task]

            label_est_train = CV_dict['performance_dict_train']['out_dict'][task]
            label_train = CV_dict['performance_dict_train']['label_dict'][task]
            
            
            mu_arr_train = CV_dict['performance_dict_train']['mu_arr']
            logvar_arr_train = CV_dict['performance_dict_train']['logvar_arr']
            
            mu_arr_val = CV_dict['performance_dict_val']['mu_arr']
            logvar_arr_val = CV_dict['performance_dict_val']['logvar_arr']
            
#             print(mu_arr_train.mean(axis=0))
#             print(logvar_arr_train.mean(axis=0))
#             print(mu_arr_val.mean(axis=0))
#             print(logvar_arr_val.mean(axis=0))
            
#             sys.exit()
            
            data_saver(mu_arr_train, 'mu', training_params['outputdir_numeric']+'train/')
            data_saver(logvar_arr_train, 'logvar', training_params['outputdir_numeric']+'train/')
            
            data_saver(mu_arr_val, 'mu', training_params['outputdir_numeric']+'val/')
            data_saver(logvar_arr_val, 'logvar', training_params['outputdir_numeric']+'val/')

            
#             if 'domain' in task:
#                 np.argmax(a, axis=1)
            
            
#             # rescale the label after making estimations
#             if 'perc' in training_params['output_names'][0]:
#                 i_meta = training_params['meta_names'].index('EEavg_est')
# #                 print(CV_dict['performance_dict_train']['meta_arr'], CV_dict['performance_dict_train']['meta_arr'].shape)
#                 meta_train = CV_dict['performance_dict_train']['meta_arr'][:, i_meta]
#                 meta_val = CV_dict['performance_dict_val']['meta_arr'][:, i_meta]

#                 label_train = label_train*meta_train
#                 label_val = label_val*meta_val
#                 label_est_train = label_est_train*meta_train
#                 label_est_val = label_est_val*meta_val
#             elif 'weighted' in training_params['output_names'][0]:
#                 i_meta = training_params['meta_names'].index('weight')
#                 meta_train = CV_dict['performance_dict_train']['meta_arr'][:, i_meta]
#                 meta_val = CV_dict['performance_dict_val']['meta_arr'][:, i_meta]

#                 label_train = label_train*meta_train
#                 label_val = label_val*meta_val
#                 label_est_train = label_est_train*meta_train
#                 label_est_val = label_est_val*meta_val

                
#             print(label_val, label_est_val)
#             sys.exit()
            
            # get performance df for training and testing dataset
#             df_performance_train[task] = df_performance_train[task].append( get_df_performance(label_train, label_est_train, i_CV, task), ignore_index=True )

#             df_performance_train[task].to_csv(training_params['outputdir_numeric']  + 'df_performance_train_{}.csv'.format(task), index=False)

#             df_outputlabel_train[task] = df_outputlabel_train[task].append(
#                 pd.DataFrame( {
#                 'label_est': label_est_train,
#                 'label': label_train,
#                 'CV': [i_CV]*label_train.shape[0],
#                 'task': [task]*label_train.shape[0]
#                 }), ignore_index=True )

#             df_outputlabel_train[task].to_csv(training_params['outputdir_numeric']  + 'df_outputlabel_train_{}.csv'.format(task), index=False)

            # df_performance_val[task] = df_performance_val[task].append( get_df_performance(label_val, label_est_val, i_CV, task), ignore_index=True )
            # df_performance_val[task].to_csv(training_params['outputdir_numeric']  + 'df_performance_val_{}.csv'.format(task), index=False)

#             df_outputlabel_val[task] = df_outputlabel_val[task].append(
#                 pd.DataFrame( {
#                 'label_est': label_est_val,
#                 'label': label_val,
#                 'CV': [i_CV]*label_val.shape[0],
#                 'task': [task]*label_val.shape[0]
#                 }), ignore_index=True )

#             df_outputlabel_val[task].to_csv(training_params['outputdir_numeric']  + 'df_outputlabel_val_{}.csv'.format(task), index=False)




            # plot performance training and testing dataset
            if (main_task not in task) and (debug_auxillary==False):
                continue
            
            
            plot_one_window(model, training_params, mode='train', fig_name='signals_CV{}'.format(i_CV), outputdir=training_params['outputdir_modelout']+'train/', show_plot=False)

            plot_one_window(model, training_params, mode='val', fig_name='signals_CV{}'.format(i_CV), outputdir=training_params['outputdir_modelout']+'val/', show_plot=False)

            # sys.exit()

            # plot_conf(df_outputlabel_train[task], training_params,  fig_name='cm_train', show_plot=False, outputdir=training_params['outputdir_modelout'])
            # plot_conf(df_outputlabel_val[task], training_params,  fig_name='cm_val', show_plot=False, outputdir=training_params['outputdir_modelout'])
            # sys.exit()

            # plot_regression(df_outputlabel_train[task], df_performance_train[task], task, fig_name='regression_train_{}'.format(task), show_plot=False, outputdir=outputdir+'model_output/')
#             plot_BA(df_outputlabel_train[task], task, fig_name='BA_train_{}'.format(task), show_plot=False, outputdir=outputdir+'model_output/')

            # plot_regression(df_outputlabel_val[task], df_performance_val[task], task, fig_name='regression_val_{}'.format(task), show_plot=False, outputdir=outputdir_modelout, log_wandb=training_params['wandb'])
#             plot_BA(df_outputlabel_val[task], task, fig_name='BA_val_{}'.format(task), show_plot=False, outputdir=outputdir+'model_output/')

#             plot_output(df_outputlabel_train[task], task, fig_name = 'outputINtime_train_{}'.format(task), show_plot=False, outputdir=outputdir_modelout)
        
        # check_featuremap(model, training_params, mode='worst', fig_name = 'DL_activation_{}_'.format(i_CV), outputdir=outputdir+'activation_layers_worst/{}/'.format(i_CV), show_plot=False)
        # check_featuremap(model, training_params, mode='best', fig_name = 'DL_activation_{}_'.format(i_CV), outputdir=outputdir+'activation_layers_best/{}/'.format(i_CV), show_plot=False)
        
        del model
        torch.cuda.empty_cache()


    # for task in training_params['model_out_names']:
    #     if main_task not in task:
    #         continue
#         if task!=main_task:
#             continue
#         plot_regression_all_agg(df_outputlabel_train[task], df_performance_train[task], fig_name='LinearR_agg_train_{}'.format(task), show_plot=False, outputdir=outputdir_modelout, log_wandb=training_params['wandb'])
#         plot_BA(df_outputlabel_train[task], task, fig_name='BA_train_{}'.format(task), show_plot=False, outputdir=outputdir_modelout, log_wandb=training_params['wandb'])

#         plot_regression_all_agg(df_outputlabel_val[task], df_performance_val[task], fig_name='LinearR_agg_val_{}'.format(task), show_plot=False, outputdir=outputdir_modelout, log_wandb=training_params['wandb'])
#         plot_BA(df_outputlabel_val[task], task, fig_name='BA_val_{}'.format(task), show_plot=False, outputdir=outputdir_modelout, log_wandb=training_params['wandb'])

        # plot_output(df_outputlabel_val[task], task, fig_name = 'outputINtime_val_{}'.format(task),  show_plot=False, outputdir=outputdir_modelout)

#     plot_BA(df_outputlabel_val[main_task], main_task, fig_name='BA_val_{}'.format(main_task), show_plot=False, outputdir=outputdir+'model_output/', log_wandb=training_params['wandb'])
#     plot_regression_all_agg(df_outputlabel_val[main_task], df_performance_val[main_task], outputdir=outputdir+'model_output/', show_plot=False, log_wandb=training_params['wandb'])

    # log metrices on wnadb
    if training_params['wandb']==True:
        
#         label = df_outputlabel_val[main_task]['label'].values
#         label_est = df_outputlabel_val[main_task]['label_est'].values
# #         print(label.shape, label)
# #         print(label_est.shape, label_est)
    
#         PCC = get_PCC(label, label_est)
#         Rsquared = get_CoeffDeterm(label, label_est)
#         MAE, _ = get_MAE(label, label_est)
#         RMSE = get_RMSE(label, label_est)
#         MAPE, _ = get_MAPE(label, label_est)

        

        wandb.log(
            {
                'val_total_loss': CV_dict['df_losses_val']['total'].values[-1],
            })


# In[ ]:





# In[ ]:





# In[ ]:





# # train_sweep is a function that wandb calls when changing to a new HP set

# In[34]:


def train_sweep(config=None):

    with wandb.init(config=config, reinit=True, dir=outputdir):

        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        
        print(config)
        
        # init the model
        for key in config.keys():
            if key=='loss_weights':
                training_params[key]['auxillary_task'] = config[key]
            else:
                training_params[key] = config[key]

        train_master(training_params)


# # master wandb that select the HP set and ask the train_sweep to train and eval a model

# In[35]:


tz_NY = pytz.timezone('America/New_York') 
datetime_start = datetime.now(tz_NY)
print("start time:", datetime_start.strftime("%Y-%b-%d %H:%M:%S"))


# In[ ]:





# In[ ]:





# In[36]:


if training_params['wandb']:
    print('sweeping for:', sweep_name)
    # get the config of current sweep
    sweep_config = training_params['sweep_config']    
    # get the config of current sweep
    sweep_id = wandb.sweep(sweep_config, entity='inanlab', project='[PPG_compress] stage3_'+training_params['sweep_name'])
    wandb.agent(sweep_id, train_sweep)
else:
    train_master(training_params)


# In[37]:


# kernel_size = 10
# # m = nn.Conv1d(16, 33, kernel_size=kernel_size, stride=1)
# input = torch.randn(20, 16, 1)
# # output = m(input)
# print(input.size())
# # print
# m = torch.nn.ConvTranspose1d(16, 33, kernel_size, stride=1,)
# output = m(input)
# print(output.size())


# In[ ]:





# In[ ]:





# In[ ]:


# import torch

# num_of_gpus = torch.cuda.device_count()
# print(num_of_gpus)


# In[ ]:





# In[ ]:





# In[ ]:



datetime_end = datetime.now(tz_NY)
print("end time:", datetime_end.strftime("%Y-%b-%d %H:%M:%S"))

duration = datetime_end-datetime_start
duration_in_s = duration.total_seconds()
days    = divmod(duration_in_s, 86400)        # Get days (without [0]!)
hours   = divmod(days[1], 3600)               # Use remainder of days to calc hours
minutes = divmod(hours[1], 60)                # Use remainder of hours to calc minutes
seconds = divmod(minutes[1], 1)               # Use remainder of minutes to calc seconds
print("Time between dates: %d days, %d hours, %d minutes and %d seconds" % (days[0], hours[0], minutes[0], seconds[0]))


# In[ ]:





# In[ ]:





# In[ ]:





# # wrap it up

# In[ ]:



if training_params['wandb']:
    wandb.finish()


# # time it

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


sys.exit()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




