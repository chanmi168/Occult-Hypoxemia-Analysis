import numpy as np
import os
import math
from math import sin

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics

import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
import matplotlib.ticker as plticker

import torch

import wandb

from filters import *
from setting import *
from preprocessing import *
from plotting_tools import *
from evaluate import *

# from models import *
# from models_resnet import *
# from dataset_util import *

    
    
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

def ax_conditioning_regression(ax, df_outputlabel, label_range=None, subject_id='All', fontsize=15):

    task = df_outputlabel['task'].unique()[0]
    props = dict(boxstyle='round,pad=0.7', facecolor='white', edgecolor='black', alpha=0.7)

    task_name = task.split('-')[0].split('_')[0]

    label = df_outputlabel['label'].values
    label_est = df_outputlabel['label_est'].values
    
    label_range_sub = [my_floor(label.min()), my_ceil(label.max())]
    if label_range is None:
        label_range = label_range_sub
        

    N_sub = len(df_outputlabel['CV'].unique())
    N_samples = df_outputlabel.shape[0]
    t_dur = N_samples*3/60

    PCC = get_PCC(label, label_est)
    Rsquared = get_CoeffDeterm(label, label_est)
    MAE, MAE_std = get_MAE(label, label_est)
    RMSE = get_RMSE(label, label_est)
    MAPE, MAPE_std = get_MAPE(label, label_est)
    
    title_str = '[{}]\n{} range: {:.1f}-{:.1f} {}'.format(subject_id, task.split('_')[0], label_range_sub[0], label_range_sub[1], unit_dict[task_name])
    textstr = 'RMSE={:.2f} {}\nMAE={:.2f} {}\nMAPE={:.2f} {}\nPCC={:.2f}\nR2={:.2f}\nN_sub={}\nN_samples={}\nduration={:.2f} min'.format(
        RMSE, unit_dict[task_name], MAE, unit_dict[task_name],MAPE*100, '%',
        PCC, Rsquared,
        N_sub, N_samples, t_dur)
    
    ax.set_title(title_str, fontsize=fontsize+1)

#     ax.set_ylabel('Est. {} - Ref. {}\n[{}]'.format(task_name, task_name, unit_dict[task_name]), fontsize=fontsize)
#     ax.set_xlabel('Avg of Est. and Ref. {}\n[{}]'.format(task_name, unit_dict[task_name]), fontsize=fontsize)

    ax.set_ylabel('Est. {}\n[{}]'.format(task_name, unit_dict[task_name]), fontsize=fontsize)
    ax.set_xlabel('Ref. {}\n[{}]'.format(task_name, unit_dict[task_name]), fontsize=fontsize)

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=0.8)

    ax.plot( label_range,label_range , color='gray', alpha=0.5, linestyle='--')

    # place a text box in bottom right in axes coords
    ax.text(0.05, 0.65, textstr, transform=ax.transAxes, fontsize=fontsize-7,
            verticalalignment='bottom', horizontalalignment='left', bbox=props)

    ax.set_ylim(label_range)
    ax.set_xlim(label_range)

    ax_no_top_right(ax)

    
def plot_regression(df_outputlabel_val, df_performance_val, task, fig_name=None, show_plot=False, outputdir=None, log_wandb=False):
#     print('regression')
    fig, ax = plt.subplots(1,1, figsize=(5, 5), dpi=100, facecolor='white')
    props = dict(boxstyle='round,pad=0.7', facecolor='white', edgecolor='black', alpha=0.7)
    fontsize = 16
    alpha=0.1

#     task_name = task.split('_')[0]
    task_name = task.split('-')[0].split('_')[0]

    sc = sns.scatterplot(data=df_outputlabel_val, x='label', y='label_est', hue='CV',  ec="None", palette=subject_palette, alpha=alpha, s=50, marker='o', ax=ax)
    # sc.set_edgecolor("none")

    
    
    ax_conditioning_regression(ax, df_outputlabel_val)

    
    
#     label_range = [my_floor(df_outputlabel_val['label'].values.min()), my_ceil(df_outputlabel_val['label'].values.max())]

#     N_sub = len(df_outputlabel_val['CV'].unique())
#     N_samples = df_outputlabel_val.shape[0]
#     t_dur = N_samples*3/60
    
    
#     PCC = get_PCC(df_outputlabel_val['label'].values, df_outputlabel_val['label_est'].values)
#     Rsquared = get_CoeffDeterm(df_outputlabel_val['label'].values, df_outputlabel_val['label_est'].values)
#     MAE, MAE_std = get_MAE(df_outputlabel_val['label'].values, df_outputlabel_val['label_est'].values)
#     RMSE = get_RMSE(df_outputlabel_val['label'].values, df_outputlabel_val['label_est'].values)
#     MAPE, MAPE_std = get_MAPE(df_outputlabel_val['label'].values, df_outputlabel_val['label_est'].values)

#     title_str = '{} range: {:.1f}-{:.1f} {}'.format(task.split('_')[0], label_range[0], label_range[1], unit_dict[task_name])
#     textstr = 'RMSE={:.2f} {}\nMAE={:.2f} {}\nMAPE={:.2f} {}\nPCC={:.2f}\nR2={:.2f}\nN_sub={}\nN_samples={}\nduration={:.2f} min'.format(
#         RMSE, unit_dict[task_name], MAE, unit_dict[task_name],MAPE*100, '%',
#         PCC, Rsquared,
#         N_sub, N_samples, t_dur)
# #     title_str = '{} range: {:.1f}-{:.1f} {}'.format(task.split('_')[0], label_range[0], label_range[1], unit_dict[task_name])
# #     textstr = 'RMSE={:.2f} {}\nMAE={:.2f} {}\nMAPE={:.2f} {}\nPCC={:.2f}\nR2={:.2f}\nN_sub={}\nN_samples={}\nduration={:.2f} min'.format(
# #         df_performance_val['rmse'].mean(), unit_dict[task_name], df_performance_val['mae'].mean(), unit_dict[task_name], df_performance_val['mape'].mean()*100, '%',
# #         PCC, Rsquared,
# #         N_sub, N_samples, t_dur)

#     # ax_conditioning(ax, plt_title, fontsize, props, 'RMSE={:.2f}%'.format(df_performance_val['rmse'].mean()))

#     ax.set_title(title_str, fontsize=fontsize+5)

#     ax.set_ylabel('{} estimated ({})'.format(task_name, unit_dict[task_name]), fontsize=fontsize)
#     ax.set_xlabel('{} label ({})'.format(task_name, unit_dict[task_name]), fontsize=fontsize)


# #     major_ticks = np.arange(label_range[0],label_range[1]+1,5)
# #     minor_ticks = np.arange(label_range[0],label_range[1]+1)
    
# #     ax.set_xticks(major_ticks)
# #     ax.set_xticks(minor_ticks, minor=True)
# #     ax.set_yticks(major_ticks)
# #     ax.set_yticks(minor_ticks, minor=True)
#     # Or if you want different settings for the grids:
#     ax.grid(which='minor', alpha=0.3)
#     ax.grid(which='major', alpha=0.8)


#     ax.plot( label_range,label_range , color='gray', alpha=0.5, linestyle='--')
    ax.legend(frameon=True, fontsize=fontsize-7, bbox_to_anchor=(1.01, 1))


#     # # place a text box in bottom right in axes coords
#     ax.text(0.05, 0.65, textstr, transform=ax.transAxes, fontsize=fontsize-7,
#             verticalalignment='bottom', horizontalalignment='left', bbox=props)

#     ax.set_ylim(label_range)
#     ax.set_xlim(label_range)

#     ax_no_top_right(ax)

    fig.tight_layout()
    
    
    if fig_name is None:
        fig_name = 'regression_analysis'
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
        
#     return fig

# def BA_plotter(ax, df, mode):
def plot_BA(df_outputlabel_val, task, fig_name=None, show_plot=False, outputdir=None, log_wandb=False):
    fig, ax = plt.subplots(1,1, figsize=(6, 5), dpi=100, facecolor='white')
    task_name = task.split('_')[0]

    label = df_outputlabel_val['label']
    label_est =  df_outputlabel_val['label_est']

    fontsize = 16
    alpha=0.1
        # plot Bland-Altman plot

    data1     = label
    data2     = label_est
    mean      = np.mean([data1, data2], axis=0)
    diff      = data2 - data1                    # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    diff_max = np.abs(diff).max()
    #     # plot running average of the error along x-axis (TBD)
    #     plot_smooth_err(ax, mean, diff)


    ax.axhline(md,           color='gray', linestyle='--')
    ax.axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax.axhline(md - 1.96*sd, color='gray', linestyle='--')


    task_name = task.split('-')[0].split('_')[0]

    for subject_id in df_outputlabel_val['CV'].unique():

        df_sub = df_outputlabel_val[df_outputlabel_val['CV']==subject_id]
        label = df_sub['label']
        label_est =  df_sub['label_est']

        mean = np.mean([label_est, label], axis=0)
        diff = label_est - label

        ax.scatter(x=mean,y=diff, alpha=alpha, color=subject_palette[subject_id])


    ax.set_ylim(-diff_max,diff_max)


    ax.set_ylabel('Estimated {} - Reference {}\n[{}]'.format(task_name, task_name, unit_dict[task_name]), fontsize=fontsize)
    ax.set_xlabel('Average of Estimated and Reference {} [{}]'.format(task_name, unit_dict[task_name]), fontsize=fontsize)

    #     ax.xaxis.grid(True, which='major')
    #     ax.xaxis.grid(True, which='minor', alpha=0.4)
    #     ax.yaxis.grid(True, which='major')
    #     ax.yaxis.grid(True, which='minor', alpha=0.4)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)


    #     ax.figure.set_size_inches(14, 7)
    # ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax_no_top_right(ax)


    fig.tight_layout()
    
    if fig_name is None:
        fig_name = 'BA_analysis'
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

#     return fig

def get_df_performance(label, label_est, subject_id, task, verbose=False):   

    conf_matrix = metrics.confusion_matrix(label, label_est)

    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    
    if verbose:
        print('True Positives:', TP)
        print('True Negatives:', TN)
        print('False Positives:', FP)
        print('False Negatives:', FN)
    
    # calculate accuracy
    conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))
    
    # calculate mis-classification
    conf_misclassification = 1- conf_accuracy
    
    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))
    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))
    
    # calculate precision
    conf_precision = (TN / float(TN + FP))
    # calculate f_1 score
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
    
    if verbose:
        print('-'*50)
        print(f'Accuracy: {round(conf_accuracy,2)}') 
        print(f'Mis-Classification: {round(conf_misclassification,2)}') 
        print(f'Sensitivity: {round(conf_sensitivity,2)}') 
        print(f'Specificity: {round(conf_specificity,2)}') 
        print(f'Precision: {round(conf_precision,2)}')
        print(f'f_1 Score: {round(conf_f1,2)}')
    
    df_performance = pd.DataFrame({
        'CV': [subject_id],
        'task': [task],
        'accuracy': [conf_accuracy],
        'sensitivity': [conf_sensitivity],
        'specificity': [conf_specificity],
        'precision': [conf_precision],
        'F1': [conf_f1],
    })

    return df_performance
# def get_df_performance(label, label_est, subject_id, task):   
#     rmse = np.sqrt(mean_squared_error(label, label_est))

#     mae, _ = get_MAE(label, label_est)
#     mape, _ = get_MAPE(label, label_est)

#     Rsquared = get_CoeffDeterm(label=label, predictions=label_est)
#     PCC = get_PCC(label=label, est=label_est)

#     df_performance = pd.DataFrame({
#         'CV': [subject_id],
#         'task': [task],
#         'Rsquared': [Rsquared],
#         'PCC': [PCC],
#         'rmse': [rmse],
#         'mae': [mae],
#         'mape': [mape],
#     })

#     return df_performance


        
def inspect_labels(ax, df, y_min, y_max, annotate=True, fontsize=15):
    task_label = df['task'].values    

    ts = df['time'].values

    TASKS = np.unique(task_label)
#     print(TASKS)

    for i, task_name in enumerate(FS_tasks):
        if task_name not in TASKS:
            continue

        indices = np.where(task_label==task_name)[0]

        task_start = ts[indices[0]]
        task_end = ts[indices[-1]]

        task_id = FS_tasks.index(task_name)
#         print(task_name, task_start, task_end)
        ax.fill_between( np.array( [ task_start, task_end ] ),
                             y_min * np.array( [1, 1] ),
                             y_max * np.array( [1, 1] ),
                             facecolor = color_dict[color_names[task_id+1]],
                             alpha = 0.1) 
        if annotate:
            annotate_alpha = 0.8

            text = ax.annotate(task_name, (task_start/2+task_end/2, y_max), fontsize=fontsize, color='black', horizontalalignment='center', verticalalignment='bottom', rotation=45)
            text.set_alpha(annotate_alpha)

def plot_output(df_outputlabel, task, fig_name=None, show_plot=False, outputdir=None):
#     print('output')

    # fig, (ax, ax2) = plt.subplots(2,1, figsize=(20,4), dpi=80)
    fig, ax = plt.subplots(1,1, figsize=(df_outputlabel.shape[0]/25,3), dpi=100)

    x_sample = np.arange(df_outputlabel.shape[0])
    ax.plot(x_sample, df_outputlabel['label_est'], label='estimated', alpha=0.6)
    ax.plot(x_sample, df_outputlabel['label'], label='label', alpha=0.6)
    # ax2.plot(x_sample, df_outputlabel_val['CV'])

#     task_name = task.split('_')[0]
    task_name = task.split('-')[0].split('_')[0]

    ax.set_xlim(x_sample.min(), x_sample.max()) # remove the weird white space at the beg and end of the plot

    ax.set_xlabel('sample')
    ax.set_ylabel('{}\n[{}]'.format(task_name, unit_dict[task_name]))
    ax.legend(loc='upper right')
    
    fig.tight_layout()

    if fig_name is None:
        fig_name = 'outputINtime'

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + fig_name + '.png', facecolor=fig.get_facecolor())

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')

        
        
        
def plot_regression_partial(ax, df_outputlabel, subject_id_plt, outputdir=None, show_plot=False, log_wandb=False):

    props = dict(boxstyle='round,pad=0.7', facecolor='white', edgecolor='black', alpha=0.7)
    fontsize = 16

    N_beats_val = 0
    
    label = df_outputlabel['label'].values
    label_range = [my_floor(label.min()), my_ceil(label.max())]

    for subject_id in df_outputlabel['CV'].unique():
        
        df_outputlabel_sub = df_outputlabel[df_outputlabel['CV']==subject_id]
        marker = marker_dict['circle']

        label = df_outputlabel_sub['label'].values
        label_est = df_outputlabel_sub['label_est'].values


        if subject_id == subject_id_plt:
            color = subject_palette[subject_id]
            alpha=0.4
            ax.set_title('{}'.format(subject_id), fontsize=fontsize+5)
        else:
            color = 'gray'
            alpha=0.05

        ax.scatter(label, label_est, alpha=alpha, color=color, marker=marker)
        
        
    df_outputlabel_sub = df_outputlabel[df_outputlabel['CV']==subject_id_plt]
    ax_conditioning_regression(ax, df_outputlabel_sub, label_range, subject_id=subject_id_plt)
    

def plot_regression_all_agg(df_outputlabel, df_performance, fig_name=None, outputdir=None, show_plot=False, log_wandb=False):

    label = df_outputlabel['label'].values
#     label_est = df_outputlabel['label_est'].values
    label_range = [my_floor(label.min()), my_ceil(label.max())]

    
    fig = plt.figure(figsize=(20, 20), dpi=100, facecolor='white')

    for k, subject_id_plt in enumerate(df_outputlabel['CV'].unique()):
#         row = k//5
#         col = k%5

        ax = fig.add_subplot(5,5,k+1)
        plot_regression_partial(ax, df_outputlabel, subject_id_plt, outputdir=None, show_plot=False, log_wandb=False)
#         ax.set_ylim(label_range)
#         ax.set_xlim(label_range)
        
    # plot all regression in one plot
    ax = fig.add_subplot(5,5,k+2)
    alpha=0.3
    fontsize=16
    sc = sns.scatterplot(data=df_outputlabel, x='label', y='label_est', hue='CV',  ec="None", palette=subject_palette, alpha=alpha, s=50, marker='o', ax=ax)
    ax_conditioning_regression(ax, df_outputlabel, label_range)
#     ax.set_ylim(label_range)
#     ax.set_xlim(label_range)
    
    ax.legend(frameon=True, fontsize=fontsize-7, bbox_to_anchor=(1.01, 1))

    fig.tight_layout()


    if fig_name is None:
        fig_name = 'LinearR_agg'
#     else:
#         fig_name = 'LinearR_agg'+fig_name

    if log_wandb:
        wandb.log({fig_name: wandb.Image(fig)})

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + fig_name+'.png', facecolor=fig.get_facecolor())

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')
        
        
        
        
# def check_featuremap(model, training_params, mode='worst', fig_name=None, show_plot=False, outputdir=None, log_wandb=False):

#     inputdir = training_params['inputdir']
#     device = training_params['device']
#     dataloaders, dataset_sizes = get_loaders(inputdir, training_params)
#     dataloader = dataloaders['val']
# #     device = torch.device('cuda:{}'.format('cpu'))
# #     device = torch.device('cpu')
# #     model = model.to(device).float()

#     # 1. set up the hook
#     activation = {}
#     def get_activation(name):
#         def hook(model, input, output):
#             activation[name] = output.detach()
#         return hook

#     data = torch.from_numpy(dataloader.dataset.data)
#     feature = torch.from_numpy(dataloader.dataset.feature)
#     data = data.to(device).float()
#     feature = feature.to(device).float()

#     label = dataloader.dataset.label

# #     print(data.size(), feature.size(), label.shape)
# #     print(data, feature, label)
    
#     meta = dataloader.dataset.meta

#     #     model_hooking(model, training_params)

#     model_name = training_params['model_name']
    
#     layer_names = []

#     # 3. define the layers that I want to look at
#     if model_name=='FeatureExtractor_CNN2':
#         key = list(model.feature_extractors.keys())[0]
#         model.feature_extractors[key].layer1.register_forward_hook(get_activation('layer1'))
#         model.feature_extractors[key].layer2.register_forward_hook(get_activation('layer2'))
#         model.feature_extractors[key].layer3.register_forward_hook(get_activation('layer3'))
#         model.feature_extractors[key].layer4.register_forward_hook(get_activation('layer4'))
#         model.regressors.EE_cosmed.fc1.register_forward_hook(get_activation('fc1'))
#         model.regressors.EE_cosmed.fc2.register_forward_hook(get_activation('fc2'))

#         layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'fc1', 'fc2']

#     elif model_name=='FeatureExtractor_CNN':
#         key = list(model.feature_extractors.keys())[0]
#         model.feature_extractors[key].layer1.register_forward_hook(get_activation('layer1'))
#         model.feature_extractors[key].layer2.register_forward_hook(get_activation('layer2'))
#         model.feature_extractors[key].layer3.register_forward_hook(get_activation('layer3'))
#         model.feature_extractors[key].layer4.register_forward_hook(get_activation('layer4'))
#         model.regressors.EE_cosmed.fc1.register_forward_hook(get_activation('fc1'))
#         model.regressors.EE_cosmed.fc2.register_forward_hook(get_activation('fc2'))
#         layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'fc1', 'fc2']

#     elif model_name=='ResNet1D':
        
# #         print(list(model.feature_extractors.keys()))
# #         sys.exit()
        
#         try:
#             for input_name in list(model.feature_extractors.keys()):
#                 model.feature_extractors[input_name].basicblock_list[-1].register_forward_hook(get_activation(input_name+'_layer_last'))
#                 model.feature_extractors[input_name].final_ch_pooling.register_forward_hook(get_activation(input_name+'_layer_pool'))
#                 layer_names = layer_names + [input_name+'_layer_last', input_name+'_layer_pool']
#         except:
#             for input_name in list(model.feature_extractors.keys()):
#                 model.feature_extractors[input_name].basicblock_list[-1].register_forward_hook(get_activation(input_name+'_layer_last'))
#                 layer_names = layer_names + [input_name+'_layer_last']
#             for regressor_name in list(model.regressors.keys()):
#                 model.regressors[regressor_name].ch_pooling.register_forward_hook(get_activation(regressor_name+'_layer_pooling'))
#                 layer_names = layer_names + [regressor_name+'_layer_pooling']

#     elif model_name == 'ResNet1D_LSTM':
        
#         for input_name in list(model.feature_extractors.keys()):
#             model.feature_extractors[input_name].basicblock_list[-1].register_forward_hook(get_activation(input_name+'_layer_last'))
#             layer_names = layer_names + [input_name+'_layer_last']

            
# #         for regressor_name in list(model.regressors.keys()):
# #             model.regressors[regressor_name].relu1.register_forward_hook(get_activation(regressor_name+'_relu1'))
# #             layer_names = layer_names + [regressor_name+'_relu1']
        
# #     print(layer_names)
# #         for input_name in list(model.feature_extractors.keys()):

#     # 4. pass the data to the model and the hook will take care of the rest (output stored in activation)
# #     output = model(data, feature)


#     model.eval()
# #     _ = model(data, feature)



#     output = model(data, feature)


# #     output = output.cpu().detach().numpy()
    
#     label = {output_name: label[:, [training_params['output_names'].index( output_name.split('-')[0] )]] for output_name in output.keys()}



# #     print(data, feature, output, label)

#     main_task = training_params['output_names'][0]
# #     error_abs = np.abs(output[main_task].cpu().detach().numpy() - label[main_task]).squeeze()
# #     print(output[main_task].size(), label[main_task].shape)
    
    
#     error_abs = np.abs(output[main_task].cpu().detach().numpy().squeeze() - label[main_task].squeeze()).squeeze()
# #     print(error_abs.shape)
# #     sys.exit()

#     if model_name == 'ResNet1D_LSTM':
#         error_abs = error_abs.mean(axis=-1)
# #     print(error_abs.shape)
    
#     if mode=='worst':
#         i_sample = np.argmax(error_abs)
#     if mode=='best':
#         i_sample = np.argmin(error_abs)
#     if mode=='random':
#         # 2. check one sample only
#         N_samples = dataloader.dataset.data.shape[0]
#         np.random.seed(0)
#         i_sample = np.random.randint(N_samples)

        
# #     print(N_samples, i_sample)
# #     sys.exit()
#     [subject_id, task_id] = meta[i_sample, :2]
# #     print(out, label, meta)
    



# #     sys.exit()
#     data = data.cpu().detach().numpy()
    

#     # 5. organize these activation layers
#     data_layers = {}
#     for layer_name in layer_names:
#         data_layers[layer_name] = activation[layer_name].cpu().detach().numpy()



#     for layer_name in layer_names:
# #         print(layer_name, )
        
#         data_layer = data_layers[layer_name]
        
#         if len(data_layer.shape)<3:
#             data_layer = data_layer[:,:,None]

# #         print(data_layer.shape)
        
#         N_sigs = len(training_params['input_names'])

#         fig, axes = plt.subplots(data_layer.shape[1]+1,1, figsize=(10, data_layer.shape[1]+1), dpi=60, gridspec_kw = {'wspace':0, 'hspace':0}, facecolor='white', constrained_layout=True)
#         fontsize = 13

#         FS_RESAMPLE_DL = training_params['FS_RESAMPLE_DL']
#         t_arr = np.arange(data.shape[-1])/FS_RESAMPLE_DL

# #         for i, input_name in enumerate(training_params['input_names']):

#         # 1. plot one physio sig at the top column
#         ax = axes[0]

        

#         if '-' in layer_name:
#             sig_name = layer_name.split('-')[1]
#         else:
#             sig_name = layer_name
            
            
#         if 'ECG' in sig_name:
#             unit = unit_dict['ecg']
#             input_name = sig_name.split('_layer')[0]
#             i_sig = training_params['input_names'].index(input_name)
# #             i_sig = training_params['input_names'].index('ECG')
#         elif ('accel' in sig_name) or ('scg' in sig_name):
#             unit = unit_dict['accel']
#             input_name = sig_name.split('_layer')[0]
#             i_sig = training_params['input_names'].index(input_name)
# #             i_sig = training_params['input_names'].index('accelZ')
#         elif 'ppg' in sig_name:
#             unit = unit_dict['ppg']
#             input_name = sig_name.split('_layer')[0]
#             i_sig = training_params['input_names'].index(input_name)
# #             i_sig = training_params['input_names'].index('ppg_g_1')



#         sig_name = training_params['input_names'][i_sig]
        
# #         print(layer_name, i_sig)
# #         sys.exit()
        
#         ax.plot(t_arr, data[i_sample,i_sig,:])
#         ax.set_xlim(t_arr.min(), t_arr.max()) # remove the weird white space at the beg and end of the plot
#         ax.set_ylabel('{}\n[{}]'.format(sig_name, unit), fontsize=fontsize,rotation = 0,  va='center', ha='center',  labelpad=20)
        
#         ax.set_title('[{}] {}\nAE={:.2f} [{}]'.format(subject_id, tasks_dict_reversed[task_id], error_abs[i_sample], unit_dict[main_task.split('_')[0]]), fontsize=fontsize)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(False)
#         ax.set_xticklabels([])

#         # 2. next, plot the feature map 
# #         t_arr = np.arange(data_layer.shape[-1]) / ( FS_RESAMPLE_DL / (2**(training_params['n_block_macro'] - 1)) )
#         t_arr = np.arange(data_layer.shape[-1]) / ( FS_RESAMPLE_DL / (training_params['stride'] **(training_params['n_block_macro'] - 1)) )
# #         t_arr = np.arange(data_layer.squeeze().shape[-1]) / ( FS_RESAMPLE_DL / (training_params['stride'] **(training_params['n_block_macro'] - 1)) )
        
#         for j_filter, ax in enumerate(axes[1:]):

#             if j_filter!=len(axes[1:])-1:
#                 ax.set_xticklabels([])

# #             print( data_layer[i_sample, j_filter, :].shape)
            
#             ax.plot(t_arr, data_layer[i_sample, j_filter, :].squeeze(), alpha=1)
# #             ax.plot(t_arr, data_layer[i_sample, j_filter, :].T, alpha=1)

#             ax.set_xlim(t_arr.min(), t_arr.max()) # remove the weird white space at the beg and end of the plot

#             # remove some borders (top and right)
#             ax.spines['right'].set_visible(False)
#             ax.spines['top'].set_visible(False)
#             ax.spines['bottom'].set_visible(False)

#             ax.set_ylabel('filter {}'.format(j_filter), fontsize=fontsize,rotation = 0,  va='center', ha='center',  labelpad=20)
#             # set tick font size
#             ax.tick_params(axis='both', which='major', labelsize=fontsize*0.8)

#             if j_filter==len(axes[1:])-1:
#                 ax.set_xlabel('time (sec)', fontsize=fontsize)
#                 ax.spines['bottom'].set_visible(True)

# #         fig.subplots_adjust(wspace=0, hspace=0)

# #         fig.tight_layout()
        
#         if fig_name is None:
#             fig_name = 'DL_activation'
# #             print('hihi', sig_name)
# #         fig_name = 'DL_activation_'+sig_name

#         if log_wandb:
#             wandb.log({fig_name: wandb.Image(fig)})

#         if outputdir is not None:
#             if not os.path.exists(outputdir):
#                 os.makedirs(outputdir)
#             fig.savefig(outputdir + fig_name+ layer_name + '.png', facecolor=fig.get_facecolor())

#         if show_plot == False:
#             plt.close(fig)
#             pyplot.close(fig)
#             plt.close('all')
            
            
# #         return fig