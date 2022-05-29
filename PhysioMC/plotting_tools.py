import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd
import os

import matplotlib
import matplotlib.pyplot as plt
# plt.style.use('dark_background')
# matplotlib.rc( 'savefig', facecolor = 'white' )
# matplotlib.rc( 'savefig', facecolor = 'black' )
from matplotlib.gridspec import GridSpec

from scipy.io import loadmat
import scipy
from scipy import signal
from scipy.fftpack import fft, ifft
pd.set_option('display.max_columns', 500)

import random
from random import randint
random.seed(0)

from setting import *

# import wandb



'''
Restricted Cubic Splines
For Pandas/Python
See https://apwheele.github.io/MathPosts/Splines.html
for class notes on how restricted cubic splines
are calculated
Andy Wheeler
'''

import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import patsy
from scipy.stats import beta



i_seed = 0


# plt.style.use('dark_background')
confidence_interv = 90

color_dict = {'Red': '#e6194b', 
              'Green': '#3cb44b', 
              'Yellow': '#ffe119', 
              'Blue': '#0082c8', 
              'Orange': '#f58231', 
              'Purple': '#911eb4', 
              'Cyan': '#46f0f0', 
              'Magenta': '#e6194b',              
              'Navy': '#000080', 
              'Teal': '#008080', 
              'Brown': '#aa6e28', 
              'Maroon': '#800000', 
              'ForestGreen': '#228b22',
              'SteelBlue': '#4682B4',
              'MidnightBlue': '#1A4876',
              'RoyalPurple': '#7851A9',
              'MangoTando': '#FF8243',
              'Sunglow': '#FFCF48',
              'Lavender': '#e6beff', 
              'Lime': '#d2f53c', 
              'Pink': '#fabebe', 
              'Olive': '#808000', 
              'Coral': '#ffd8b1',
              'Cardinal': '#CC2336',
              'Black': '#000000',
              'Deep Carrot Orange': '#E4682A',
              'burntumber': '#8A3324',
              'darkgoldenrod': '#b8860b',
              'gold': '#FFD700',
              'Navy': '#000080',
              'Firebrick': '#b22222',
              'White': '#FFFFFF',
             }

color_names = ['Red',
 'Green',
 'Yellow',
 'Blue',
 'Orange',
 'Purple',
 'Cyan',
 'Magenta',
 'Navy',
 'Teal',
 'Brown',
 'Maroon',
 'ForestGreen',
 'SteelBlue',
 'MidnightBlue',
 'RoyalPurple',
 'MangoTando',
 'Sunglow',
 'Lavender',
 'Lime',
 'Pink',
 'Olive',
 'Coral',
 'Cardinal',
 'Black',
 'Deep Carrot Orange',
 'burntumber',
 'Firebrick']

sync_color_dict = {

    
    
    # patch
    'ECG': 'Blue',
    'ecg_beats': 'Blue',
    
    'accelX': 'MangoTando',
    'scg_x': 'MangoTando',
    'accelY': 'Olive',
    'scg_y': 'Olive',
    'accelZ': 'Teal',
    'scg_z': 'Teal',

    'ppg_r_1': 'Maroon',
    'ppg_g_1': 'ForestGreen',
    'ppg_ir_1': 'darkgoldenrod',
    
    'ppg_r_2': 'Red',
    'ppg_g_2': 'Green',
    'ppg_ir_2': 'gold',
    
    'temp_skin': 'Pink',
    'pres': 'Deep Carrot Orange',
    
    # COSMED
    'HR_cosmed': 'Cyan',
    'RR_cosmed': 'Magenta',
    'VT_cosmed': 'Navy',
    'VE_cosmed': 'burntumber', # VE = VT x RR
    
    'VO2_cosmed': 'Cardinal',
    'VCO2_cosmed': 'Coral',
    'EE_cosmed': 'RoyalPurple',
    'SPO2_cosmed': 'Firebrick',
    
    'OUES_cosmed': 'MidnightBlue',
    
}


random_colors = []
for i in range(60):
    random_colors.append('#%06X' % randint(0, 0xFFFFFF))
    
    

subject_palette = {
    100: '#e6194b',
    101: '#3cb44b',
    102: '#ffe119',
    103: '#0082c8',
    104: '#f58231',
    105: '#911eb4',
    106: '#46f0f0',
    107: '#e6194b',
    108: '#000080',
    109: '#008080',
    110: '#aa6e28',
    111: '#800000',
    112: '#228b22',
    113: '#4682B4',
    114: '#1A4876',
    115: '#7851A9',
    116: '#FF8243',
    117: '#FFCF48',
    118: '#e6beff',
    119: '#d2f53c',
    120: '#fabebe',
    121: '#808000',
    122: '#ffd8b1',
    123: '#CC2336',
    124: '#000000',
    125: '#E4682A',
    126: '#8A3324'
}

legend_dict = {
    0: 'White',
    1: 'Black',
}

race_color_dict = {
    'Caucasian or White': '#cc8963',
    'African American  or Black': '#5975a4',
}

# sig_color_dict = {
#     'ECG_biopac': 'Cyan',
#     'PPG_biopac': 'Magenta',
#     'spiro_biopac': 'Navy',
#     'SpO2_biopac': 'Firebrick',
    
    
#     # patch
#     'ECG_filt': 'Blue',
#     'ECG': 'Blue',
    
#     'accelX': 'MangoTando',
#     'accelY': 'Olive',
#     'accelZ': 'Teal',
#     'accelX_filt': 'MangoTando',
#     'accelY_filt': 'Olive',
#     'accelZ_filt': 'Teal',

#     'ppg_r_1': 'Maroon',
#     'ppg_g_1': 'ForestGreen',
#     'ppg_ir_1': 'darkgoldenrod',
#     'ppg_r_1_filt': 'Maroon',
#     'ppg_g_1_filt': 'ForestGreen',
#     'ppg_ir_1_filt': 'darkgoldenrod',
    
#     'ppg_r_2': 'Red',
#     'ppg_g_2': 'Green',
#     'ppg_ir_2': 'gold',
#     'ppg_r_2_filt': 'Red',
#     'ppg_g_2_filt': 'Green',
#     'ppg_ir_2_filt': 'gold',
    
    
# #     'accelZ_filt': 'Teal',
# #     'accelY_filt': 'RoyalPurple', # caudal-cranial (C-C)
# #     'accelX_filt': 'Coral', # left-right (L-R),
# }
# plotted_sigs = ['ECG_biopac', 'PPG_biopac', 'spiro_biopac', 'SpO2_biopac', 'ECG_filt', 'accelZ_filt', 'ppg_r_1_filt', 'ppg_ir_1_filt', 'ppg_r_2_filt', 'ppg_ir_2_filt']

# sig_color_dict = {
#     'ECG': 'Blue',
    
#     'ppg_r': 'Red',
#     'ppg_g': 'Green',
#     'ppg_ir': 'Orange',
    
# #     'ppg_r_2': 'Maroon',
# #     'ppg_g_2': 'ForestGreen',
# #     'ppg_ir_2': 'Brown',
    
#     'SCG': 'Teal',
# #     'SCG-CC': 'RoyalPurple', # caudal-cranial (C-C)
# #     'SCG-LR': 'Coral', # left-right (L-R),
# }

marker_dict = {
    'circle': 'o',
    'triangle_down': 'v',
    'tri_down': '1',
    'square': 's',
    'pentagon': 'p',
    'plus': 'P',
    'star': '*',
    'hexagon2': 'H',
    'x': 'X',
    'diamond': 'D',
    

    'triangle_up': '^',
    'triangle_right': '>',
    'tri_up': '2',
    'octagon': '8',
    'hexagon1': 'h',
    'x (filled)': 'X',
    'thin_diamond': 'd',
    'alpha': r'$\alpha$',
    'music': r'$\u266B$',
    'lambda': r'$\lambda$',

}


Fitz_dict = {
    1: '#F6D0B1',
    2: '#E8B58F',
    3: '#D29F7C',
    4: '#BC7951',
    5: '#A65E2B',
    6: '#3B1F1B',
}

def ax_no_top_right(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
# def ax_no_top_right(ax):
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

# colornames = list(color_dict.keys())
# marker_names = list(marker_dict.keys())
# kwargs = {'alpha': 0.8, 'color': 'white', 'dashes': [6, 2], 'linewidth':0.5}

# # alpha=0.5


# def plot_sig(ax, data_matrix, fs, sig_name, color=color_dict[colornames[0]], allpivots=True):
#     """plot_sig: function that plots a PPG (with PPG foot) or SCG (with AO point).

#     Args:
#         ax: 
#         data_matrix: NxM (M windows, N samples in each window)
#         sig_name:
#         color:
#     Returns:
#         None, only plot the signals

#     """
    
#     alpha = 0.1
#     markersize = 2
#     kwargs = {'alpha': 1, 'color': 'white', 'dashes': [6, 2], 'linewidth':1}


#     ts = np.arange(0,1000)/fs

#     ax.plot(ts, data_matrix, color=color, alpha=alpha)
#     ax.plot(ts, np.mean(data_matrix,axis=1), 'w', label=sig_name)
    

#     if sig_name.split('_')[0] in ['gs0', 'gs1', 'gs2', 'gw0', 'gw1', 'gw2', 'is0', 'is1', 'is2', 'iw0', 'iw1', 'iw2', 'rs0', 'rs1', 'rs2', 'rw0', 'rw1', 'rw2']:
# #     if sig_name in 'gs0gs1gs2gw0gw1gw2is0is1is2iw0iw1iw2rs0rs1rs2rw0rw1rw2':
#         i_min, i_derivmax, i_foot, data_derivmax, data_min = PPG_argfoot(data_matrix,fs=fs)

#         ax.plot(i_foot/fs, data_min, '.', markersize=markersize, color = 'w', markerfacecolor='white', alpha=alpha)

#         if allpivots:
#             ax.plot(i_min/fs, data_min, 'o', markersize=markersize, color = 'w', markerfacecolor='white', alpha=alpha)
#             ax.plot(i_derivmax/fs, data_derivmax, 'o', markersize=markersize, color = 'w', markerfacecolor='white', alpha=alpha)
#     #     ax.plot([i_PTT, i_derivmax], [data_matrix[i_min], data_matrix[i_derivmax]], **kwargs)
#     #     ax.annotate(r'$PPG_{foot}$', (i_PTT+20,  data_matrix[i_min]+2),fontsize=10, color='white')


#         data_mean = data_matrix.mean(axis=1).reshape(data_matrix.shape[0], 1)
#         i_min_mean, i_derivmax_mean, i_foot_mean, data_derivmax_mean, data_min_mean = PPG_argfoot(data_mean,fs=fs) 

#         ax.plot(i_foot_mean/fs, data_min_mean, 'o', markersize=3, color = 'w', markerfacecolor='white')

#         if allpivots:
#             ax.plot(i_min_mean/fs, data_min_mean, 'o', markersize=3, color = 'w', markerfacecolor='white')
#             ax.plot(i_derivmax_mean/fs, data_derivmax_mean, 'o', markersize=3, color = 'w', markerfacecolor='white')
#         ax.axhline(y=data_min_mean, xmin=0, xmax=1000, **kwargs)
#         ax.plot([i_foot_mean/fs, i_derivmax_mean/fs], [data_min_mean, data_derivmax_mean], **kwargs)

#         ax.annotate(r'$PPG_{foot}$', ((i_foot_mean+20)/fs,  data_min_mean+2),fontsize=10, color='white')
        
#         ylim_range = data_mean.max()-data_mean.min()
#         ax.set_ylim([data_mean.min()-0.1*ylim_range, data_mean.max()+0.1*ylim_range])
#         ax.set_xlim([0, data_matrix.shape[0]/fs+0.01])

#     if 'scg' in sig_name:
#         i_peaks_ao, data_SCGao = SCG_ao(data_matrix,fs=fs)
#         ax.plot(i_peaks_ao/fs, data_SCGao, 'o', markersize=markersize, color = 'w', markerfacecolor='white', alpha=alpha)

    
#         data_mean = data_matrix.mean(axis=1).reshape(data_matrix.shape[0], 1)
#         i_peaks_ao_mean, data_SCGao_mean = SCG_ao(data_mean,fs=fs)
#         ax.plot(i_peaks_ao_mean/fs, data_SCGao_mean, '.', markersize=markersize, color = 'w', markerfacecolor='white', alpha=alpha)

#         ylim_range = data_mean.max()-data_mean.min()
#         ax.set_ylim([data_mean.min()-0.1*ylim_range, data_mean.max()+0.1*ylim_range])
#         ax.set_xlim([0, data_matrix.shape[0]/fs+0.01])

# #     ax.legend(loc='upper right')

# def plot_distrubtion(ax, data_matrix, fs, sig_name, task_id, color=color_dict[colornames[0]], remove_ol=False):
    
#     if sig_name.split('_')[0] in ['gs0', 'gs1', 'gs2', 'gw0', 'gw1', 'gw2', 'is0', 'is1', 'is2', 'iw0', 'iw1', 'iw2', 'rs0', 'rs1', 'rs2', 'rw0', 'rw1', 'rw2']:
#         i_min, i_derivmax, i_foot, data_derivmax, data_min = PPG_argfoot(data_matrix,fs=fs)
#         data_dist = i_foot/FS_PPG
#     if 'scg' in sig_name:
#         i_peaks_ao, data_SCGao = SCG_ao(data_matrix,fs=fs)
#         data_dist = i_peaks_ao/FS_SCG
     
#     # keep data within confidence_interv
#     if remove_ol:
#         indices_CI = arg_CI(data_dist, confidence_interv=confidence_interv)
#         data_dist = data_dist[indices_CI]


#     c = 'white'
    
#     bp = ax.boxplot([data_dist], vert=True, patch_artist=True,
#                 boxprops=dict(facecolor=color_dict[colornames[task_id]], color=c, alpha=0.5),
#                 capprops=dict(color=c),
#                 whiskerprops=dict(color=c),
#                 flierprops=dict(color=c, markeredgecolor=c),
#                 medianprops=dict(color=c),
#                 showfliers=True,
#                 labels=[sig_name.split('_')[0]],
#                    )
    
# def plot_dist_tasks(list_data_dist, sig_name, subject_id, remove_ol=False, fig_dir=None):
#     c = "white"
#     fig = plt.figure(figsize=(12, 5), dpi=120)
#     ax = fig.add_subplot(1,1,1)

#     for i, data_dist in enumerate(list_data_dist):
#         # keep data within confidence_interv
#         if remove_ol:
#             indices_CI = arg_CI(data_dist, confidence_interv=confidence_interv)
#             data_dist = data_dist[indices_CI]
            
#             if sig_name.split('_')[0] in ['gs0', 'gs1', 'gs2', 'gw0', 'gw1', 'gw2', 'is0', 'is1', 'is2', 'iw0', 'iw1', 'iw2', 'rs0', 'rs1', 'rs2', 'rw0', 'rw1', 'rw2']:
#                 data_dist = data_dist/FS_PPG
#             if 'scg' in sig_name:
#                 data_dist = data_dist/FS_SCG
                
# #             data_dist = arg_CI(data_dist, confidence_interv=confidence_interv)
# #             value_up = np.percentile(data_dist, int((100-confidence_interv)/2+confidence_interv))
# #             value_lw = np.percentile(data_dist, int((100-confidence_interv)/2))
# #             data_dist = data_dist[(data_dist<value_up) & (data_dist>value_lw)]

#         task_id = i+1
#         ax.boxplot(data_dist, positions=[task_id], vert=True, patch_artist=True,
#                     boxprops=dict(facecolor=color_dict[colornames[task_id]], color=c, alpha=0.5),
#                     capprops=dict(color=c),
#                     whiskerprops=dict(color=c),
#                     flierprops=dict(color=c, markeredgecolor=c),
#                     showfliers=True,
#                     medianprops=dict(color=c),
#                   )


#     ax.set_xlim(0.5,0.5+len(list_data_dist))
#     ax.set_xticks(range(1, len(list_data_dist)+1))
    
#     ax.set_xticklabels([task_dict[i+1] for i in range(len(list_data_dist))], rotation=60)
#     ax.set_title(sig_name)
    

#     if fig_dir is not None:  
#       fig.savefig(fig_dir + '/{}peaks_id{}_dist.png'.format(sig_name, int(subject_id)), bbox_inches='tight', facecolor=fig.get_facecolor())

	
	
# def plot_BPprediction(ax, performance_dict, mode='val'):
#     fontsize = 17

#     prediction = performance_dict['final_prediction_'+mode]
#     label_SDbp = performance_dict['label_'+mode]
#     meta = performance_dict['meta_'+mode]

#     task_id = int(np.unique(meta[:,0])[0])
#     subject_id = int(np.unique(meta[:,1])[0])

#     sp_data = label_SDbp[:,0]
#     dp_data = label_SDbp[:,1]

#     sp_prediction = prediction[:,0]
#     dp_prediction = prediction[:,1]
    
#     dummy_prediction = performance_dict['label_train'].mean(axis=0)

#     ax.plot(sp_data, 'o', c=color_dict[colornames[9]], alpha=0.2, label='ground truth BP')
#     ax.plot(dp_data, 'o', c=color_dict[colornames[9]], alpha=0.2)

#     ax.plot(sp_prediction, '-', c=color_dict[colornames[11]], alpha=0.8, label='predicted BP')
#     ax.plot(dp_prediction, '-', c=color_dict[colornames[11]], alpha=0.8)

#     ax.axhline(y=dummy_prediction[0], color='w', linestyle='-', label='dummy regression')
#     ax.axhline(y=dummy_prediction[1], color='w', linestyle='-')


#     y_min, y_max = ax.get_ylim()

#     for task_id in np.unique(meta[:,0]).astype(int):
#         task_start = np.where(meta[:,0]==task_id)[0][0]
#         task_end = np.where(meta[:,0]==task_id)[0][-1]

#         ax.fill_between( np.array( [ task_start, task_end ] ),
#                              y_min * np.array( [1, 1] ),
#                              y_max * np.array( [1, 1] ),
#                              facecolor = color_dict[colornames[task_id]],
#                              alpha = 0.05) 

#         annotate_alpha = 0.2
#         ax.annotate(task_dict[task_id], (task_start/2+task_end/2, dummy_prediction[0]+15), fontsize=15, color='white', horizontalalignment='center', verticalalignment='bottom', rotation=60)

# #     ax.fill_between( np.array( [ 0, sp_data.shape[0] ] ),
# #                          y_min * np.array( [1, 1] ),
# #                          y_max * np.array( [1, 1] ),
# #                          facecolor = color_dict[colornames[task_id]],
# #                          alpha = 0.05) 

# #     ax.annotate(task_dict[task_id], (0/2+sp_data.shape[0]/2,  0.3*dummy_prediction[0]+0.7*y_max),fontsize=15, color='white', horizontalalignment='center', verticalalignment='bottom', rotation=60)

#     ax.set_ylabel('SBP and DBP (mmHg)', fontsize=fontsize)
#     ax.set_xlabel('time (sec)', fontsize=fontsize)
#     ax.set_title('[{}]: SBP and DBP ground truth and prediction'.format(mode), fontsize = 20)

#     ax.tick_params(axis = 'both', which = 'major', labelsize = fontsize)

#     ax.set_ylim([dummy_prediction[1]-55, dummy_prediction[0]+55])
#     ax.set_xlim([0, sp_data.shape[0]+0.01])
# #     ax.legend(fontsize=fontsize,loc='upper right', bbox_to_anchor=(1.3, 1))
# #     ax.legend(fontsize=fontsize,loc='upper right')
    
    
# def plot_performance_epochs(ax, performance_dict, error_type='loss'):

#     N_epochs = performance_dict[error_type + '_train'].shape[0]

#     N_epochs = performance_dict[error_type + '_train'].shape[0]
#     ax.plot(range(N_epochs), performance_dict[error_type + '_train'], color = 'skyblue', label='train')
#     ax.plot(range(N_epochs), performance_dict[error_type + '_val'], color = 'coral', label='val')

# #     ax.legend( fontsize = 15 )
#     ax.set_ylabel('error', fontsize = 15)
#     ax.set_xlabel('epochs', fontsize = 15)
#     ax.set_title(error_type, fontsize = 20)
    

# def performance_art(fig, performance_dict):

#     gs=GridSpec(3,5) # 2 rows, 3 columns

#     ax0=fig.add_subplot(gs[0,0]) # First row, first column
#     ax1=fig.add_subplot(gs[0,1]) # First row, second column
#     ax2=fig.add_subplot(gs[0,2]) # First row, third column
#     ax3=fig.add_subplot(gs[0,3]) # First row, third column
#     ax4=fig.add_subplot(gs[0,4]) # First row, third column

#     ax5=fig.add_subplot(gs[1,:]) # First row, third column
#     ax6=fig.add_subplot(gs[2,:]) # First row, third column

#     plot_performance_epochs(ax0, performance_dict, error_type='total_loss')
#     plot_performance_epochs(ax1, performance_dict, error_type='rmse_SBP')
#     plot_performance_epochs(ax2, performance_dict, error_type='rmse_DBP')
#     plot_performance_epochs(ax3, performance_dict, error_type='mae_SBP')
#     plot_performance_epochs(ax4, performance_dict, error_type='mae_DBP')
# #     ax.legend( fontsize = 15 )
#     ax4.legend(fontsize=15,loc='upper right', bbox_to_anchor=(1.7, 1))

#     plot_BPprediction(ax5, performance_dict, mode='train')
#     ax5.legend(fontsize=15,loc='upper right', bbox_to_anchor=(1.2, 1))

#     plot_BPprediction(ax6, performance_dict, mode='val')

#     fig.tight_layout()
# #     plt.show()