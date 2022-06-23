import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
import matplotlib.ticker as plticker
from matplotlib.gridspec import GridSpec


from ECG_module import *
from filters import *
from setting import *

import seaborn as sns
import pandas as pd

import numpy as np

# TODO: change file name to bed_manager.py

def plot_all_sync(df, rec_id, header_dict, t_start=-1, t_end=-1, plt_scale=0.3, fig_name=None, outputdir=None, show_plot=False):

    
    unit_dict = header_dict['unit_dict']
    plotted_sigs = list(unit_dict.keys())
        
    if "PPG_DC" in plotted_sigs:
        plotted_sigs.remove("PPG_DC") 


    for sig_name in plotted_sigs:

        if 'I' == sig_name or 'II' == sig_name or 'III' == sig_name or 'V' == sig_name:
            sig_title = 'ECG (lead {})'.format(sig_name)+ '\n({})'.format(unit_dict[sig_name])
        elif 'RR' == sig_name:
            sig_title = 'RR'+ '\n({})'.format('a.u.')
        elif 'SPO2' == sig_name:
            sig_title = 'PPG'+ '\n({})'.format('a.u.')
        elif 'AR1' == sig_name:
            sig_title = 'AR1'+ '\n({})'.format(unit_dict[sig_name])
        elif 'CVP3' == sig_name:
            sig_title = 'CVP3'+ '\n({})'.format(unit_dict[sig_name])
        elif 'CO2' == sig_name:
            sig_title = 'CO2'+ '\n({})'.format(unit_dict[sig_name])
        else:
            sig_title = 'a.u.'

        # print(sig_name, sig_title)
    # if plotted_sigs is None:
    #     plotted_sigs = PLOTTED_SIGS

    #     subject_id = df['subject_id'].unique()[0]

    # t_start = 1000
    # t_end = 2000
    if t_start==-1:
        t_start = df['time'].min()
    if t_end==-1:
        t_end = df['time'].max()
    
    df['time'] = df['time'] - df['time'].min()
    df_seg = df[(df['time']>=t_start) & (df['time']<=t_end)].copy()

    t_arr = df_seg['time'].values
    #     t_arr = t_arr - t_arr[0]

    t_start = t_arr[0]
    t_end = t_arr[-1]

    t_dur = t_arr[-1] - t_arr[0]
    plt_scale = 0.5

    fig, axes = plt.subplots(len(plotted_sigs), 1, figsize=(25,len(plotted_sigs)), gridspec_kw = {'wspace':0, 'hspace':0}, dpi=80)

    # TODO: make a plot dict
    fontsize = 20*plt_scale
    linewidth = 2
    alpha = 0.8

    for i, ax in enumerate(axes):
        # condition grid
        ax.grid('on', linestyle='--')
        # no x ticks except for the bottom ax
        if i<len(axes)-1:
            ax.set_xticklabels([])
        # add y ticks to all axes
        ax.tick_params(axis='y', which='both', labelsize=20)

        sig_name = plotted_sigs[i]
        sig_plt = df_seg[sig_name].values

        # if sig_name in sync_color_dict.keys():
        #     color = color_dict[sync_color_dict[sig_name]]
        # else:
        #     color = random_colors[i]
        ax.plot(t_arr, sig_plt, alpha=alpha ,zorder=1, linewidth=linewidth)
        ax.set_xlim(t_start, t_end) # remove the weird white space at the beg and end of the plot


    #     # remove some borders (top and right)
    #     ax.spines['right'].set_visible(False)
    #     if i==0:
    #         ax.spines['top'].set_visible(False)

    #     # add y label, indicate their unit
        if 'I' == sig_name or 'II' == sig_name or 'III' == sig_name or 'V' == sig_name:
            sig_title = 'ECG (lead {})'.format(sig_name)+ '\n({})'.format(unit_dict[sig_name])
        elif 'RR' == sig_name:
            sig_title = 'RR'+ '\n({})'.format('a.u.')
        elif 'SPO2' == sig_name:
            sig_title = 'PPG'+ '\n({})'.format('a.u.')
        elif 'AR1' == sig_name:
            sig_title = 'AR1'+ '\n({})'.format(unit_dict[sig_name])
        elif 'CVP3' == sig_name:
            sig_title = 'CVP3'+ '\n({})'.format(unit_dict[sig_name])
        elif 'CO2' == sig_name:
            sig_title = 'CO2'+ '\n({})'.format(unit_dict[sig_name])
        else:
            sig_title = '{}\n(a.u.)'.format(sig_name)



        ax.set_ylabel(sig_title, fontsize=fontsize,rotation = 0,  va='center', ha='center',  labelpad=100)

        # set tick font size
        ax.tick_params(axis='both', which='major', labelsize=fontsize*0.8)


    #     # do this so there's no weird white space on top and bottom of each ax
    #     y_min, y_max = ax.get_ylim()
    #     ax.set_ylim(y_min, y_max)




    # ax.set_xlabel('time (sec)', fontsize=fontsize)
    # fig.subplots_adjust(wspace=0, hspace=0)
    
    fig.tight_layout()

    if fig_name is None:
        fig_name = 'All_rec{}'.format(rec_id)
    else:
        fig_name = fig_name + '_rec{}'.format(rec_id)

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + fig_name, bbox_inches='tight', transparent=False)

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')
        
# TODO: need to renovate this function so it can work again
def plot_ALL_beats(beats_dict, beats_id, subject_id, Fs, show_good=None, fig_name=None, outputdir=None, show_plot=False):
    # beats_id = [1,2,3,4,5,6]
    # beats_names = ['I', 'II', 'III', 'V', 'SPO2', 'PPG_DC']
    # unit_dict['PPG_DC'] = ''
    t_beat = np.arange(beats_dict['I'].shape[0])/Fs

    fig = plt.figure(figsize=(16, 10), dpi=80)
    fontsize = 20
    alpha = 0.03


    # for (beat_name, beat_i) in zip(beats_dict, beats_id):
    for (beat_name, beat_i) in zip(beats_names, beats_id):

        beats = beats_dict[beat_name]

        ax = fig.add_subplot(3, 4, beat_i)
        ax.set_title(beat_name+'\n', fontsize=fontsize)

        show_good = True
        if show_good is not None:


            template = beats_dict[beat_name].mean(axis=1)
            if beat_name=='SPO2':
                mask_all, ol_rate = clean_PPG(beats, template, Fs)
                ax.set_title(beat_name+'\nacception_rate:{:.2f}'.format(1-ol_rate), fontsize=fontsize)


        ax.plot(t_beat, beats, color='gray', alpha=alpha)
        ax.plot(t_beat, np.mean(beats,axis=1), color='firebrick', linewidth=3)

        if 'SPO2' in beat_name or 'I' == beat_name or 'II' == beat_name or 'III' == beat_name or 'V' == beat_name:
            beats_mean = np.mean(beats,axis=1)

            ymin = beats_mean.mean() - beats_mean.std()*5
            ymax = beats_mean.mean() + beats_mean.std()*5
            ax.set_ylim(ymin, ymax)

        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.set_ylabel(unit_dict[beat_name], fontsize=fontsize-3)
        ax.set_xlabel('time (sec)', fontsize=fontsize)



    fig.tight_layout()

    if outputdir is not None:

        if fig_name is None:
            fig_name = 'beats_ensemble_sub{}'.format(subject_id)

        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + fig_name+'.png', transparent=False)

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')


        
def plot_feature_recid(df_features_all, outputdir=None):

    feature_names = list(df_features_all.keys())
    feature_names.remove('i_R_peak')
    feature_names.remove('rec_id')

    fig, axes = plt.subplots(len(feature_names),1, figsize=(10,20), dpi=100)

    for i_ax, (ax, feature_name) in enumerate(zip(axes, feature_names)):
        sns.violinplot(data=df_features_all, y=feature_name, x='rec_id', ax=ax)

        ax.set_ylabel(feature_name, fontsize=15)

        if i_ax!=len(axes)-1:
            ax.xaxis.set_ticklabels([])
            ax.set(xlabel=None)
        else:
            ax.set_xlabel('rec_id', fontsize=15)
            ax.set_xticklabels(v.get_xmajorticklabels(), fontsize = 8,  rotation=60, ha='center')

    fig.tight_layout()

    if outputdir is not None:
        fig.savefig(outputdir + 'features_agg.png', facecolor=fig.get_facecolor())


        

# Signal level function
def get_ecg_IP_norm(ecg1, ecg2):
    return (ecg1 @ ecg2) / ((ecg1 @ ecg1)**0.5 * (ecg2 @ ecg2)**0.5)


# Signal to beat
def segment_df(df_seg, QRS_detector_dict, Fs):
    
    df = get_filt_df(df_seg, Fs).copy()
    
    
    ecg_dict = QRS_detector_dict['ecg_dict']
    i_R_peaks = QRS_detector_dict['i_R_peaks']
#     i_R_peaks = QRS_detector_dict['i_R_peaks']
    # end_offset = QRS_detector_dict['end_offset']
    end_offset = int(1*Fs)
    # print(end_offset)
    # sys.exit()
    
    beats_dict = {}
    

    for sig_name in df.columns:
        if sig_name =='time':
            continue
        if sig_name =='RR':
            continue
        sig_beats, i_R_peaks_used = beat_segmentation(df[sig_name].values, i_R_peaks, start_offset=0, end_offset=end_offset)
        beats_dict[sig_name] = sig_beats
        
        
    sig_beats, i_R_peaks_used = beat_segmentation(df_seg['SPO2'].values, i_R_peaks, start_offset=0, end_offset=end_offset)
    beats_dict['PPG_DC'] = sig_beats

    beats_dict['i_R_peaks'] = i_R_peaks_used
    
    return beats_dict


# feature level function
def get_B2C(sig,beat_deriv1, beat_deriv2, Fs):
    N_window = int(0.15*Fs)
    i_min, i_derivmax, i_foot, data_derivmax, data_min = PPG_argfoot(sig, fs=Fs)
    # np.argmin(np.abs(beat_deriv1[i_min-N_window:i_min+N_window]))
    # i_foot = i_max-N_window

    i_min = i_min[0]

    i_A = np.argmax(beat_deriv2[i_min:i_min+int(0.1*Fs)])
    i_A += i_min

    i_B = np.argmin(beat_deriv2[i_A:i_A+int(0.2*Fs)])
    i_B += i_A

    i_C = np.argmax(beat_deriv2[i_B:i_B+int(0.2*Fs)])
    i_C += i_B

    i_D = np.argmin(beat_deriv2[i_C:i_C+int(0.2*Fs)])
    i_D += i_C

    amplitude_A = np.abs(beat_deriv2[i_A])
    amplitude_B = np.abs(beat_deriv2[i_B])
    amplitude_C = np.abs(beat_deriv2[i_C])
    amplitude_D = np.abs(beat_deriv2[i_D])


    feature_B2C = amplitude_B/amplitude_C
    
    output_dict = {
    'i_min': i_min,
    'i_A': i_A,
    'i_B': i_B,
    'i_C': i_C,
    'i_D': i_D,
    'amplitude_A': amplitude_A,
    'amplitude_B': amplitude_B,
    'amplitude_C': amplitude_C,
    'amplitude_D': amplitude_D,
    'feature_B2C': feature_B2C,
    }
    
    return output_dict

def debug_B2C(beats_PPG, Fs):
    k = 20
    sig= beats_PPG[:,[k]]
    dx = 1/Fs

    beat_deriv1 = np.gradient(sig.squeeze(), dx)
    beat_deriv2 = np.gradient(np.gradient(sig.squeeze(), dx), dx)

    output_dict = get_B2C(sig, beat_deriv1, beat_deriv2, Fs)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8,5), dpi=100)

    ax1.plot(t_beat, beats[:,k], color='gray', alpha=1)
    ax1.set_ylabel('PPG')
    ax2.plot(t_beat, beat_deriv1, color='gray', alpha=1)
    ax2.set_ylabel('PPG 1st derivative')

    ax3.plot(t_beat, beat_deriv2, color='gray', alpha=1)

    ax3.scatter(output_dict['i_min']/Fs, beat_deriv2[output_dict['i_min']])
    ax3.scatter(output_dict['i_A']/Fs, beat_deriv2[output_dict['i_A']], color='r')
    ax3.scatter(output_dict['i_B']/Fs, beat_deriv2[output_dict['i_B']], color='g')
    ax3.scatter(output_dict['i_C']/Fs, beat_deriv2[output_dict['i_C']], color='k')
    ax3.scatter(output_dict['i_D']/Fs, beat_deriv2[output_dict['i_D']], color='y')
    ax3.set_ylabel('PPG 2nd derivative')

    
def get_max2min(sig):

    feature_max2min = sig.max()-sig.min()

    output_dict = {
        'feature_max2min': feature_max2min
    }

    return output_dict

def get_PI(feature_max2min, sig_DC):
    
    featrue_PI = feature_max2min/sig_DC.mean()
    
    output_dict = {
        'featrue_PI': featrue_PI
    }
    
    return output_dict


def get_std(sig):
    return np.std(sig)

def get_skewness(sig):
    return scipy.stats.skew(sig)[0]

def get_kurtosis(sig):
    return scipy.stats.kurtosis(sig)[0]

def get_energy(sig):
    return np.sum((sig-sig.mean())**2)

def get_rms(sig):
    return np.sqrt(np.mean(sig**2))

def get_mean(sig_DC):
    return np.mean(sig_DC)

# beat2feature functions

def beat2features(beats_dict, Fs):

    beats_PPG = beats_dict['SPO2']
    beats_PPG_DC = beats_dict['PPG_DC']
    i_R_peaks = beats_dict['i_R_peaks']
    dx = 1/Fs

    df_features = pd.DataFrame()

    i_R_peak_past = i_R_peaks[0]
    for k in range(beats_PPG.shape[1]):
        
        if k==0:
            continue
        # print(k)
        sig = beats_PPG[:,[k]]
        sig_DC = beats_PPG_DC[:,[k]]
        i_R_peak = i_R_peaks[k]

        beat_deriv1 = np.gradient(sig.squeeze(), dx)
        beat_deriv2 = np.gradient(np.gradient(sig.squeeze(), dx), dx)

        feature_B2C = get_B2C(sig, beat_deriv1, beat_deriv2, Fs)['feature_B2C']
        feature_max2min = get_max2min(sig)['feature_max2min']
        featrue_PI = get_PI(feature_max2min, sig_DC)['featrue_PI']



        # features that use cardiac PPG (filtered between 0.8-4Hz)
        feature_std = get_std(sig)
        feature_skewness = get_skewness(sig)
        feature_kurtosis = get_kurtosis(sig)
        feature_rms = get_rms(sig)
        feature_energy = get_energy(sig)
        
        feature_HR = 1 / ((i_R_peak-i_R_peak_past)/Fs) *60
        # 1/(i_R_peak-i_R_peak_past)/Fs*60
        # i_R_peak_past = i_R_peak


        # features that use raw PPG
        feature_mean = get_mean(sig_DC)


        df_beat = pd.DataFrame({
            'i_R_peak': [i_R_peak],
            'B2C': [feature_B2C],
            'max2min': [feature_max2min],
            'PI': [featrue_PI],
            'std': [feature_std],
            'skewness': [feature_skewness],
            'kurtosis': [feature_kurtosis],
            'rms': [feature_rms],
            'energy': [feature_energy],
            'mean': [feature_mean],
            'HR': [feature_HR],
        })

        df_features = pd.concat([df_features, df_beat])

        # df_features = df_features.append(df_beat)

    return df_features



def get_FQI_dict(data):

    data_median = medfilt(data, k=9)

    data_deviation = np.abs(data - data_median)

    mask_feature = data_deviation < data_deviation.std()*3

    data_deviation_normed = (data_deviation-data_deviation.min()) / (data_deviation.max()-data_deviation.min())

    FQI = np.exp(-0.5*data_deviation_normed)

    FQI_dict = {
        'data_deviation': data_deviation,
        'mask_feature': mask_feature,
        'FQI': FQI,
    }
    return FQI_dict




# feature level quality assessment
def get_mask_all(df_features):

    mask_feature_list = []
    for feature_name in df_features.columns:
        if feature_name == 'i_R_peak':
            continue
        if feature_name == 'HR':
            continue

        FQI_dict = get_FQI_dict(df_features[feature_name].values)
        mask_feature = FQI_dict['mask_feature']
        mask_feature_list.append(mask_feature)
        # print(feature_name)

    mask_feature_list = np.stack(mask_feature_list).T
    mask_feature_list = np.prod(mask_feature_list, axis=1)
    mask_feature_list = mask_feature_list==1
    return mask_feature_list

def masking_beats_dict(beats_dict, mask):

    # print(beats_dict['AR1'].shape)
    for beat_name in beats_dict.keys():
        if beat_name == 'i_R_peaks':
            beats_dict[beat_name] = beats_dict[beat_name][mask]
        else:
            beats_dict[beat_name] = beats_dict[beat_name][:, mask]

    # print(beats_dict['AR1'].shape)

    return beats_dict


# get feature from segments of dataframe
def get_df_features_bed(df_bed, header_dict, seg_duration=30*60, verbose=False):
    # df_bed stores the waveform
    # header_dict stores the meta data
    # seg_duration: the duration to look at the data (in second)
    
    # seg_duration = 1 * 60 * 60 # seconds
    # seg_duration = 30 * 60 # seconds
    if verbose:
        print('seg_duration is {} sec'.format(seg_duration))
    # unit_dict = header_dict['unit_dict']
    Fs = header_dict['Fs']
    N_segs = int((df_bed.shape[0]/Fs)//seg_duration)

    df_features_bed = pd.DataFrame()

    for i_seg in range(N_segs):
        
        if i_seg>3:
            continue

        if verbose:
            print('i_seg: ', i_seg)

        # if i_seg != 5:
        #     continue
        i_start = i_seg*seg_duration*Fs
        i_end = (i_seg+1)*seg_duration*Fs

        df_seg = df_bed[i_start:i_end]

        if df_seg.shape[0]==0:
            continue

        if np.mean(np.diff(df_seg['SPO2'])==0)>signal_quite_threshold:
            continue
        if np.mean(np.diff(df_seg['I'])==0)>signal_quite_threshold:
            continue
        if np.mean(np.diff(df_seg['II'])==0)>signal_quite_threshold:
            continue
        if np.mean(np.diff(df_seg['III'])==0)>signal_quite_threshold:
            continue
        if np.mean(np.diff(df_seg['V'])==0)>signal_quite_threshold:
            continue
            
            
        norm1 = np.abs(get_ecg_IP_norm(df_seg['I'].values, df_seg['II']))
        norm2 = np.abs(get_ecg_IP_norm(df_seg['I'].values, df_seg['III']))
        norm3 = np.abs(get_ecg_IP_norm(df_seg['I'].values, df_seg['V']))

        if (norm1 < 0.3) or (norm2 < 0.3) or (norm3 < 0.3):
            if verbose:
                print('\tbad ECG')
            continue


            
        # TODO: replace the following with get_sig2beats(df_bed)
        sig_name = 'I'
        ECG = df_seg[sig_name].values


        # ECG = df[sig_name].values

        fig_name = 'ECG_diagnostics_'+sig_name
        # QRS_detector_dict = task_HR_detector(ECG, Fs, fig_name=fig_name, outputdir=None, show_plot=True)
        QRS_detector_dict = task_HR_detector(ECG, Fs, fig_name=fig_name, outputdir=None, show_plot=False)

    #     t_ecg = np.arange(ECG.shape[0])/Fs
    #     hr_interp = np.interp(t_ecg, QRS_detector_dict['ts_hr'], QRS_detector_dict['hr'])
        # segment the beats
        beats_dict = segment_df(df_seg, QRS_detector_dict, Fs)
        mask_ppg, ol_rate = clean_PPG(beats_dict['SPO2'], beats_dict['SPO2'].mean(axis=1), Fs)
        beats_dict = masking_beats_dict(beats_dict, mask_ppg)

        if ol_rate > 0.1:
            if verbose:
                print('\tbad PPG')
            continue
        print('\tPPG outlier rate: {:.2f}%'.format( ol_rate*100) )
        # sys.exit()


        # sys.exit()

        if verbose:
            print('\tpass ECG and PPG SQI tests...')


        beats_dict['i_R_peaks'] = beats_dict['i_R_peaks'] + i_start    

        # ensemble the beats (N=4)
        use_ensemble = True

        if use_ensemble:
            for key in beats_dict:
                if 'ppg' not in key:
                    continue
                sig_beats = beats_dict[key]
                beats_dict[key] = get_ensemble_beats(sig_beats, N_enBeats=4, use_woody=False)


        df_features = beat2features(beats_dict, Fs)

        # reject unnecessary features
        mask_all = get_mask_all(df_features)
        # sys.exit()
        if verbose:
            print('\tfaeture rejection rate: {:.2f}%'.format( 100-mask_all.mean()*100 ) )
        df_features = df_features.loc[mask_all, :].copy()

        df_features_bed = pd.concat([df_features_bed, df_features])

    if verbose:
        print('done')
        print('show df_features_bed size:', df_features_bed.shape)
    
    return df_features_bed


def get_sig2beats(df_bed):
    # convert signals into beat_dict
    ecg_name = 'I'
    ECG = df_bed[ecg_name].values
    Fs = np.median(1/np.diff(df_bed['time'].values))

    # ECG = df[sig_name].values

    fig_name = 'ECG_diagnostics_'+ecg_name
    QRS_detector_dict = task_HR_detector(ECG, Fs, fig_name=fig_name, outputdir=None, show_plot=False)

    # segment the beats
    beats_dict = segment_df(df_bed, QRS_detector_dict, Fs)
    beats_dict.pop('PPG_DC', None)

    mask_ppg, ol_rate = clean_PPG(beats_dict['SPO2'], beats_dict['SPO2'].mean(axis=1), Fs)
    beats_dict = masking_beats_dict(beats_dict, mask_ppg)

    return beats_dict


def get_filt_df(df_sync, Fs):
    df = df_sync.copy()

    for sig_name in df.columns:
        if 'I' == sig_name or 'II' == sig_name or 'III' == sig_name or 'V' == sig_name:
            df[sig_name] = get_padded_filt(df[sig_name].values, filter_padded=1, lowcutoff=FILT_ECG[0], highcutoff=FILT_ECG[1], Fs=Fs)
            # df[sig_name] = (df[sig_name].values-df[sig_name].values.mean()) / df[sig_name].values.std()
            
        elif 'RR' == sig_name:
            df[sig_name] = get_padded_filt_DSwrapper(df[sig_name].values, filter_padded=1, lowcutoff=FILT_RESP[0], highcutoff=FILT_RESP[1], Fs=Fs)
            # df[sig_name] = (df[sig_name].values-df[sig_name].values.mean()) / df[sig_name].values.std()

        elif 'SPO2' == sig_name:
            df[sig_name] = get_padded_filt(df[sig_name].values, filter_padded=1, lowcutoff=FILT_PPG[0], highcutoff=FILT_PPG[1], Fs=Fs)
            # df[sig_name] = (df[sig_name].values-df[sig_name].values.mean()) / df[sig_name].values.std()

        else:
            pass

    return df

def get_df_interp(df_bed):
    time = df_bed['time'].values

    t_start = np.min(time)
    t_end = np.max(time)

    time_interp = np.arange(my_ceil(t_start, decimal=-3)*FS_RESAMPLE, my_floor(t_end, decimal=-3)*FS_RESAMPLE+1)/FS_RESAMPLE

    sig_dict = {}

    for sig_name in df_bed.keys():
        if sig_name == 'time':
            continue
        if 'I' == sig_name or 'II' == sig_name or 'III' == sig_name or 'V' == sig_name:
            sig_dict[sig_name] = np.interp(time_interp, time, df_bed[sig_name].values)
        elif 'RR' == sig_name:
            sig_dict[sig_name] = np.interp(time_interp, time, df_bed[sig_name].values)
        elif 'SPO2' == sig_name:
            sig_dict[sig_name] = np.interp(time_interp, time, df_bed[sig_name].values)
        else:
            pass

    sig_dict['time'] = time_interp

    df_interp = pd.DataFrame.from_dict(sig_dict)
    return df_interp


def df_bed2PPGmatrix(df_bed, verbose=True):
    df = get_df_interp(df_bed)
    df = get_filt_df(df, FS_RESAMPLE)
    
    if np.mean(np.diff(df['SPO2'])==0)>signal_quite_threshold:
        # print('SPO2 channel is too quiet')
        debug_message = '[SQI] SPO2 channel is too quiet'
        if verbose:
            print(debug_message)
        return None, debug_message
    if np.mean(np.diff(df['I'])==0)>signal_quite_threshold:
        debug_message = '[SQI] I channel is too quiet'
        if verbose:
            print(debug_message)
        return None, debug_message
    
    
    df['time'] = df['time'] - df['time'].min()

    window_size = 8 # seconds
    window_length = window_size * FS_RESAMPLE

    i_starts = np.arange(df.shape[0]//window_length)*window_length
    end_offset = window_length
    # # ECG segments
    ECG_segments, i_start_final = beat_segmentation(df['I'], i_starts, start_offset=0, end_offset=window_length)
    # PPG segments
    PPG_segments, i_start_final = beat_segmentation(df['SPO2'], i_starts, start_offset=0, end_offset=window_length)

    debug_message = '[perfecto]'
    return PPG_segments, ECG_segments, debug_message