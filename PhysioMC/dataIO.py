import os
import sys
import numpy as np
from scipy.io import loadmat
import pandas as pd

def data_saver(data, name, outputdir):
    """ usage: data_saver(df_merged_interp_alldicts, 'data', outputdir)"""
    outputdir_data = os.path.join(outputdir, name+'.npz')
    #   print('outputdir for {}:'.format(name), outputdir_data)
    np.savez(outputdir_data, data=data, allow_pickle=True)
    loaded_data = np.load(outputdir_data, allow_pickle=True)['data']
    #     loaded_data = np.load(outputdir_data, allow_pickle=True)['data']
    #   print('Are {} save and loadded correctly? '.format(name), np.array_equal(loaded_data, data))
    #   print('')
    
def data_loader(name, inputdir):
    """ usage: data = data_loader('data', outputdir)"""
    inputdir_data = os.path.join(inputdir, name+'.npz')
    data = np.load(inputdir_data, allow_pickle=True)['data']
    return data

def header_extract(header):
    Fs = float(header[0].split(' ')[2])
    N_sigs = int(header[0].split(' ')[1])
    start_time = header[0].split(' ')[4]
    date =  header[0].split(' ')[5].split('\n')[0] # TODO: check if this is birthday
    N_samples = int(header[0].split(' ')[3])
    t_dur = N_samples/Fs/60/60 # hr
    
    # sig_names = []
    unit_dict = {}
    gain_dict = {}
    for h in header[1:]:
        sig_name = h.split(' ')[-1].split('\n')[0]
        
        unit = h.split(' ')[2].split('/')[1]
        if unit == '':
            unit = 'a.u.'
        unit_dict[sig_name] = unit
        
        gain = h.split(' ')[2].split('/')[0]
        gain_dict[sig_name] = float(gain)
        # sig_names.append( h.split(' ')[-1].split('\n')[0] )
        
    
    header_dict = {
        'Fs': Fs,
        'N_sigs': N_sigs,
        'start_time': start_time,
        'date': date,
        'N_samples': N_samples,
        't_dur(hr)': t_dur,
        'unit_dict': unit_dict,
        'gain_dict': gain_dict,
    }
    return header_dict

def get_df_bed(rec_id, inputdir, header_id='0000'):

    # 1. get folder dir of the header and the waveform

    # rec_id = list(recording_lookup.keys())[selected_id]
    # rec_id = row['deBedTime']
    bed_id = rec_id.split('-')[0]
    subject_id = rec_id.split('-')[1]

    # header_id = recording_lookup[rec_id]
    # header_id = '0000'
    header_name = rec_id+'_{}.hea'.format(header_id)
    # header_name = rec_id+'_0n.hea'
    # A118-0520880083.hea
    # get recording directory
    inputdir_rec = inputdir + rec_id + '/'
    # print(os.listdir(inputdir_rec))

    # 2. get header file directory
    headerdir = inputdir_rec + header_name
    # print(headerdir)

    # load header file
    with open(headerdir, 'r') as f:
        header = f.readlines()

    header_dict = header_extract(header)
    unit_dict = header_dict['unit_dict']
    gain_dict = header_dict['gain_dict']
    Fs = header_dict['Fs']

    # print(header_dict)

    # 3. get recording file
    recording_name = rec_id+'_'+header_id

    waveformdir = inputdir+rec_id+'/' +rec_id+'_'+header_id+'.mat'
    data = loadmat(waveformdir)['val']

    # 4. convert the unit of the data, store it in dataframe
    raw_dict = {}

    for i, sig_name in enumerate(gain_dict.keys()):
        raw_dict[sig_name] = data[i, :]/gain_dict[sig_name]

    df_bed = pd.DataFrame(raw_dict)
    
    t_arr = np.arange(df_bed.shape[0])/header_dict['Fs']
    df_bed['time'] = t_arr
#     t_arr = np.arange(df.shape[0])/Fs

#     df['time'] = t_arr
#     df['time'] = df['time']-df['time'].values[0]

#     t_dur = df.shape[0]/Fs
#     print('t_dur: {:.2f}s'.format(t_dur))

    return df_bed, header_dict

def get_header_valid(inputdir_rec):

    for header_name in sorted(os.listdir(inputdir_rec)):
        if '.hea' not in header_name:
            continue

        if '_' not in header_name:
            continue

        if '_0n' in header_name:
            continue
        if 'layout' in header_name:
            continue

        # print(header_name)
        headerdir = inputdir_rec + header_name


        header_id = header_name.split('_')[-1].split('.hea')[0]


        with open(headerdir, 'r') as f:
            header = f.readlines()

        header_dict = header_extract(header)


        allheader_hours = int(header_id)*8
        currentheader_hours = header_dict['t_dur(hr)']

        # waveform_start = row['BED_LOCATION_START'] + timedelta(hours=allheader_hours)
        # waveform_end = row['BED_LOCATION_START'] + timedelta(hours=allheader_hours) + timedelta(hours=currentheader_hours)

        # using DateStart since it is the start time of the waveform
        waveform_start = row['DateStart'] + timedelta(hours=allheader_hours)
        waveform_end = row['DateStart'] + timedelta(hours=allheader_hours) + timedelta(hours=currentheader_hours)

        header_dict['waveform_start'] = waveform_start
        header_dict['waveform_end'] = waveform_end
        header_dict['header_name'] = header_name
        header_dict['header_id'] = header_id

        if (waveform_start > row['times']) or (waveform_end < row['times']):
            # print('wrong recording')
            continue
        else:
            # print(header_start, header_end, row['times'])
            # print('bingo!')
            return header_dict


        return None
        # print('\t', header_dict['start_time'], header_dict['date'])

    # header_id = header_name.split('_')[-1]
