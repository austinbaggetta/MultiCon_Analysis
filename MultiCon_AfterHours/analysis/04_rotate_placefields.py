import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from os.path import join as pjoin
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

sys.path.append('/home/austinbaggetta/csstorage3/CircleTrack/CircleTrackAnalysis')
import circletrack_behavior as ctb
import place_cells as pc

## Set parameters
dataset = {}
result_dict = {'mouse': [], 'session': [], 'rotation': [], 'r': [], 'p': []}
dpath = os.path.abspath('../MultiCon_AfterHours/MultiCon_EEG1/output/aligned_minian')
save_path = os.path.abspath('../MultiCon_AfterHours/MultiCon_EEG1/output/rotated_placefields/mc_EEG1_01/')
feather_name = '1_5_pearson_all.feat'
mouse_dict = {'mc_EEG1_01': ['1', '5']} ## set which sessions you will be comparing
degree_increment = 180 ## step for x,y position rotation
forward = False ## whether or not to include only forward (correct) trials
place_cells = True ## whether or not to include only place cells
neuron_by_neuron = True ## whether to get the correlation between place fields
ref_sess = '1' ## reference session
alpha = 0.05
nbins = 20

for mouse in mouse_dict.keys():
    for session in mouse_dict[mouse]:
        file_name = '_'.join([mouse, session])
        da = xr.open_dataset(pjoin(dpath, f'{mouse}/{file_name}.nc'))['S'] ## select S matrix
        dataset[session] = da

    reference_session = dataset[ref_sess] ## set reference session
    if forward:
        forward_trials, reverse_trials = ctb.forward_reverse_trials(reference_session, reference_session['trials'])
        data = reference_session[:, reference_session['trials'] == forward_trials[0]]
        for trial in forward_trials[1:]:
            subdata = reference_session[:, reference_session['trials'] == trial]
            data = xr.concat([data, subdata], dim='frame')
    else:
        data = reference_session
    
    pf_reference = pc.PlaceFields(x=data['x'].values,
                                  y=data['y'].values,
                                  t=data['behav_t'].values,
                                  neural_data=data.values,
                                  circular=True,
                                  shuffle_test=True,
                                  nbins=nbins)
    data = data.assign_coords(place_cell_r = ('unit_id', pf_reference.data['spatial_info_pvals'] < alpha))

    for sess in dataset.keys():
        d = dataset[sess] ## data for that session

        if forward:
            forward, reverse = ctb.forward_reverse_trials(d, d['trials'])
            output = d[:, d['trials'] == forward[0]]
            for ftrial in forward:
                sd = d[:, d['trials'] == trial]
                output = xr.concat([output, sd], dim='frame')
        else:
            output = d

        center = ctb.find_center(output['x'].values, output['y'].values)
        for rotation in tqdm(np.arange(0, (360 + degree_increment), degree_increment)):
            rotated_points = np.empty(shape=(2, output['x'].shape[0]))

            for idx, (x, y) in enumerate(zip(output['x'].values, output['y'].values)):
                p = (x, y)
                rot = ctb.rotate(p, origin=center, degrees=rotation)
                rotated_points[0, idx] = rot[0] ## x values
                rotated_points[1, idx] = rot[1] ## y values

            pf_rotated = pc.PlaceFields(x=rotated_points[0],
                                        y=rotated_points[1],
                                        t=output['behav_t'].values,
                                        neural_data=output.values,
                                        circular=True,
                                        shuffle_test=True,
                                        nbins=nbins)
            output = output.assign_coords(place_cell_r = ('unit_id', pf_rotated.data['spatial_info_pvals'] < alpha))

            print(sess)
            if neuron_by_neuron:
                if place_cells:
                    neuron_list = np.arange(0, output['unit_id'][(data['place_cell_r']) & (output['place_cell_r'])].shape[0])
                    ref_data = pf_reference.data['normalized_placefields'][(data['place_cell_r']) & (output['place_cell_r'])]
                    rot_data = pf_rotated.data['normalized_placefields'][(data['place_cell_r']) & (output['place_cell_r'])]
                else:
                    neuron_list = np.arange(0, output['unit_id'].shape[0])
                    ref_data = pf_reference.data['normalized_placefields']
                    rot_data = pf_rotated.data['normalized_placefields']

                for neuron in neuron_list:
                    result = pearsonr(ref_data[neuron], rot_data[neuron])
                    result_dict['mouse'].append(mouse)
                    result_dict['session'].append(sess)
                    result_dict['rotation'].append(rotation)
                    result_dict['r'].append(result[0])
                    result_dict['p'].append(result[1])
            else:
                if place_cells:
                    ref_data = pf_reference.data['normalized_placefields'][(data['place_cell_r']) & (output['place_cell_r'])]
                    rot_data = pf_rotated.data['normalized_placefields'][(data['place_cell_r']) & (output['place_cell_r'])]
                else:
                    ref_data = pf_reference.data['normalized_placefields']
                    rot_data = pf_rotated.data['normalized_placefields']

                for bin in np.arange(0, ref_data.shape[1]):
                    result = pearsonr(ref_data.T[bin], rot_data.T[bin]) ## population vector of spatial bin
                    result_dict['mouse'].append(mouse)
                    result_dict['session'].append(sess)
                    result_dict['rotation'].append(rotation)
                    result_dict['r'].append(result[0])
                    result_dict['p'].append(result[1])

## Create dataframe to save
rotation_df = pd.DataFrame(result_dict)
rotation_df.to_feather(pjoin(save_path, feather_name))