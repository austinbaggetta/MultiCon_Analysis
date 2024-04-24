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
project_dir = 'MultiCon_Imaging'
experiment_dir = 'MultiCon_Imaging2'
data_type = 'S' ## one of 'C' or 'S'
dpath = os.path.abspath(f'../{project_dir}/{experiment_dir}/output')
out_path = os.path.abspath(f'../{project_dir}/{experiment_dir}/output/rotated_placefields/')
ref_sess = '15' ## reference session
feather_name = 'C5_C5_pearson_place.feat'
mouse_dict = {'mc23': ['11', '12', '13', '14', '15'],
              'mc26': ['11', '13', '14', '15']} ## set which sessions you will be comparing
degree_increment = 10 ## step for x,y position rotation
forward = True ## whether or not to include only forward (correct) trials
place_cells = True ## whether or not to include only place cells
neuron_by_neuron = True ## whether to get the correlation between place fields
centroid_distance = 4
alpha = 0.05
nbins = 20

for mouse in mouse_dict.keys():
    dataset = {}
    result_dict = {'mouse': [], 'session': [], 'ref_session': [], 'shared_cells': [], 'rotation': [], 
                   'ref_unit_id': [], 'sess_unit_id': [], 'r': [], 'p': []}
    for session in mouse_dict[mouse]:
        file_name = '_'.join([mouse, data_type, session])
        da = xr.open_dataset(pjoin(dpath, f'aligned_minian/{mouse}/{data_type}/{file_name}.nc'))['S'] ## select S matrix
        dataset[session] = da

    ## Load cross registration results to choose only cells that are shared between sessions
    mappings = pd.read_pickle(pjoin(dpath, f'cross_registration_results/circletrack_data/{mouse}/mappings_{centroid_distance}.pkl'))

    reference_session = dataset[ref_sess] ## set reference session
    if forward:
        forward_trials, reverse_trials = ctb.forward_reverse_trials(reference_session, reference_session['trials'])
        df = reference_session[:, reference_session['trials'] == forward_trials[0]]
        for trial in forward_trials[1:]:
            subdata = reference_session[:, reference_session['trials'] == trial]
            df = xr.concat([df, subdata], dim='frame')
    else:
        df = reference_session

    for sess in dataset.keys():
        loop_data = dataset[sess] ## data for that session
        ## Get cell pairs
        pairs = mappings['session'][[df.attrs['date'], loop_data.attrs['date']]].dropna(how='any').drop_duplicates().reset_index(drop=True)
        data = df.sel(unit_id=pairs.iloc[:, 0].values)
        d = loop_data.sel(unit_id=pairs.iloc[:, 1].values)

        if forward:
            forward_t, reverse_t = ctb.forward_reverse_trials(d, d['trials'])
            output = d[:, d['trials'] == forward_t[0]]
            for ftrial in forward_t[1:]:
                sd = d[:, d['trials'] == ftrial]
                output = xr.concat([output, sd], dim='frame')
        else:
            output = d
        
        pf_reference = pc.PlaceFields(x=data['x'].values,
                                  y=data['y'].values,
                                  t=data['behav_t'].values,
                                  neural_data=data.values,
                                  circular=True,
                                  shuffle_test=True,
                                  nbins=nbins)
        data = data.assign_coords(place_cell_r = ('unit_id', pf_reference.data['spatial_info_pvals'] < alpha))

        center = ctb.find_center(output['x'].values, output['y'].values)
        for rotation in tqdm(np.arange(0, (350 + degree_increment), degree_increment)):
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
            
            if neuron_by_neuron:
                if place_cells:
                    neuron_list = np.arange(0, output['unit_id'][(data['place_cell_r'].values) & (output['place_cell_r'].values)].shape[0])
                    ref_data = pf_reference.data['normalized_placefields'][(data['place_cell_r'].values) & (output['place_cell_r'].values)]
                    rot_data = pf_rotated.data['normalized_placefields'][(data['place_cell_r'].values) & (output['place_cell_r'].values)]
                else:
                    neuron_list = np.arange(0, output['unit_id'].shape[0])
                    ref_data = pf_reference.data['normalized_placefields']
                    rot_data = pf_rotated.data['normalized_placefields']

                for neuron in neuron_list:
                    result = pearsonr(ref_data[neuron], rot_data[neuron])
                    result_dict['mouse'].append(mouse)
                    result_dict['session'].append(sess)
                    result_dict['ref_session'].append(ref_sess)
                    result_dict['shared_cells'].append(output['unit_id'][(data['place_cell_r'].values) & (output['place_cell_r'].values)].shape[0])
                    result_dict['rotation'].append(rotation)
                    result_dict['ref_unit_id'].append(data['unit_id'].values[neuron])
                    result_dict['sess_unit_id'].append(output['unit_id'].values[neuron])
                    result_dict['r'].append(result[0])
                    result_dict['p'].append(result[1])
            else:
                if place_cells:
                    ref_data = pf_reference.data['normalized_placefields'][(data['place_cell_r'].values) & (output['place_cell_r'].values)]
                    rot_data = pf_rotated.data['normalized_placefields'][(data['place_cell_r'].values) & (output['place_cell_r'].values)]
                else:
                    ref_data = pf_reference.data['normalized_placefields']
                    rot_data = pf_rotated.data['normalized_placefields']

                for bin in np.arange(0, ref_data.shape[1]):
                    result = pearsonr(ref_data.T[bin], rot_data.T[bin]) ## population vector of spatial bin
                    result_dict['mouse'].append(mouse)
                    result_dict['session'].append(sess)
                    result_dict['ref_session'].append(ref_sess)
                    result_dict['shared_cells'].append(pairs.shape[0])
                    result_dict['rotation'].append(rotation)
                    result_dict['ref_unit_id'].append(np.nan)
                    result_dict['sess_unit_id'].append(np.nan)
                    result_dict['r'].append(result[0])
                    result_dict['p'].append(result[1])
    ## Create dataframe to save
    save_path = pjoin(out_path, mouse)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    rotation_df = pd.DataFrame(result_dict)
    rotation_df.to_feather(pjoin(save_path, feather_name))