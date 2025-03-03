# %%
import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from os.path import join as pjoin
from natsort import natsorted
from scipy.stats import pearsonr

sys.path.append('/home/austinbaggetta/csstorage3/CircleTrack/CircleTrackAnalysis')
import circletrack_neural as ctn
import circletrack_behavior as ctb
import place_cells as pc

project_dir = 'MultiCon_Imaging'
experiment_dir = 'MultiCon_Imaging5'
mouse_list = ['mc44', 'mc46', 'mc48', 'mc49', 'mc51', 'mc52']
dpath = f'../../../{project_dir}/{experiment_dir}/output/aligned_minian/'
spath = f'../../../{project_dir}/{experiment_dir}/output/aligned_minian_place/'
data_type = 'S'
only_running = True
correct_dir = True
smooth = False
velocity_thresh = 10
bin_size = 0.16

# %%
for mouse in tqdm(mouse_list):
    mpath = pjoin(dpath, f'{mouse}/{data_type}')
    for session in os.listdir(mpath):
        save_path = pjoin(spath, f'{mouse}/{data_type}')
        sdata = xr.open_dataset(pjoin(mpath, session))[data_type]
        num_neurons = sdata.shape[0]

        if correct_dir:
            forward, _ = ctb.forward_reverse_trials(sdata, sdata['trials'])
            sess = sdata[:, sdata['trials'] == forward[0]]
            for trial in forward[1:]:
                loop_sess = sdata[:, sdata['trials'] == trial]
                sess = xr.concat([sess, loop_sess], dim='frame')
        else:
            sess = sdata.copy()

        x_cm, y_cm = ctb.convert_to_cm(x=sess['x'].values, y=sess['y'].values)
        if only_running:
            velocity, running = pc.define_running_epochs(x_cm, 
                                                         y_cm, 
                                                         sess['behav_t'].values, 
                                                         velocity_thresh=velocity_thresh)
            position_data = sess['lin_position'].values[running]
            neural_data = sess.values[:, running]
        else:
            position_data = sess['lin_position'].values
            neural_data = sess.values
        
        if smooth:
            smoothed_data = ctn.moving_average(neural_data, ksize=8) ## 264 ms
        else:
            smoothed_data = neural_data.copy()

        population_activity, occupancy, _ = pc.average_spatial_activity(smoothed_data, position_data, bin_size=bin_size)
        avg_of_avg = pc.calculate_spatial_coherence(population_activity, ksize=5)
        bits_per_spike = pc.skaggs_information_content(population_activity, occupancy)

        spatial_coherence = np.zeros((num_neurons))
        for n in np.arange(0, num_neurons):
            spatial_coherence[n] = pearsonr(population_activity[:, n], avg_of_avg[:, n])[0]
        
        shuffled_info = pc.shuffled_spatial_info(smoothed_data, position_data, bin_size=bin_size, nshuffles=500)
        shuffled_coh = pc.shuffled_spatial_coherence(smoothed_data, position_data, ksize=5, bin_size=bin_size, nshuffles=500)
        z_values_skaggs = ctn.bootstrap_z_values(bits_per_spike, shuffled_info)
        z_values_coh = ctn.bootstrap_z_values(spatial_coherence, shuffled_coh)

        sdata = sdata.assign_coords(skaggs_info=('unit_id', bits_per_spike),
                                    skaggs_z=('unit_id', z_values_skaggs),
                                    coherence=('unit_id', spatial_coherence),
                                    coherence_z=('unit_id', z_values_coh))
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        sdata.to_netcdf(pjoin(save_path, f'{session}'))

# %%
