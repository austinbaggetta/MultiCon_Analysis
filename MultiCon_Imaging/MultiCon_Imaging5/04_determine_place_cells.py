# %%
import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from os.path import join as pjoin
from natsort import natsorted

sys.path.append('../../')
import circletrack_neural as ctn
import circletrack_behavior as ctb
import place_cells as pc

project_dir = 'MultiCon_Imaging'
experiment_dir = 'MultiCon_Imaging5'
mouse_list = ['mc44', 'mc46', 'mc49', 'mc51', 'mc52'] 
dpath = f'../../../{project_dir}/{experiment_dir}/output/aligned_minian/'
spath = f'../../../{project_dir}/{experiment_dir}/output/aligned_place_cells/'
data_type = 'S'
only_running = True
correct_dir = True
smooth = False
binarized = False
velocity_thresh = 10
bin_size = 0.16
nshuffles = 500
percentile = 95
min_event_amount = 0.2
# %%
for mouse in tqdm(mouse_list):
    mpath = pjoin(dpath, f'{mouse}/{data_type}')
    for session in natsorted(os.listdir(mpath)):
        if (mouse == 'mc56') & (session == f'mc56_{data_type}_1.nc'):
            pass
        else:
            save_path = pjoin(spath, f'{mouse}/{data_type}')
            sdata = xr.open_dataset(pjoin(mpath, session))[data_type]
            num_neurons = sdata.shape[0]
            minimum_act_bool = pc.minimum_activity_level(sdata, minimum_event_amount=min_event_amount, bin_size_seconds=60, fps=30, func=np.sum, binarized=0)

            if smooth:
                smoothed_data = ctn.moving_average_xarray(sdata, ksize=8) 
            else:
                smoothed_data = sdata.copy()

            neural_data, position_data = ctn.subset_correct_dir_and_running(smoothed_data, correct_dir=correct_dir, only_running=only_running, 
                                                                            velocity_thresh=velocity_thresh)


            ## Calculate observed Skagg's information content and observed spatial coherence
            population_activity, occupancy, _ = pc.spatial_activity(neural_data, position_data, bin_size=bin_size, binarized=binarized)
            tuning_curves = pc.get_tuning_curves(population_activity, occupancy)
            avg_of_avg, spatial_coherence_values = pc.calculate_spatial_coherence(tuning_curves, ksize=8)
            bits_per_event = pc.skaggs_information_content(tuning_curves, occupancy)
            first_second = pc.first_second_half_stability(neural_data, bin_size=bin_size)
            odd_even = pc.odd_even_stability(neural_data, bin_size=bin_size)

            ## Shuffle data to create a null distribution
            shuffled_si, shuffled_sc = pc.shuffle_spatial_metrics(neural_data, lin_pos_col='lin_position', bin_size=bin_size, binarized=binarized, nshuffles=nshuffles)
            shuffled_first_second, shuffled_odd_even = pc.shuffle_stability_metrics(neural_data, bin_size=bin_size, nshuffles=nshuffles)
            
            place_cells_skaggs = np.array([bits_per_event[neuron] > np.percentile(shuffled_si[:, neuron], percentile) for neuron in np.arange(0, num_neurons)])
            place_cells_coherence = np.array([spatial_coherence_values[neuron] > np.percentile(shuffled_sc[:, neuron], percentile) for neuron in np.arange(0, num_neurons)])
            place_cells_first = np.array([first_second[neuron] > np.percentile(shuffled_first_second[:, neuron], percentile) for neuron in np.arange(0, num_neurons)])
            place_cells_odd = np.array([odd_even[neuron] > np.percentile(shuffled_odd_even[:, neuron], percentile) for neuron in np.arange(0, num_neurons)])

            sdata = sdata.assign_coords(skaggs_info=('unit_id', bits_per_event),
                                        skaggs_place=('unit_id', place_cells_skaggs),
                                        coherence=('unit_id', spatial_coherence_values),
                                        coherence_fisherz=('unit_id', np.arctanh(spatial_coherence_values)),
                                        coherence_place=('unit_id', place_cells_coherence),
                                        first_second=('unit_id', first_second),
                                        first_second_fisherz=('unit_id', np.arctanh(first_second)),
                                        first_second_place=('unit_id', place_cells_first),
                                        odd_even=('unit_id', odd_even),
                                        odd_even_fisherz=('unit_id', np.arctanh(odd_even)),
                                        odd_even_place=('unit_id', place_cells_odd),
                                        shuffled_avg_si=('unit_id', np.mean(shuffled_si, axis=0)),
                                        shufled_std_si=('unit_id', np.std(shuffled_si, axis=0, ddof=1)),
                                        minimum_activity_met=('unit_id', minimum_act_bool))
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            sdata.to_netcdf(pjoin(save_path, f'{session}'))

# %%
