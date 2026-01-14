# %%
import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from sklearn.metrics import mutual_info_score
from os.path import join as pjoin
from natsort import natsorted

sys.path.append('../../')
import circletrack_neural as ctn
import circletrack_behavior as ctb
import place_cells as pc

project_dir = 'MultiCon_Imaging'
experiment_dir = 'MultiCon_Imaging5'
mouse_list = ['mc44', 'mc46', 'mc49', 'mc51', 'mc52']
dpath = f'../../../{project_dir}/{experiment_dir}/output/aligned_place_cells/'
spath = f'../../../{project_dir}/{experiment_dir}/output/aligned_mutual_info/'
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
min_trials = None ## lowest number of trials a mouse ran on day 16 is 27 trials, which was a two-context mouse

# %%

for mouse in tqdm(mouse_list):
    mpath = pjoin(dpath, f'{mouse}/{data_type}')
    for session in natsorted(os.listdir(mpath)):
        if (mouse == 'mc56') & (session == f'mc56_{data_type}_1.nc'):
            pass
        else:
            print(session)
            save_path = pjoin(spath, f'{mouse}/{data_type}')
            sdata = xr.open_dataset(pjoin(mpath, session))[data_type]
            num_neurons = sdata.shape[0]
            minimum_act_bool = pc.minimum_activity_level(sdata, minimum_event_amount=min_event_amount, bin_size_seconds=60, fps=30, func=np.sum, binarized=0)

            if min_trials is not None:
                min_data = sdata[:, sdata['trials'] <= min_trials]
            else:
                min_data = sdata.copy()

            if smooth:
                smoothed_data = ctn.moving_average_xarray(min_data, ksize=8) 
            else:
                smoothed_data = min_data.copy()

            neural_data, position_data = ctn.subset_correct_dir_and_running(smoothed_data, correct_dir=correct_dir, only_running=only_running, 
                                                                            velocity_thresh=velocity_thresh)
            ## Calculate mutual information
            population_activity, occupancy, bins = pc.spatial_activity(neural_data, position_data, bin_size=bin_size, binarized=binarized)
            discrete_bins = np.arange(0, bins.shape[0]-1)
            mi = [mutual_info_score((population_activity[:, uid] * 100).astype(int), discrete_bins) for uid in np.arange(0, population_activity.shape[1])]

            ## Shuffle neural activity and calculate mutual info
            shuffled_mutual = pc.shuffle_mutual_info(neural_data, lin_pos_col='lin_position', bin_size=bin_size, nshuffles=nshuffles)
            mutual_cells = np.array([mi[neuron] > np.percentile(shuffled_mutual[:, neuron], percentile) for neuron in np.arange(0, num_neurons)])

            sdata = sdata.assign_coords(mutual_info=('unit_id', mi),
                                        mutual_info_bool=('unit_id', mutual_cells))
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            sdata.to_netcdf(pjoin(save_path, f'{session}'))
# %%
