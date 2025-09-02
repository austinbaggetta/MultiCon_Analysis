# %%
import sys
import os
from os.path import join as pjoin

import numpy as np
import pandas as pd
from natsort import natsort_keygen
from tqdm import tqdm

sys.path.append('/home/austinbaggetta/csstorage3/CircleTrack/CircleTrackAnalysis')
import circletrack_behavior as ctb

# %%
## Set parameters
starting_idx = 0 ## can use to change which days you want preprocessed
num_days = 5 ## how many days of linear track
parent_dir = 'CircleTrack_Recall'
experiment_dir = 'Recall4'
behavior_path = os.path.abspath(f"../../../{parent_dir}/{experiment_dir}/lineartrack_data/")
output_path = os.path.abspath(f"../../../{parent_dir}/{experiment_dir}/output/lin_behav")
cohort_number = 'mcr4'
mouse_list = [f'mcr{x}' for x in np.arange(60, 90)]
mouse_list = [f'mcr{x}' for x in np.arange(68, 90)]
str2match = "(mcr[0-9]+)"
if not os.path.exists(output_path):
    os.makedirs(output_path)
## Set relative path variable for behavior data
csv_path = pjoin(behavior_path, "data/**/**/**/linear_track.csv")
log_path = pjoin(behavior_path, "data/**/**/**/**.log")
## Create list of files
file_list = ctb.get_file_list(csv_path)
log_list = ctb.get_file_list(log_path)
## Loop through file_list to extract mouse name
mouseID = []
for file in file_list:
    mouse = ctb.get_mouse(file, str2match)
    mouseID.append(mouse)
## Combine file_list and mouseID
combined_list = ctb.combine(file_list, mouseID)
combined_log = ctb.combine(log_list, mouseID)

# %%
for mouse in mouse_list:
    print(mouse)
    natsort_key = natsort_keygen()
    subset = ctb.subset_combined(combined_list, mouse).reset_index(drop=True)
    subset_log = ctb.subset_combined(combined_log, mouse).reset_index(drop=True)
    subset = sorted(subset, key=natsort_key)
    subset_log = sorted(subset_log, key=natsort_key)
    for i, session in tqdm(enumerate(np.arange(0, num_days)), leave=False): ## five days of linear track
        if i < starting_idx:
            pass 
        else:
            print(session)
            linear_track = pd.read_csv(subset[i])
            reward_one, reward_two = ['reward1', 'reward2']
            rewards = [reward_one, reward_two]
            linear_track = ctb.crop_data(linear_track)
            unix_start =  pd.to_numeric(linear_track.loc[linear_track['event'] == 'START', 'timestamp'].values[0])
            linear_track.loc[:, "frame"] = np.arange(len(linear_track))
            locations = linear_track[linear_track['event'] == 'LOCATION'].copy().reset_index(drop=True)
            data_out = linear_track[(linear_track["event"] != "START") & (linear_track["event"] != "TERMINATE")].copy().reset_index(drop=True)
            data_out['timestamp'] = data_out['timestamp'].astype(float)
            locations['timestamp'] = locations['timestamp'].astype(float)

            if any(np.diff(locations['timestamp']) < 0):
                print(f"Location time backwards jumps: {np.where(np.diff(locations['timestamp']) < 0)[0]}")
                jump_value = abs(np.min(np.diff(locations['timestamp'])))
                loc_idx = np.where(np.diff(locations['timestamp']) < 0)[0][0]
                jump_timestamp = locations.loc[loc_idx+1, 'timestamp']
                timestamp_idx = data_out[data_out['timestamp'] == jump_timestamp].index[0]
                data_out.loc[timestamp_idx:, 'timestamp'] = data_out['timestamp'][timestamp_idx:] + jump_value
            
            data_out = data_out.sort_values(by='timestamp').reset_index(drop=True)
            data_out['t'] = (data_out['timestamp'] - unix_start)
            pattern = "X(?P<x>[0-9]+)Y(?P<y>[0-9]+)"
            extracted = data_out['data'].str.extract(pattern)
            data_out[["x", "y"]] = extracted.astype(float)
            data_out["lick_port"] = -1
            data_out["water"] = False
            for idx, row in data_out.iterrows():
                port = int(row['data'][-1])
                if row['event'] == 'LICK':
                    data_out.loc[idx, 'lick_port'] = port
                    data_out['lick_port'] = pd.to_numeric(data_out['lick_port'])

                if row['event'] == 'REWARD':
                    data_out.loc[idx, 'water'] = True 
                    data_out.loc[idx, 'lick_port'] = port
            data_out['x'] = data_out['x'].ffill()
            data_out['y'] = data_out['y'].ffill()
            data_out[["animal", "cohort"]] = mouse, cohort_number
            data_out[['reward_one', 'reward_two']] = int(rewards[0][-1]), int(rewards[1][-1])
            data_out = (
                data_out.drop(columns=["event", "data"])
                .rename(columns={"timestamp": "unix"})
                .reset_index(drop=True)
            )
            if pd.isna(subset_log[i]):
                data_out['maze'] = 'No behavior log'
            else:
                data_out['maze'] = f'maze{subset_log[i][-5]}'
            result_path = pjoin(output_path, mouse)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            if i <= 8:
                data_out.to_feather(pjoin(result_path, "{}_{}.feat".format(mouse, session))) ## label with day number
            else:
                data_out.to_feather(pjoin(result_path, "{}_{}.feat".format(mouse, session)))
# %%