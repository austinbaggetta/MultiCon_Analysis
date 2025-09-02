# %%
import re
import sys
import os
from os.path import join as pjoin

import numpy as np
import pandas as pd
from natsort import natsort_keygen
from tqdm import tqdm

sys.path.append('/home/austinbaggetta/csstorage3/CircleTrack/CircleTrackAnalysis')
import circletrack_behavior as ctb
import plotting_functions as pf

# %%
## Set parameters
starting_idx = 0 ## can use to specify which days you want processed
parent_dir = 'MultiCon_Behavior'
experiment_dir = 'MultiCon_Lesion3'
todays_mazes = pd.read_csv(f'../../../{parent_dir}/{experiment_dir}/maze_yml/{experiment_dir} - TodaysMazes.csv')
todays_mazes_type2 = pd.read_csv(f'../../../{parent_dir}/{experiment_dir}/maze_yml/{experiment_dir} - TodaysMazes2.csv')
behavior_path = os.path.abspath(f"../../../{parent_dir}/{experiment_dir}/circletrack_data/")
output_path = os.path.abspath(f"../../../{parent_dir}/{experiment_dir}/output/behav")
save_path = os.path.abspath(f"../../../{parent_dir}/{experiment_dir}/output/behav_preprocessing_plots")
if not os.path.exists(output_path):
    os.makedirs(output_path)
cohort_name = 'mcls3'
mouse_list = ['mcls47', 'mcls48', 'mcls49', 'mcls51'] + [f'mcls{x}' for x in np.arange(53, 66)]
str2match = "(mcls[0-9]+)" ## Set str2match variable (regex for mouse name)
## Set relative path variable for circletrack behavior data
csv_path = pjoin(behavior_path, "data/**/**/**/circle_track.csv")
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
    for i, session in tqdm(enumerate(todays_mazes.columns[1:]), leave=False): ## start from index 1 since index 0 is mouseID
        if i < starting_idx:
            pass
        else:
            print(session)
            circle_track = pd.read_csv(subset[i])
            reward_ports = ctb.get_rewarding_ports(circle_track)
            rewards = [x for x in reward_ports['data']]
            for idx in np.arange(0, len(circle_track['event'])):
                if 'probe' in circle_track['event'][idx]:
                    probe_end = float(re.search('probe length: ([0-9]+)', circle_track['event'][idx])[1])
                else:
                    pass
            circle_track = ctb.crop_data(circle_track)
            unix_start =  pd.to_numeric(circle_track.loc[circle_track['event'] == 'START', 'timestamp'].values[0])
            circle_track.loc[:, "frame"] = np.arange(len(circle_track))
            locations = circle_track[circle_track['event'] == 'LOCATION'].copy().reset_index(drop=True)
            data_out = circle_track[(circle_track["event"] != "START") & (circle_track["event"] != "TERMINATE")].copy().reset_index(drop=True)
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
            pattern = "X(?P<x>[0-9]+)Y(?P<y>[0-9]+)A(?P<ang>[0-9]+)"
            extracted = data_out['data'].str.extract(pattern)
            data_out[["x", "y", "a_pos"]] = extracted.astype(float)
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
            data_out['a_pos'] = data_out['a_pos'].ffill()
            data_out["lin_position"] = data_out["a_pos"] * (np.pi/180)
            data_out['correct_dir'] = ctb.get_correct_direction(data_out['a_pos'])
            data_out["trials"] = ctb.get_trials(data_out["a_pos"])
            data_out[["animal", "session", "cohort"]] = mouse, todays_mazes[session][todays_mazes['Mouse'] == mouse].tolist()[0], cohort_name
            data_out['session_two'] = todays_mazes_type2[session][todays_mazes_type2['Mouse'] == mouse].tolist()[0]
            data_out[['reward_one', 'reward_two']] = int(rewards[0][-1]), int(rewards[1][-1])
            data_out = (
                data_out.drop(columns=["event", "data"])
                .rename(columns={"timestamp": "unix"})
                .reset_index(drop=True)
            )
            data_out['probe'] = data_out['t'] < probe_end
            if pd.isna(subset_log[i]):
                data_out['maze'] = 'No behavior log'
            else:
                data_out['maze'] = subset_log[i][-9:-4]
            result_path = pjoin(output_path, mouse)
            spath = pjoin(save_path, mouse)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            if not os.path.exists(spath):
                os.makedirs(spath)
            if i <= 8:
                data_out.to_feather(pjoin(result_path, f"{mouse}_{session[-1]}.feat")) ## label with day number
            else:
                data_out.to_feather(pjoin(result_path, f"{mouse}_{session[-2:]}.feat"))
            
            fig = pf.preprocessed_plots(data_out, angle_type='degrees', save_path=spath)
# %%
