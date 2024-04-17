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

# %%
## Set parameters
behavior_path = os.path.abspath("../../../AD_Updating/AD_Updating1/lineartrack_data/")
output_path = os.path.abspath("../../../AD_Updating/AD_Updating1/output/lin_behav")
cohort_number = 'z_1'
mouse_list = [f'Z{x}' for x in np.arange(33, 39)] + ['Z40'] + [f'Z{x}' for x in np.arange(46, 50)] + [f'Z{x}' for x in np.arange(51, 55)] + ['Z63', 'Z64', 'Z66']
downsample = False
if downsample:
    sampling_rate = 1/30 ## started keeping minian output at 1/30
    frame_count = 20 * 60 / sampling_rate ## session length in minutes x 60s per minute / sampling rate
## Set relative path variable for circletrack behavior data
csv_path = pjoin(behavior_path, "data/**/**/**/linear_track.csv")
## Set str2match variable (regex for mouse name)
str2match = "(Z[0-9]+)"
## Create list of files
file_list = ctb.get_file_list(csv_path)
## Loop through file_list to extract mouse name
mouseID = []
for file in file_list:
    mouse = ctb.get_mouse(file, str2match)
    mouseID.append(mouse)
## Combine file_list and mouseID
combined_list = ctb.combine(file_list, mouseID)

# %%
for mouse in mouse_list:
    print(mouse)
    natsort_key = natsort_keygen()
    subset = ctb.subset_combined(combined_list, mouse).reset_index(drop=True)
    subset = sorted(subset, key=natsort_key)
    for i, session in tqdm(enumerate(np.arange(0, 5)), leave=False): ## five days of linear track
        print(session)
        linear_track = pd.read_csv(subset[i])
        reward_one, reward_two = ['reward1', 'reward2']
        rewards = [reward_one, reward_two]
        linear_track = ctb.crop_data(linear_track)
        unix_start =  pd.to_numeric(linear_track.loc[linear_track['event'] == 'START', 'timestamp'].values[0])
        linear_track["frame"] = np.arange(len(linear_track))
        data_out = linear_track[linear_track["event"] == "LOCATION"].copy().reset_index(drop=True)
        data_out['timestamp'] = pd.to_numeric(data_out['timestamp']) ## sometimes saved as string, not sure why
        if downsample:
            time_vector = np.arange(unix_start, (frame_count * sampling_rate + unix_start), sampling_rate)
            arg_mins = [np.abs(data_out['timestamp'] - t).argmin() for t in time_vector] ## resample to sampling freq of time_vector
            data_out = data_out.loc[arg_mins, :].reset_index(drop=True)
        events = linear_track[linear_track["event"] != "LOCATION"].copy().reset_index(drop=True)
        events['timestamp'] = pd.to_numeric(events['timestamp'])
        data_out['t'] = (data_out['timestamp'] - unix_start)
        data_out[["x", "y"]] = (
            data_out["data"]
            .apply(
                lambda d: pd.Series(
                    re.search(
                        r"X(?P<x>[0-9]+)Y(?P<y>[0-9]+)", d
                    ).groupdict()
                )
            )
            .astype(float)
        )
        data_out["lick_port"] = -1
        data_out["water"] = False
        for _, row in events.iterrows():
            ts = row["timestamp"]
            idx = np.argmin(np.abs(data_out["timestamp"] - ts))
            try:
                port = int(row["data"][-1])
            except TypeError:
                continue
            data_out.loc[idx, "lick_port"] = port
            if row["event"] == "REWARD":
                data_out.loc[idx, "water"] = True
        data_out[["animal", "cohort"]] = mouse, cohort_number
        data_out[['reward_one', 'reward_two']] = int(rewards[0][-1]), int(rewards[1][-1])
        data_out = (
            data_out.drop(columns=["event", "data"])
            .rename(columns={"timestamp": "unix"})
            .reset_index(drop=True)
        )
        result_path = pjoin(output_path, mouse)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if i <= 8:
            data_out.to_feather(pjoin(result_path, "{}_{}.feat".format(mouse, session))) ## label with day number
        else:
            data_out.to_feather(pjoin(result_path, "{}_{}.feat".format(mouse, session)))
# %%