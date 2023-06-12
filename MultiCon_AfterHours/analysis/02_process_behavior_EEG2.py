# %%
import re
import sys
from os.path import join as pjoin

import numpy as np
import pandas as pd
from natsort import natsort_keygen
from tqdm import tqdm

sys.path.append("../..")
import circletrack_behavior as ctb

# %%
## Set parameters
session_dict = {
    'mc_EEG2_02': [
        'A1',
        'A2',
        'A3',
        'A4',
        'A5',
        'B1',
        'B2',
        'B3',
        'B4',
        'B5',
        'C1',
        'C2',
        'C3',
        'C4',
        'C5',
        'A2_1',
        'A2_2',
        'A2_3',
        'A2_4',
        'A2_5'
    ]
}
behavior_path = "../../../MultiCon_AfterHours/MultiCon_EEG2/circletrack_data"
output_path = "../../../MultiCon_AfterHours/MultiCon_EEG2/output/behav"
cohort_number = 'cohort1'
mouse_list = ['mc_EEG2_02']
downsample = False
if downsample:
    sampling_rate = 1/30 ## started keeping minian output at 1/30
    frame_count = 20 * 60 / sampling_rate ## session length in minutes x 60s per minute / sampling rate
## Set relative path variable for circletrack behavior data
csv_path = pjoin(behavior_path, "data/**/**/**/circle_track.csv")
log_path = pjoin(behavior_path, "data/**/**/**/**.log")
## Set str2match variable (regex for mouse name)
str2match = "(mc_EEG2_[0-9]+)"
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
    natsort_key = natsort_keygen()
    subset = ctb.subset_combined(combined_list, mouse).reset_index(drop=True)
    subset_log = ctb.subset_combined(combined_log, mouse).reset_index(drop=True)
    subset = sorted(subset, key=natsort_key)
    subset_log = sorted(subset_log, key=natsort_key)
    for i, session in tqdm(enumerate(session_dict[mouse]), leave=False):
        print(session)
        circle_track = pd.read_csv(subset[i])
        rewards = circle_track.loc[circle_track['event'] == 'initializing', 'data'].tolist()
        for idx in np.arange(0, len(circle_track['event'])):
            if 'probe' in circle_track['event'][idx]:
                probe_end = float(re.search('probe length: ([0-9]+)', circle_track['event'][idx])[1])
            else:
                next
        circle_track = ctb.crop_data(circle_track)
        unix_start =  circle_track.loc[circle_track['event'] == 'START', 'timestamp'].values[0]
        circle_track["frame"] = np.arange(len(circle_track))
        data_out = circle_track[circle_track["event"] == "LOCATION"].copy().reset_index(drop=True)
        if downsample:
            time_vector = np.arange(unix_start, (frame_count * sampling_rate + unix_start), sampling_rate)
            arg_mins = [np.abs(data_out['timestamp'] - t).argmin() for t in time_vector] ## resample to sampling freq of time_vector
            data_out = data_out.loc[arg_mins, :].reset_index(drop=True)
        events = circle_track[circle_track["event"] != "LOCATION"].copy().reset_index(drop=True)
        data_out['t'] = (pd.to_numeric(data_out['timestamp'] - unix_start))
        data_out[["x", "y", "a_pos"]] = (
            data_out["data"]
            .apply(
                lambda d: pd.Series(
                    re.search(
                        r"X(?P<x>[0-9]+)Y(?P<y>[0-9]+)A(?P<ang>[0-9]+)", d
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
        data_out[["animal", "session", "cohort"]] = mouse, session, cohort_number
        data_out["trials"] = ctb.get_trials(
            data_out, shift_factor=0, angle_type="radians", counterclockwise=True
        )
        data_out["lin_position"] = ctb.linearize_trajectory(
            data_out, angle_type="radians", shift_factor=0
        )
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
        data_out.to_feather(pjoin(result_path, "{}_{}.feat".format(mouse, session)))
# %%
