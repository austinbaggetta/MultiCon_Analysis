# %%
import re
import sys
from os.path import join as pjoin

import numpy as np
import pandas as pd
from natsort import natsort_keygen
from tqdm import tqdm

sys.path.append("..")
import circletrack_behavior as ctb

# %%
## Set parameters
session_dict = {
    "mc03": [
        "Training1",
        "Training2",
        "Training3",
        "Training4",
        "Reversal1",
        "Reversal2",
        "Reversal4",
        "Training_Reversal",
    ],
    "mc06": [
        "Training1",
        "Training2",
        "Training3",
        "Training4",
        "Reversal1",
        "Reversal2",
        "Reversal3",
        "Reversal4",
        "Training_Reversal",
    ],
    "mc07": [
        "Training1",
        "Training2",
        "Training3",
        "Training4",
        "Reversal1",
        "Reversal2",
        "Reversal3",
        "Reversal4",
        "Training_Reversal",
    ],
    "mc09": [
        "Training1",
        "Training2",
        "Training3",
        "Training4",
        "Reversal1",
        "Reversal2",
        "Reversal3",
        "Reversal4",
        "Training_Reversal",
    ],
    "mc11": [
        "Training1",
        "Training2",
        "Training3",
        "Training4",
        "Reversal1",
        "Reversal2",
        "Reversal3",
        "Reversal4",
        "Training_Reversal",
    ],
}
behavior_path = "../../EnsembleRemodeling_Resubmission/circletrack_data"
output_path = "../../EnsembleRemodeling_Resubmission/circletrack_data/output/behav"
cohort_number = "cohort1"
mouse_list = ["mc03", "mc06", "mc07", "mc09", "mc11"]
## Set relative path variable for circletrack behavior data
path = pjoin(behavior_path, "Data/**/**/**/circle_track.csv")
## Set str2match variable (regex for mouse name)
str2match = "(mc[0-9]+)"
## Create list of files
file_list = ctb.get_file_list(path)
## Loop through file_list to extract mouse name
mouseID = []
for file in file_list:
    mouse = ctb.get_mouse(file, str2match)
    mouseID.append(mouse)
## Combine file_list and mouseID
combined_list = ctb.combine(file_list, mouseID)


# %%
for mouse in tqdm(mouse_list):
    natsort_key = natsort_keygen()
    subset = ctb.subset_combined(combined_list, mouse).reset_index(drop=True)
    subset = sorted(subset, key=natsort_key)
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
        circle_track = ctb.normalize_timestamp(circle_track).reset_index(drop=True)
        circle_track["frame"] = np.arange(len(circle_track))
        data_out = circle_track[circle_track["event"] == "LOCATION"].copy()
        events = circle_track[circle_track["event"] != "LOCATION"].copy()
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
            idx = data_out.iloc[np.argmin(np.abs(data_out["timestamp"] - ts))].name
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
            .rename(columns={"timestamp": "t"})
            .reset_index(drop=True)
        )
        data_out['probe'] = data_out['t'] < probe_end
        result_path = pjoin(output_path, mouse)
        data_out.to_feather(pjoin(result_path, "{}_{}.feat".format(mouse, session)))
# %%
