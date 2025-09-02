# %%
import re
import sys
import os
from os.path import join as pjoin

import numpy as np
import pandas as pd
import datetime

sys.path.append('/home/austinbaggetta/csstorage3/CircleTrack/CircleTrackAnalysis')
import circletrack_behavior as ctb

# %%
## Set parameters
behavior_path = os.path.abspath("../../../AD_Updating/AD_Updating1/openfield_data/")
output_path = os.path.abspath("../../../AD_Updating/AD_Updating1/output/openfield_behav")
cohort_number = 'z_1'
# mouse_list = [f'Z{x}' for x in np.arange(33, 39)] + ['Z40'] + [f'Z{x}' for x in np.arange(46, 50)] + [f'Z{x}' for x in np.arange(51, 55)] + ['Z63', 'Z64', 'Z66']
mouse_list = ['Z38', 'Z40', 'Z33', 'Z36', 'Z48', 'Z49', 'Z63', 'Z66'] + ['Z34', 'Z35', 'Z37', 'Z46', 'Z47', 'Z51', 'Z53', 'Z54', 'Z64']
sampling_rate = 1/29.608 ## Determined from Z33 having 17765 frames and a 10 minute recording
downsample = False
if downsample:
    frame_count = 10 * 60 / sampling_rate ## session length in minutes x 60s per minute / sampling rate
## Set relative path variable for circletrack behavior data
csv_path = pjoin(behavior_path, "**/**/**/LocationOutput.csv")
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
    subset = ctb.subset_combined(combined_list, mouse).reset_index(drop=True)
    open_field = pd.read_csv(subset[0])

    date = re.search('20[0-2][0-9]_[0-9]+_[0-9]+', subset[0])[0]
    date_format = datetime.datetime.strptime(f"{date}, {subset[0][-27:-19]}", "%Y_%m_%d, %H_%M_%S")
    unix_start = datetime.datetime.timestamp(date_format)

    data_out = pd.DataFrame()
    data_out['frame'] = pd.to_numeric(open_field['Frame']) ## sometimes saved as string, not sure why
    data_out['t'] = np.arange(0, open_field.shape[0] * sampling_rate, sampling_rate)
    data_out['timestamp'] = data_out['t'] + unix_start
    if downsample:
        time_vector = np.arange(unix_start, (frame_count * sampling_rate + unix_start), sampling_rate)
        arg_mins = [np.abs(data_out['timestamp'] - t).argmin() for t in time_vector] ## resample to sampling freq of time_vector
        data_out = data_out.loc[arg_mins, :].reset_index(drop=True)
    data_out['t'] = (data_out['timestamp'] - unix_start)
    data_out['x'] = open_field['X']
    data_out['y'] = open_field['Y']
    delta = np.diff(np.asarray((open_field['X'], open_field['Y'])).T, axis=0)
    dists = np.hypot(delta[:, 0], delta[:, 1])
    dists = np.insert(dists, 0, 0)
    data_out['euc_distance'] = dists
    data_out[["animal", "cohort"]] = mouse, cohort_number
    result_path = pjoin(output_path, mouse)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    data_out.to_feather(pjoin(result_path, f"{mouse}_openfield.feat"))
# %%