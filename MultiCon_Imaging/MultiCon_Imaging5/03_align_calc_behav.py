# %%
import os
import sys
import pandas as pd
from os.path import join as pjoin
from tqdm import tqdm
from natsort import natsorted
import re

sys.path.append('/home/austinbaggetta/csstorage3/CircleTrack/CircleTrackAnalysis')
import circletrack_neural as ctn

project_dir = 'MultiCon_Imaging'
experiment_dir = 'MultiCon_Imaging5'
minian_path = os.path.abspath(f'../../../{project_dir}/{experiment_dir}/minian_results')
output_path = os.path.abspath(f'../../../{project_dir}/{experiment_dir}/output')
mouse_list = ['mc44', 'mc46', 'mc48', 'mc49', 'mc51', 'mc52']
session_type = 'YrA'

for mouse in mouse_list:
    mpath = os.path.abspath(pjoin(minian_path, mouse))
    for idx, session in tqdm(enumerate(natsorted(os.listdir(mpath)))):
        print(f'Aligning {session}...')
        if (mouse == 'mc44') & (idx > 7):
            idx += 1
        elif (mouse == 'mc46') & (idx > 9):
            idx += 1
        elif (mouse == 'mc52') & (idx > 2):
            idx += 1
        dpath = pjoin(mpath, session)
        timestamp = os.listdir(dpath)[0] ## minian timestamp is first folder
        date = re.search('20[0-2][0-9]_[0-9]+_[0-9]+', dpath)[0]
        data_path = pjoin(dpath, f'{timestamp}/minian')
        spikes = ctn.open_minian(data_path)[session_type] ## select spike matrix

        spath = pjoin(output_path, f'aligned_minian/{mouse}/{session_type}')
        if not os.path.exists(spath):
            os.makedirs(spath)

        file_name = os.listdir(pjoin(output_path, f'behav/{mouse}'))[idx]
        behav = pd.read_feather(pjoin(output_path, f'behav/{mouse}/{file_name}'))
        save_path = pjoin(spath, f'{mouse}_{session_type}_{idx+1}.nc')
        ## Select session
        imaging_session = spikes.copy()
        ## Align calcium and behavior based on unix timestamps
        cropped_calc, aligned_behav = ctn.align_calcium_behavior(imaging_session, behav)
        ## Assign behavior data as coordinates
        cropped_calc = cropped_calc.assign_coords(behav_t=('frame', aligned_behav['t']),
                                                behav_frame=('frame', aligned_behav['frame']),
                                                x=('frame', aligned_behav['x']),
                                                y=('frame', aligned_behav['y']),
                                                a_pos=('frame', aligned_behav['a_pos']),
                                                correct_dir=('frame', aligned_behav['correct_dir']),
                                                lick_port=('frame', aligned_behav['lick_port']),
                                                water=('frame', aligned_behav['water']),
                                                trials=('frame', aligned_behav['trials']),
                                                lin_position=('frame', aligned_behav['lin_position']),
                                                probe=('frame', aligned_behav['probe']))
        cropped_calc = cropped_calc.assign_attrs(dict(animal=aligned_behav.loc[0, 'animal'],
                                                    session=aligned_behav.loc[0, 'session'],
                                                    session_two=aligned_behav.loc[0, 'session_two'],
                                                    cohort=aligned_behav.loc[0, 'cohort'],
                                                    reward_one=aligned_behav.loc[0, 'reward_one'],
                                                    reward_two=aligned_behav.loc[0, 'reward_two'],
                                                    maze=aligned_behav.loc[0, 'maze'],
                                                    date=date,
                                                    timestamp=timestamp))
        cropped_calc = cropped_calc.reset_coords(names='animal', drop=True)
        ## Save cropped_calc as a netcdf file
        cropped_calc.to_netcdf(save_path)
# %%
