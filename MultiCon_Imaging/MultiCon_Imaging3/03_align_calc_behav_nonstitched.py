import os
import sys
import pandas as pd
from os.path import join as pjoin
from tqdm import tqdm
from natsort import natsorted

sys.path.append('/home/austinbaggetta/csstorage3/CircleTrack/CircleTrackAnalysis')
import circletrack_neural as ctn

minian_path = os.path.abspath('../MultiCon_Imaging/MultiCon_Imaging3/minian_results')
output_path = os.path.abspath('../MultiCon_Imaging/MultiCon_Imaging3/output')
session_type = 'C'
# for mouse in os.listdir(minian_path):
for mouse in ['mc27', 'mc28']:
    mpath = os.path.abspath(pjoin(minian_path, mouse))
    for idx, session in tqdm(enumerate(natsorted(os.listdir(mpath)))):
        print(f'Aligning {session}...')
        dpath = pjoin(mpath, session)
        timestamp = os.listdir(dpath)[0] ## minian timestamp is first folder
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
                                                    maze=aligned_behav.loc[0, 'maze']))
        cropped_calc = cropped_calc.reset_coords(names='animal', drop=True)
        ## Save cropped_calc as a netcdf file
        cropped_calc.to_netcdf(save_path)