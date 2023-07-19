import os
import sys
import numpy as np
import pandas as pd
from os.path import join as pjoin
from tqdm import tqdm
from natsort import natsorted

sys.path.append('/home/austinbaggetta/csstorage3/CircleTrack/CircleTrackAnalysis')
import circletrack_neural as ctn
import place_cells as pc 

behavior_path = os.path.abspath('../MultiCon_AfterHours/MultiCon_EEG1/output')
mouse_dict = {'mc_EEG1_01': ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5',
                             'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5']}
alpha = 0.05

for mouse in mouse_dict.keys():
    minian_path = os.path.abspath(f'../MultiCon_AfterHours/MultiCon_EEG1/minian_result/{mouse}/minian/')
    spikes = ctn.open_minian(minian_path)['S']
    spath = pjoin(behavior_path, f'aligned_minian/{mouse}')

    for idx, session in tqdm(enumerate(natsorted(np.unique(spikes['session_id'].values)))):
        session_name = mouse_dict[mouse][idx]
        file_name = '_'.join([mouse, session_name])
        behav = pd.read_feather(pjoin(behavior_path, f'behav/{mouse}/{file_name}.feat'))
        save_path = pjoin(spath, f'{file_name}.nc')
        ## Select session
        imaging_session = spikes[:, spikes['session_id'].values == session].copy()
        ## Align calcium and behavior based on unix timestamps
        cropped_calc, aligned_behav = ctn.align_calcium_behavior(imaging_session, behav)
        ## Assign behavior data as coordinates
        cropped_calc = cropped_calc.assign_coords(behav_t=('frame', aligned_behav['t']),
                                                behav_frame=('frame', aligned_behav['frame']),
                                                x=('frame', aligned_behav['x']),
                                                y=('frame', aligned_behav['y']),
                                                a_pos=('frame', aligned_behav['a_pos']),
                                                lick_port=('frame', aligned_behav['lick_port']),
                                                water=('frame', aligned_behav['water']),
                                                trials=('frame', aligned_behav['trials']),
                                                lin_position=('frame', aligned_behav['lin_position']),
                                                probe=('frame', aligned_behav['probe']))
        cropped_calc = cropped_calc.assign_attrs(dict(animal=aligned_behav.loc[0, 'animal'],
                                                    session=aligned_behav.loc[0, 'session'],
                                                    cohort=aligned_behav.loc[0, 'cohort'],
                                                    reward_one=aligned_behav.loc[0, 'reward_one'],
                                                    reward_two=aligned_behav.loc[0, 'reward_two'],
                                                    maze=aligned_behav.loc[0, 'maze']))
        cropped_calc = cropped_calc.reset_coords(names='animal', drop=True)
        ## Calculate place fields
        pf_object = pc.PlaceFields(x=cropped_calc['x'].values,
                                   y=cropped_calc['y'].values,
                                   t=cropped_calc['behav_t'].values,
                                   neural_data=cropped_calc.values,
                                   circular=True,
                                   nbins=20)
        ## Assign place cell boolean as a coordinate
        cropped_calc = cropped_calc.assign_coords(place_cell = ('unit_id', pf_object.data['spatial_info_pvals'] < alpha))
        ## Save cropped_calc as a netcdf file
        cropped_calc.to_netcdf(save_path)