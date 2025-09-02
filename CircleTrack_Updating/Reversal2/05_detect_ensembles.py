# %%
import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
from os.path import join as pjoin
from tqdm import tqdm
from natsort import natsorted

sys.path.append('/home/austinbaggetta/csstorage3/CircleTrack/CircleTrackAnalysis')
import circletrack_neural as ctn
import pca_ica as ica

# %%
## Settings
project_dir = 'CircleTrack_Updating'
experiment_dir = 'Reversal2'
minian_path = os.path.abspath(f'../../../{project_dir}/{experiment_dir}/output/aligned_minian')
output_path = os.path.abspath(f'../../../{project_dir}/{experiment_dir}/output')
# mouse_list = ['rv03', 'rv04', 'rv05', 'rv07', 'rv09', 'rv11']
mouse_list = ['rv04']
data_type = 'S'
binarize = True
smooth_factor = 5
nshuffles = 500

for mouse in mouse_list:
    mpath = os.path.abspath(pjoin(minian_path, f'{mouse}/{data_type}'))
    for idx, session in enumerate(natsorted(os.listdir(mpath))):
        print(f'Detecting ensembles for {session}...')
        if (mouse == 'rv07') & (idx > 2):
            idx += 1

        if (mouse == 'rv07') & (idx > 7):
            idx += 1

        S = xr.open_dataset(pjoin(mpath, session))[data_type]

        if binarize:
            sbin = S > 0
        else:
            sbin = S
        
        if smooth_factor is not None:
            smoothed = ctn.moving_average(sbin, ksize=smooth_factor)
        else:
            smoothed = sbin.values
        
        assemblies = ica.find_assemblies(smoothed, nullhyp='circ', n_shuffles=nshuffles)
        act = assemblies['activations']
        pat = assemblies['patterns']

        activations = xr.DataArray(act, dims=['en_id', 'frame'], name='activations',
                                   coords={'en_id': np.arange(0, act.shape[0]), 'frame': S['frame'].values})
        patterns = xr.DataArray(pat, dims=['en_id', 'unit_id'], name='patterns',
                                coords={'en_id': np.arange(0, pat.shape[0]), 'unit_id': S['unit_id'].values})
        data = xr.merge([activations, patterns])

        data = data.assign_coords(behav_t=('frame', S['behav_t'].values), behav_frame=('frame', S['behav_frame'].values),
                                  x=('frame', S['x'].values), y=('frame', S['y'].values), a_pos=('frame', S['a_pos'].values),
                                  correct_dir=('frame', S['correct_dir'].values), lick_port=('frame', S['lick_port'].values),
                                  water=('frame', S['water'].values), trials=('frame', S['trials'].values),
                                  lin_position=('frame', S['lin_position'].values), probe=('frame', S['probe'].values))
        data = data.assign_attrs(dict(animal=S.attrs['animal'],
                                    session=S.attrs['session'],
                                    session_two=S.attrs['session_two'],
                                    cohort=S.attrs['cohort'],
                                    reward_one=S.attrs['reward_one'],
                                    reward_two=S.attrs['reward_two'],
                                    maze=S.attrs['maze'],
                                    date=S.attrs['date'],
                                    timestamp=S.attrs['timestamp']))
        
        spath = pjoin(output_path, f'ica_ensembles/{mouse}/')
        save_path = pjoin(spath, f'{mouse}_ensembles_{idx+1}.nc')
        if not os.path.exists(spath):
            os.makedirs(spath)
        data.to_netcdf(save_path)

# %%
