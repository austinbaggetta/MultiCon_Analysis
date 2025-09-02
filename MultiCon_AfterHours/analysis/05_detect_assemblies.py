# %%
import xarray as xr
from os.path import join as pjoin
import sys

# %%
sys.path.append('/media/caishuman/csstorage3/Austin/CircleTrack/CircleTrackAnalysis')
import circletrack_neural as ctn
import pca_ica as ica

# %%
## Save assemblies of all sessions in the keys.yml file
mouse_dict = {'mc_EEG1_01': ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5',
                             'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5']}
dpath = '/media/caishuman/csstorage3/Austin/CircleTrack/MultiCon_AfterHours/MultiCon_EEG1/output/aligned_minian'
spath = '/media/caishuman/csstorage3/Austin/CircleTrack/MultiCon_AfterHours/MultiCon_EEG1/output/assemblies'
dataset = {}
for mouse in mouse_dict.keys():
    mpath = pjoin(spath, mouse)
    for session in mouse_dict[mouse]:
        file_name = '_'.join([mouse, session])
        da = xr.open_dataset(pjoin(dpath, f'{mouse}/{file_name}.nc'))['S'] ## select S matrix
        dataset[session] = da
    ica.save_detected_ensembles(mpath, mouse, dataset, binarize=False, smooth_factor=5, nullhyp='circ', n_shuffles=500)
# %%
