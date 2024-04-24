# %%
import xarray as xr
import os
from os.path import join as pjoin
import sys
sys.path.append('/media/caishuman/csstorage3/Austin/CircleTrack/CircleTrackAnalysis')
import pca_ica as ica

# %%
## Save assemblies of all sessions in the keys.yml file
mouse_list = ['mc23', 'mc26']
dpath = '../../../MultiCon_Imaging/MultiCon_Imaging2/output/aligned_minian'
save_path = '../../../MultiCon_Imaging/MultiCon_Imaging2/output/assemblies'
for mouse in mouse_list:
    dataset = {}
    mpath = pjoin(dpath, f'{mouse}/S')
    spath = pjoin(save_path, mouse)
    if not os.path.exists(spath):
        os.makedirs(spath)
    for session in os.listdir(mpath):
        if session == 'mc23_S_3.nc':
            pass 
        else:
            d = xr.open_dataset(pjoin(mpath, session))['S'] ## select S matrix
            dataset[session] = d
    ica.save_detected_ensembles(spath, dataset, binarize=False, smooth_factor=5, nullhyp='circ', n_shuffles=500)
