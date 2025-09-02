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
mouse_list =  ['mc03', 'mc06', 'mc07', 'mc09', 'mc11']    
dpath = '/media/caishuman/csstorage3/Austin/CircleTrack/EnsembleRemodeling_Resubmission/circletrack_data'
spath = '/media/caishuman/csstorage3/Austin/CircleTrack/EnsembleRemodeling_Resubmission/circletrack_data/assemblies'
for mouse in mouse_list:
    mpath = pjoin(spath, mouse)
    neural_dictionary = ctn.import_mouse_neural_data(dpath, mouse, key_file = 'minian_keys.yml', session = '30min', plot_frame_usage = False)
    ica.save_detected_ensembles(mpath, mouse, neural_dictionary, binarize = True, smooth_factor = 5, nullhyp = 'circ', n_shuffles = 500)