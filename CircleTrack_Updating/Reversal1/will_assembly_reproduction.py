# %%
import xarray as xr
from os.path import join as pjoin
import sys

# %%
sys.path.append('/media/caishuman/csstorage3/Austin/CircleTrack/CircleTrackAnalysis')
import circletrack_neural as ctn
import pca_ica as ica

#%%
## Set path variables
dpath = '/media/caishuman/csstorage/phild/git/MazeProjects/output/'
mouse = 'Fornax'
session = 'Reversal1'
minian_path = pjoin(dpath, 'processed/{}_{}.nc'.format(mouse, session))
minian_data = xr.open_dataset(minian_path)
## Detect assemblies
neural_data = minian_data.S_bin.values
smoothed_data = ctn.moving_average(neural_data, ksize = 5)

# # %%
assemblies = ica.find_assemblies(smoothed_data, nullhyp = 'circ', n_shuffles = 500)
