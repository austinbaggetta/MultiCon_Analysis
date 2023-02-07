# %%
import pandas as pd
import numpy as np
import xarray as xr
import os
from tqdm import tqdm
from os.path import join as pjoin
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr, zscore, wilcoxon
import sys

# %%
sys.path.append('/media/caishuman/csstorage3/Austin/CircleTrack/CircleTrackAnalysis')
import circletrack_neural as ctn
import circletrack_behavior as ctb
import pca_ica as ica
import plotting_functions as pf

# %%
## Set path variables
dpath = '/media/caishuman/csstorage/phild/git/MazeProjects/output/'
mouse = 'Fornax'
session = 'Reversal1'
minian_path = pjoin(dpath, 'processed/{}_{}.nc'.format(mouse, session))
minian_data = xr.open_dataset(minian_path)
## Detect assemblies
neural_data = minian_data.S_bin.values
smoothed_data = ctn.moving_average(neural_data, ksize = 5)


# %%
zdata = zscore(smoothed_data, axis = 1)


# %%
assemblies = ica.find_assemblies(smoothed_data, nullhyp = 'circ', n_shuffles = 10)


