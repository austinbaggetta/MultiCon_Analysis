import os
import pickle
import yaml
from os import listdir
from os.path import isdir
from os.path import join as pjoin

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pylab as pl
import xarray as xr
from numpy.polynomial.polynomial import polyfit
from plotly.subplots import make_subplots
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from skimage.filters import threshold_otsu
from scipy.stats import pearsonr, zscore, spearmanr
from scipy.ndimage import convolve
import itertools
## Austin's custom py files
import pca_ica as ica


def get_seconds(time_str, character='_'):
    """
    Takes the timestamp from the computer format and converts to seconds.
    Args:
        time_str : str
            folder timestamp
        character : str
            character that separates the hh_mm_ss (hh:mm:ss)
    Returns:
        seconds : int
    """
    hh, mm, ss = time_str.split(character)
    return int(hh) * 3600 * int(mm) * 60 + int(ss)


def open_minian(dpath, post_process=None, return_dict=False):
    """
    Opens a file previously saved in minian handling the proper data format and chunks
    Args:
        dpath ([string]): contains the normalized absolutized version of the pathname path,which is the path to minian folder;
        Post_process (function): post processing function, parameters: dataset (xarray.DataArray), mpath (string, path to the raw backend files)
        return_dict ([boolean]): default False
    Returns:
        xarray.DataArray: [loaded data]
    """
    dslist = [
        xr.open_zarr(pjoin(dpath, d), consolidated=False)
        for d in os.listdir(dpath)
        if isdir(pjoin(dpath, d)) and d.endswith(".zarr")
    ]
    if return_dict:
        dslist = [list(d.values())[0] for d in dslist]
        ds = {d.name: d for d in dslist}
    else:
        ds = xr.merge(dslist, compat="no_conflicts")
    if (not return_dict) and post_process:
        ds = post_process(ds, dpath)
    return ds
 

def calculate_activity_correlation(first_session, second_session, test='spearman'):
    """
    Calculates the pearson or spearman correlation coefficients for two sessions.
    Uses each cell's average activity across the session.
    Args:
        first_session, second_session: xarray.DataArray
            DataArray of either spike or calcium trace activity
        test: str
            options are ['pearson', 'spearman']
    Returns:
        res : list
            res[0] gives the test statistic, res[1] gives the pvalue
    """
    ## Calculate mean activity
    avg_first_session = first_session.values.mean(axis=1)
    avg_second_session = second_session.values.mean(axis=1)
    ## Perform correlation test
    if test == 'pearson':
        res = pearsonr(avg_first_session, avg_second_session)
    elif test == 'spearman':
        res = spearmanr(avg_first_session, avg_second_session)
    else:
        raise Exception('No test selected!')
    return res


def moving_average(data, ksize = 5):
    kernel = np.ones(ksize)/ksize
    result = np.empty([data.shape[0], data.shape[1]])
    for i in np.arange(0, data.shape[0]):
        result[i] = convolve(input=data[i], output='float', weights=kernel, mode='nearest')
    return result


def calculate_event_rates(data, bin_size_sec, fps=15, zscore=False):
    """
    Calculate event rate.
    Args: 
    Returns:
        event_rates : np.array
            event rate of each cell within each time bin, cell x bin array
    """
    if zscore:
        activity = zscore(data, axis=1) ## z-score each cell
    else:
        activity = data
    num_spikes = ica.bin_transients(activity, bin_size_sec, fps=fps, analysis_type='num_spikes')
    event_rates = num_spikes / bin_size_sec
    return event_rates


def cell_quantiles(cell_array, quantile, test, quantile_lower=None):
    """
    Get indices of cells that satisfy the given condition.
    Args:
        cell_array : numpy.ndarray
            array of cell values; for example, average event rates for cells
        quantile : float
            quantile, will get cells whose activity is above or below this value
        test : str
            greater than, less than, within
        quantile_lower : float
            by default None; only needed if test == 'within'
    Returns:
        output : numpy.ndarray
            cell indices that satisfy conditions
    """
    if test == 'greater_than':
        output = np.where(cell_array > np.quantile(cell_array, quantile))[0]
    elif test == 'less_than':
        output = np.where(cell_array < np.quantile(cell_array, quantile))[0]
    elif test == 'within':
        output = np.where((cell_array < quantile) & (cell_array > quantile_lower))[0]
    else:
        print('Must be greater_than, less_than, or within!')
    return output


## Works on xarray.DataArray
def align_start_end_times(act, behav, col_name='unix'):
    """
    Crops miniscope data to start and end when the behavior data starts and ends. For experiments without a ttl on circle track.
    Args:
        act : xarray.DataArray
            calcium activity (C, S, S_binarized) with unix timestamps
        behav : pandas.DataFrame
            behavior data containing unix timestamps
    Returns:
        xarray.DataArray (S_shifted) where frame start and end is when behavior starts and ends
    """
    act_copy = act.copy()
    start_idx = np.abs(behav.loc[0, col_name] - act_copy[col_name].values).argmin()
    end_idx = np.abs(behav[col_name].tail(1).to_numpy() - act_copy[col_name].values).argmin()
    return act_copy[:, start_idx:end_idx]


def align_calcium_behavior(act, behav, col_name='unix'):
    """
    Aligns calcium imaging data to behavior data for place cell analyses.
    Args:
        act : xarray.DataArray
           calcium activity (C, S, S_binarized) with unix timestamps
        behav : pandas.DataFrame
             behavior data containing unix timestamps
        col_name : str
            one of ['unix', 'timestamps'], depending on whether you are using unix timestamps or millisecond timestamps
    Returns:
        act_shifted : xarray.DataArray cropped
        pandas.DataFrame with aligned indices and all columns
    """
    act_shifted = align_start_end_times(act, behav, col_name=col_name)
    events = behav[(behav['lick_port'] != -1) | (behav['water'])].reset_index(drop=True)
    timestamps_calc = act_shifted[col_name].values
    if np.where(np.diff(timestamps_calc) < 0)[0].size > 0:
        jump_value = abs(np.min(np.diff(timestamps_calc)))
        timestamp_idx = np.where(np.diff(timestamps_calc) < 0)[0][0]
        timestamps_calc[timestamp_idx+1:] = timestamps_calc[timestamp_idx+1:] + jump_value
    timestamps_calc = timestamps_calc.reshape(len(timestamps_calc), 1)
    timestamps_behav = np.array(behav.loc[:, col_name])
    if col_name == 'timestamps':
        timestamps_behav = np.array(behav.loc[:, col_name]) / 1000 ## convert to seconds
    aligned_indices = np.abs(timestamps_calc - timestamps_behav).argmin(axis=1)
    aligned_behav = behav.loc[aligned_indices, :].reset_index(drop=True)
    ## Re-insert events. Captures all rewards, but some licks are dropped due to how fast the licking is.
    aligned_behav['lick_port'] = -1
    aligned_behav['water'] = False
    timestamps_events = np.array(events.loc[:, col_name])
    timestamps_aligned_behav = np.array(aligned_behav.loc[:, col_name])
    timestamps_events = timestamps_events.reshape(len(timestamps_events), 1)
    new_aligned_indices = np.abs(timestamps_events - timestamps_aligned_behav).argmin(axis=1)
    for event_idx, aligned_idx in enumerate(new_aligned_indices):
        aligned_behav.loc[aligned_idx, :] = events.loc[event_idx, :]
    aligned_behav = aligned_behav.sort_values(by=col_name).reset_index(drop=True)
    return act_shifted, aligned_behav


def calculate_otsu_thresh(spike_data):
    """
    Calculate the otsu threshold for the spike amplitude histogram.
    Used to binarize approximated spike data (since S has some noise)
    Args:
        spike_data : xarray.DataArray or numpy.ndarray
            output from Minian or aligned calcium and behavior data
    Returns:
        T : float
            otsu threshold to minimize intra-class variance.
    """
    counts, bins = np.histogram(spike_data)
    center = (bins[:-1] + bins[1:]) / 2
    hist = (counts, center)
    return threshold_otsu(hist=hist)


def extract_windowed_data(data, window_val, window_size, col_name=None):
    """
    Used to extract the data around a point of interest.
    """
    if col_name is not None:
        d = data[(data[col_name] >= window_val[col_name] - window_size) & (data[col_name] <= window_val[col_name] + window_size)]
    else:
        d = data[(data >= window_val - window_size) & (data <= window_val + window_size)]
    return d


def bootstrap_z_values(val, shuffled_val):
    z_values = (val - np.mean(shuffled_val, axis=0)) / np.std(shuffled_val, axis=0, ddof=1)
    return z_values
