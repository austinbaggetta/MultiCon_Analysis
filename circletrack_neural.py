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
from scipy.stats import pearsonr, zscore, spearmanr
import itertools


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
        xr.open_zarr(pjoin(dpath, d))
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


def align_miniscope_frames(minian_timestamps, time, plot_frame_usage=False, down_sample = True, down_sample_factor = 2, ttl = False):
    """
    Takes timestamps matrix associated with a miniscope recording and a regularly spaced time vector the expected length of the session. 
    For each timeframe in 'time', the closest frame from minian_timestamps is acquired.
    Some frames are used more than once. 
    Args:
        minian_timestamps : pandas.DataFrame
            minian timestamps from preprocessing; argument set in load_and_align_minian function
        time : list
            regularly spaced time vector the expected length of the session; argument set in load_and_align_minian function
        plot_frame_usage : boolean
            if True, creates a plot of frame usage; by default set to False
        downsample: boolean
            whether or not minian was downsampled during pre-processing; by default True
        downsample_factor: float
            what factor the data was downsampled by
        ttl: boolean
            whether or not a ttl was used to trigger miniscope imaging
    Returns:
        lined_up_timeframes : list
            vector of lined up frames to use to align recording to the time vector.
    """
    ## If Minian was processed with temporal downsampling, set downsample to True so that minian_timestamps lines up with neural data
    if down_sample:
        minian_timestamps = minian_timestamps[::down_sample_factor]
    ## If ttl triggered, first 5 frames are dark, so exclude them
    if ttl:
        minian_timestamps[4:]
    ## Determine which frame is closest to the ideal time vector
    arg_mins = [np.abs(minian_timestamps["Time Stamp (ms)"] - (t * 1000)).argmin() for t in time]
    lined_up_timeframes = np.array(minian_timestamps['Frame Number'].values[arg_mins])

    if plot_frame_usage:
        duplicated_timeframes = np.unique(lined_up_timeframes, return_counts=True)
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=duplicated_timeframes[0],
                y=duplicated_timeframes[1],
                mode="lines+markers",
                marker_size=5,
            )
        )
        fig.update_layout(
            template="simple_white",
            xaxis_title="Time (ms)",
            yaxis_title="Frequency",
            title_text="Number of times each miniscope frame was re-used",
        )
        fig.show(config={"scrollZoom": True})

    return lined_up_timeframes


def load_and_align_minian(path, mouse, date, session = '20min', neural_type="spikes", sigma=None, sampling_rate=1/15, downsample = False, downsample_factor=2, plot_frame_usage = True):
    """
    Loads minian data and aligns the data to a regularly spaced time vector.
    Args:
        dpath : str
            experiment directory
        mouse : str
            name of the mouse (e.g. 'mc01')
        date : str
            date of session
        timestamp : str
            timestamp of session
        session : str
            one of ['20min'], may incorporate other options if length of sessions decrease from 20min
        neural_type : str
            one of ['traces', 'spikes', 'smoothed']
        sigma : int
            smoothing kernel if neural_type=smoothed
        sampling_rate : float
            sampling rate of desired time vector; by default 1/15 since I downsample Minian by a factor of 2
        downsample : boolean
            if you want to further downsample from 15 fps
        downsample_factor : int
            factor by which you want to downsample by
    Returns:
        neural_activity : xarray.DataArray
            can be aligned spike or calcium trace data
    """
    # Create time vector based on expected length of session
    if session == '20min':
        frame_count = (
            20 * 60 / sampling_rate
        )  # 20min x 60sec/min / sampling_rate 
    elif session == '30min':
        frame_count = (
            30 * 60 / sampling_rate
        )  # 20min x 60sec/min / sampling_rate 
    else:
        raise Exception(
            "Invalid 'session' argument. Must be one of: ['20min']"
        )
    time = np.arange(0, frame_count * sampling_rate, sampling_rate)
    # load the specified type of neural activity
    rpath = pjoin(path, 'Results/{}/{}/'.format(mouse, date))
    timestamp = os.listdir(rpath) ## get timestamp associated with that day
    rpath = pjoin(rpath, timestamp[0]) 
    rpath = pjoin(rpath, 'minian') ## navigate to .zarr files
    mouse_minian = open_minian(rpath, return_dict=False)
    if neural_type == "traces":
        neural_activity = mouse_minian["C"]
    elif (neural_type == "spikes") or (neural_type == "smoothed"):
        neural_activity = mouse_minian["S"]
    else:
        raise Exception(
            "Not a valid 'neural_type'; must be one of ['traces', 'spikes', 'smoothed']."
        )
    tpath = pjoin(path, 'Data/{}/{}/{}/miniscope'.format(mouse, date, timestamp[0]))
    minian_timestamps = pd.read_csv(tpath + "/timeStamps.csv")
    ## If you downsampled during minian processing, can change downsample_factor
    lined_up_timeframes = align_miniscope_frames(minian_timestamps, time, plot_frame_usage = plot_frame_usage)
    if downsample:
        lined_up_timeframes = lined_up_timeframes[::downsample_factor]
    ## Select frames based on line-up timeframes
    neural_activity = neural_activity.sel(frame=lined_up_timeframes)
    if (
        neural_type == "smoothed"
    ):  # this filtering must be done after the previous line because this converts neural_activity to numpy array
        neural_activity = gaussian_filter(neural_activity, sigma=(1, sigma))
    return neural_activity


def import_mouse_neural_data(path, mouse, key_file, neural_type = 'spikes'):
    """
    Import all data for one mouse. Requires a yml file that contains session identifier keys.
    Args:
        path : str
            path to experiment directory
        mouse : str
            mouse name
        key_file : str
            name of yaml file that contains mouse as key and inner dictionary with context as key and date as value (e.g. A1 : 2022_06_08)
        neural_type : str
            what type of minian data to be loaded; by default spike data
    Returns:
        sessions : dict
            key is context, value is xarray.DataArray from minian output
    """
    ## Initialize sessions
    sessions = {}
    ## Load keys
    key_path = pjoin(path, key_file)
    with open(key_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    ## Select keys for a specific mouse
    keys = data_loaded[mouse]
    dpath = pjoin(path, 'Results/')
    dpath = pjoin(dpath, '{}/'.format(mouse))
    for date in os.listdir(dpath):
        sessions[list(keys.keys())[list(keys.values()).index([date])]] = ctn.load_and_align_minian(path, mouse, date, neural_type = neural_type)
    return sessions


def calculate_activity_correlation(first_session, second_session, test = 'pearson'):
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
    avg_first_session = first_session.values.mean(axis = 1)
    avg_second_session = second_session.values.mean(axis = 1)
    ## Perform correlation test
    if test == 'pearson':
        res = pearsonr(avg_first_session, avg_second_session)
    elif test == 'spearman':
        res = spearmanr(avg_first_session, avg_second_session)
    else:
        raise Exception('No test selected!')
    return res


def pairwise_session_analysis(
    path, mouse, mappings_path, neural_type='spikes', pairs = True, analysis = 'correlation', test = 'pearson'
):
    """
    Used to calculate cell activity correlations between pairs of sessions.
    Args:
        path : str
            experiment directory
        mouse: str
            name of the mouse (e.g. 'mc01')
        mappings_path : str
            path to cross registration results
        neural_type: str
            one of ['traces', 'spikes', 'smoothed']
        pairs: boolean
            if true, will only calculate the correlation between cells present in both sessions
        analysis: str
            one of ['correlation']
        test: str
            one of ['pearson', 'spearman']
    Returns:
        activity_summary : pandas.DataFrame
    """
    ## Get mappings
    mappings = pd.read_pickle(mappings_path)
    ## Create empty list
    activity_summary = []
    for d1,d2 in itertools.combinations_with_replacement(mappings.session.columns, r = 2):
        ## Load first session's data
        session1 = load_and_align_minian(path, mouse, date = d1, neural_type = neural_type)
        ## Load second session's data
        session2 = load_and_align_minian(path, mouse, date = d2, neural_type = neural_type)
        
        if pairs:
            ## Since d1 and d2 can be the same with replacement, the number of cell pairs in this combination is higher than
            ## the actual number of cells, so drop_duplicates() is used
            if d1 == d2:
                cell_ids = mappings.session[[d1, d2]].dropna(how = 'any').drop_duplicates().reset_index(drop = True)
            else:
                cell_ids = mappings.session[[d1, d2]].dropna(how = 'any').reset_index(drop = True)
            ## Select unit_ids based on cell_ids
            first_session = session1.sel(unit_id = np.array(cell_ids.iloc[:, 0]))
            second_session = session2.sel(unit_id = np.array(cell_ids.iloc[:, 1]))
            
            if analysis == 'correlation':
                res = calculate_activity_correlation(first_session, second_session, test)
                ## Create dictionary of session IDs and results
                tmp = [d1, d2, res[0], res[1]] 
                tmp2 = [d2, d1, res[0], res[1]]
                activity_summary.append(tmp)
                activity_summary.append(tmp2)
    ## Convert list to pd.DataFrame            
    activity_summary = pd.DataFrame(activity_summary, columns = ['session_id1', 'session_id2', 'statistic', 'pvalue'])
    return activity_summary