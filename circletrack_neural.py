import os
import pickle
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
from scipy.stats import pearsonr, zscore


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


def align_miniscope_frames(minian_timestamps, time, plot_frame_usage=False):
    """
    Takes timestamps matrix associated with a miniscope recording and a regularly spaced time vector the
    expected length of the session. For each timeframe in 'time', the closest frame from minian_timestamps is acquired.
    Some frames are used more than once. Returns vector of lined up frames to use to align recording to the time vector.
    """
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


def load_and_align_minian(path, mouse, date, timestamp, session = '20min', neural_type="spikes", sigma=None, sampling_rate=1/15, downsample = True, downsample_factor=2):
    """
    Parameters:
    ==========
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
    downsample : boolean
        if data was downsampled during minian
    downsample_factor : int
        factor that minian downsampled data
    """

    # create time vector based on expected length of session
    if "20min" in session:
        frame_count = (
            20 * 60 / sampling_rate
        )  # 20min x 60sec/min / sampling_rate (usually 1/15 because of temporal downsampling during Minian processing)
    else:
        raise Exception(
            "Invalid 'session' argument. Must be one of: ['20min']"
        )
    time = np.arange(0, frame_count * sampling_rate, sampling_rate)

    # load the specified type of neural activity
    rpath = pjoin(path, 'Results/{}/{}/{}/minian'.format(mouse, date, timestamp))
    mouse_minian = open_minian(rpath, return_dict=True)
    if neural_type == "traces":
        neural_activity = mouse_minian["C"]
    elif (neural_type == "spikes") or (neural_type == "smoothed"):
        neural_activity = mouse_minian["S"]
    else:
        raise Exception(
            "Not a valid 'neural_type'; must be one of ['traces', 'spikes', 'smoothed']."
        )
    tpath = pjoin(path, 'Data/{}/{}/{}/miniscope'.format(mouse, date, timestamp))
    minian_timestamps = pd.read_csv(tpath + "/timeStamps.csv")
    if downsample:
        minian_timestamps = minian_timestamps[::downsample_factor]
    lined_up_timeframes = align_miniscope_frames(minian_timestamps, time)
    neural_activity = neural_activity.sel(frame=lined_up_timeframes)
    if (
        neural_type == "smoothed"
    ):  # this filtering must be done after the previous line because this converts neural_activity to numpy array
        neural_activity = gaussian_filter(neural_activity, sigma=(1, sigma))

    return neural_activity

