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


def align_miniscope_frames(neural_activity, minian_timestamps, time, date, plot_frame_usage=False, down_sample_factor = 2, ttl_darkness = False):
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
        ttl_darkness: boolean
            whether or not a ttl had dark frames in the beginning
    Returns:
        lined_up_timeframes : list
            vector of lined up frames to use to align recording to the time vector.
    """
    ## If Minian was processed with temporal downsampling, set downsample to True so that minian_timestamps lines up with neural data
    if down_sample_factor is not None:
        minian_timestamps = minian_timestamps[::down_sample_factor]
    ## If ttl triggered, first 5 frames are dark, so exclude them
    if ttl_darkness:
        minian_timestamps[4:]
    ## Determine which frame is closest to the ideal time vector
    ## If there are NaNs in minian_timestamps because the session needed to be spliced together
    if minian_timestamps['Time Stamp (ms)'].isnull().values.any():
        null_values = np.where(minian_timestamps['Time Stamp (ms)'].isnull())[0]
        idx_first = null_values[0]
        idx_last = null_values[-1]
        first_half = time[0:idx_first]
        last_half = time[idx_last:]
        time = np.concatenate((first_half,last_half))
        ## Get the number of frames that were lost
        difference = minian_timestamps.reset_index().loc[idx_last+1, 'Frame Number'] - minian_timestamps.reset_index().loc[idx_first-1, 'Frame Number']
        ## Get the frames that need the difference value added to them to account for the part in the middle that is NaN
        need_to_fix = neural_activity.frame.values[idx_first:]
        ## Add the difference to these frames
        neural_activity.frame.values[idx_first:] = need_to_fix + difference
    ## If the minian result frames are shorter than minian_timestamps, unclear why this happens in rare instances
    if len(neural_activity.frame.values) < len(minian_timestamps):
        last_row = len(minian_timestamps)
        minian_timestamps = minian_timestamps.drop(minian_timestamps.index[last_row-1])
    arg_mins = [np.abs(minian_timestamps["Time Stamp (ms)"] - (t * 1000)).argmin() for t in time]
    lined_up_timeframes = np.array(minian_timestamps['Frame Number'].values[arg_mins])
    lined_up_milliseconds = np.array(minian_timestamps['Time Stamp (ms)'].values[arg_mins])
    lined_up_seconds = lined_up_milliseconds / 1000
    ## Get the last frame of the video
    last_frame = lined_up_timeframes[-1]
    args = np.argwhere(lined_up_timeframes == last_frame)
    if len(args) > 3: ## if the last frame is repeated more than 3 times
        lined_up_timeframes = lined_up_timeframes[:args[0][0]+1]

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
            title_text="Number of times each miniscope frame was re-used for {}".format(date),
        )
        fig.show(config={"scrollZoom": True})

    return lined_up_timeframes, lined_up_seconds


def load_and_align_minian(path, mouse, date, session = '20min', neural_type = 'spikes', sigma = None, sampling_rate = 1/15, 
                          downsample_further = False, downsample_factor = 2, plot_frame_usage = True):
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
        frames_per_file : int
            how many frames are in each .avi file
    Returns:
        neural_activity : xarray.DataArray
            can be aligned spike or calcium trace data
    """
    ## Load the specified type of neural activity
    rpath = pjoin(path, 'Results/{}/{}/'.format(mouse, date))
    timestamp = os.listdir(rpath) ## get timestamp associated with that day
    rpath = pjoin(rpath, timestamp[0]) 
    rpath = pjoin(rpath, 'minian') ## navigate to .zarr files
    mouse_minian = open_minian(rpath, return_dict=False)
    if neural_type == 'traces':
        neural_activity = mouse_minian['C']
    elif (neural_type == 'spikes') or (neural_type == 'smoothed'):
        neural_activity = mouse_minian['S']
    else:
        raise Exception(
            "Not a valid 'neural_type'; must be one of ['traces', 'spikes', 'smoothed']."
        )
    ## Set miniscope path as tpath
    tpath = pjoin(path, 'Data/{}/{}/{}/miniscope'.format(mouse, date, timestamp[0]))
    minian_timestamps = pd.read_csv(tpath + "/timeStamps.csv")
    ## Create time vector based on frame count
    if session == '20min':
        frame_count = 20 * 60 / sampling_rate ## 20min * 60 sec/min / sampling rate
    elif session == '30min':
        frame_count = 30 * 60 / sampling_rate
    else:
        raise Exception(
            "Invalid 'session' argument. Must be one of: ['20min', '30min']"
        )
    time = np.arange(0, frame_count * sampling_rate, sampling_rate)
    ## If you downsampled during minian processing, can change downsample_factor
    lined_up_timeframes, lined_up_seconds = align_miniscope_frames(neural_activity, minian_timestamps, time, date = date, plot_frame_usage = plot_frame_usage)
    if downsample_further:
        lined_up_timeframes = lined_up_timeframes[::downsample_factor]
    ## Select frames based on line-up timeframes
    neural_activity = neural_activity.sel(frame=lined_up_timeframes)
    if (
        neural_type == "smoothed"
    ):  # this filtering must be done after the previous line because this converts neural_activity to numpy array
        neural_activity = gaussian_filter(neural_activity, sigma=(1, sigma))
    return neural_activity


def minian_to_netcdf(path, mouse, date, session_id, cohort_number, session_length, sampling_rate = 1/15, down_sample_factor = 2):
    """
    Used to align miniscope data to a perfect time vector and subsequently save the S, C, and S_binary matrices as a combined netcdf file.
    Args:
        session_id : str
            name of session (Training4, A2, etc)
        cohort_number : str
            used to create a coordinate in the xarray.DataArray identifying which cohort this is
        session_length : str
            one of ['20min', '30min']
    Returns:
        named netcdf file
    """
    ## Create save_path
    spath = pjoin(path, 'Results/{}/{}/'.format(mouse, date))
    timestamp = os.listdir(spath) ## get timestamp associated with that day
    spath = pjoin(spath, '{}/processed'.format(timestamp[0])) 
    if not os.path.exists(spath):
        os.makedirs(spath)
    save_path = pjoin(spath, '{}_{}.nc'.format(mouse, session_id))
    ## Load in S, C, and create S_binary
    S_data = load_and_align_minian(path = path, mouse = mouse, date = date, session = session_length, neural_type = 'spikes', sampling_rate = sampling_rate, 
                                        downsample_factor = down_sample_factor, plot_frame_usage = False)
    S_data = S_data.assign_coords(cohort = cohort_number)
    C_data = load_and_align_minian(path = path, mouse = mouse, date = date, session = session_length, neural_type = 'traces', sampling_rate = sampling_rate, 
                                        downsample_factor = down_sample_factor, plot_frame_usage = False)  
    C_data = C_data.assign_coords(cohort = cohort_number)
    S_bin = S_data > 0   
    S_bin.name = 'S_bin'    
    neural_data = xr.merge([C_data, S_data, S_bin])  
    neural_data.to_netcdf(save_path)  


def load_preprocessed_minian(path, mouse, key_file):
    """
    Used to load minian output saved as a netcdf files.
    Args:
        path : str
            path to data
        mouse : str
            mouse  name
        key_file : str
            name of YML file containing mouse, session, and date information (mc03: Training1: '2022_09_30')
    Returns:
        sessions : dict
            dictionary with keys for mouse: session: minian_data
    """
    sessions = {}
    ## Load keys
    key_path = pjoin(path, key_file)
    with open(key_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    ## Select keys for a specific mouse
    keys = data_loaded[mouse]
    swapped_keys = {y[0]: x for x, y in keys.items()}
    for date in swapped_keys:
        dpath = pjoin(path, 'Results/{}/{}'.format(mouse, date))
        timestamp = os.listdir(dpath) ## get timestamp associated with that day
        rpath = pjoin(dpath, timestamp[0]) 
        minian_path = pjoin(rpath, 'processed/{}_{}.nc'.format(mouse, swapped_keys[date]))
        minian_data = xr.open_dataset(minian_path)
        sessions[swapped_keys[date]] = minian_data
    return sessions

def import_mouse_neural_data(path, mouse, key_file, session = '20min', neural_type = 'spikes', plot_frame_usage = False):
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
        print(date)
        sessions[list(keys.keys())[list(keys.values()).index([date])]] = load_and_align_minian(path, mouse, date, session = session, neural_type = neural_type, plot_frame_usage = plot_frame_usage)
    return sessions


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


def moving_average(data, ksize = 5):
    kernel = np.ones(ksize)/ksize
    result = np.empty([data.shape[0], data.shape[1]])
    for i in np.arange(0, data.shape[0]):
        result[i] = convolve(input = data[i], output = 'float', weights = kernel, mode = 'nearest')
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
