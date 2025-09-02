import os
import pickle
from os.path import isdir
from os.path import join as pjoin
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter, label
from skimage.filters import threshold_otsu
from scipy.stats import pearsonr, zscore, spearmanr
from scipy.ndimage import convolve
import itertools
## Austin's custom .py files
import circletrack_behavior as ctb
import pca_ica as ica
import place_cells as pc


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
 

def calculate_activity_correlation(first_session, second_session, session_avg=True, test='spearman'):
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
    if session_avg:
        ## Calculate mean activity
        d1 = first_session.values.mean(axis=1)
        d2 = second_session.values.mean(axis=1)
    else:
        d1 = first_session
        d2 = second_session
    ## Perform correlation test
    if test == 'pearson':
        res = pearsonr(d1, d2)
    elif test == 'spearman':
        res = spearmanr(d1, d2)
    else:
        raise Exception('No test selected!')
    return res


def moving_average(data, ksize=8):
    kernel = np.ones(ksize)/ksize
    result = np.empty([data.shape[0], data.shape[1]])
    for i in np.arange(0, data.shape[0]):
        result[i] = np.convolve(data[i], v=kernel, mode='same')
    return result


def moving_average_xarray(data, ksize=8):
    return xr.apply_ufunc(
    np.convolve,
    data,
    input_core_dims=[['frame']],
    output_core_dims=[['frame']],
    vectorize=True,
    keep_attrs=True,
    kwargs={'v': np.ones(ksize)/ksize, 'mode': 'same'}).rename(f'{data.name}_smoothed')


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


def extract_windowed_data_by_index(data, window_val, window_size, fps=30):
    w = window_size * fps ## window_size in seconds
    return data[int(window_val - w):int(window_val + w + 1)]


def bootstrap_z_values(val, shuffled_val):
    z_values = (val - np.mean(shuffled_val, axis=0)) / np.std(shuffled_val, axis=0, ddof=1)
    return z_values


def subset_correct_dir_and_running(sdata, correct_dir=True, only_running=True, lin_pos_col='a_pos', velocity_thresh=10, filter_width=2):
    """ 
    Function to subset data based on whether mouse is moving in the correct direction or not and whether
    the mouse is running or not.
    Args:
        sdata : xarray.DataArray
            aligned miniscope and behavior data
        correct_dir, only_running : bool
            boolean for whether mouse is moving in the correct direction and running
        velocity_thresh : int or float
            speed threshold to determine running in cm/s
    Returns:
        neural_data : numpy.array
            matrix of neural response values that are subset based on direction and running
        position_data : numpy.array
            linearized position values that are also subset based on direction and running
    """
    if correct_dir:
        forward, _ = ctb.get_forward_reverse_trials(sdata)
        sess = sdata[:, sdata['trials'] == forward[0]]
        for trial in forward[1:]:
            loop_sess = sdata[:, sdata['trials'] == trial]
            sess = xr.concat([sess, loop_sess], dim='frame')
    else:
        sess = sdata.copy()

    x_pos, y_pos, _ = ctb.smooth_over_trials(sess, lin_pos_col=lin_pos_col, filter_width=filter_width)

    x_cm, y_cm = ctb.convert_to_cm(x=x_pos, y=y_pos)
    if only_running:
        velocity, running = pc.define_running_epochs(x_cm, 
                                                     y_cm, 
                                                     sess['behav_t'].values, 
                                                     velocity_thresh=velocity_thresh)
        position_data = sess[lin_pos_col].values[running]
        neural_data = sess[:, running]
    else:
        position_data = sess[lin_pos_col].values 
        neural_data = sess
    return neural_data, position_data


def bin_activity(data, bin_size_seconds, fps=30, func=np.mean, binarized=None):
    if type(data) is not np.ndarray:
        data = np.asarray(data)
    samples = bin_size_seconds * fps
    bins = np.arange(samples, data.shape[1], samples).astype(int)
    binned = np.split(data, bins, axis=1)
    if binarized is not None:
        act = [func(bin > binarized, axis=1) for bin in binned]
    else:
        act = [func(bin, axis=1) for bin in binned]
    return np.vstack(act).T 


def define_population_bursts(ar, min_len=3, zthresh=2, first_zscore=True, second_zscore=True):
    """
    Extract synchronous population events (bursts) by first thresholding above some zthresh then combining all events
    of at least the min_len.
    Args:
        ar : xarray.DataArray or numpy.ndarray
            Preprocessed calcium imaging data
        min_len : int
            minimum number of frames above the threshold to be considered a burst
        zthresh : int
            any value above this z-value will be considered part of a burst
        first_zscore, second_zscore : bool
            first_zscore normalizes each cell
            second_zscore will then zscore the average population activity
    """
    if first_zscore:
        ## Z-score each cell
        zdata = zscore(ar, axis=1)
    else:
        zdata = ar
    ## Average population activity
    pop_act = np.nanmean(zdata, axis=0)
    if second_zscore:
        pop_act = zscore(pop_act)
    ## Get frames where the population activity is above some z-threshold
    above_thresh = pop_act > zthresh
    ## Label every frame not above zthresh with zero, label frames above zthresh with what burst they belong to
    burst_array, burst_count = label(above_thresh)
    burst_start = np.array([np.min(np.where(burst_array==b + 1)[0]) for b in np.arange(burst_count) if np.where(burst_array==b+1)[0].shape[0] >= min_len])
    burst_end = np.array([np.max(np.where(burst_array==b + 1)[0]) for b in np.arange(burst_count) if np.where(burst_array==b+1)[0].shape[0] >= min_len])
    return burst_start, burst_end


def bin_in_time(da, bin_size=5, session_time=900, time_col='behav_t'):
    """
    Bins preprocessed xarray.DataArray based on behavior time.
    Args:
        da : xarray.DataArray
            aligned calcium and behavior data
        bin_size : int
            bin size in seconds
        session_time : int
            session time in seconds
    Returns:
        ar : list
            list of binned xarray.DataArrays
    """
    ar = []
    time_bins = np.arange(0, session_time + bin_size, bin_size)
    for bin in time_bins[:-1]:
        ar.append(da[:, (da[time_col] >= bin) & (da[time_col] < (bin + bin_size))])
    return ar


def bin_angle_data(ar, bin_size=30, col='a_pos'):
    """
    Bin neural activity into average activity within an angle bin.
    Args:
        ar : xarray.DataArray
            preprocessed calcium imaging activity
        bin_size : int or float
            size of your angle bin - by default 30 degrees
        col : str
            name of angle column, one of ['a_pos', 'lin_position']
    Returns:
        angle_ar : numpy.ndarray
            matrix of average activity of each neuron in that angle bin
    """
    assert type(ar) == xr.DataArray
    if col == 'a_pos':
        angles = np.arange(0, 360, bin_size)
    else:
        print('Radians not yet supported!')
    angle_ar = np.zeros((ar.shape[0], angles.shape[0])) ## neuron by angle 
    for idx, angle in enumerate(angles):
        loop_data = ar[:, (ar[col] >= angle) & (ar[col] < angle + bin_size)]
        angle_ar[:, idx] = loop_data.mean(dim='frame')
    return angle_ar
