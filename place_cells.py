import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from os.path import join as pjoin
from sklearn.metrics import mutual_info_score
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import pearsonr, zscore, spearmanr
from scipy.ndimage import generic_filter

import circletrack_behavior as ctb
import plotting_functions as pf
import circletrack_neural as ctn


def define_running_epochs(x, y, t, velocity_thresh=10):
    """
    Used to determine where a mouse is running above a given threshold.
    Args:
        x, y, t : array
            x and y position, time of each frame
        velocity_thresh : int
            any velocity value above this integer will be determined as running
    Returns:
        running : np.array
            array of boolean values where True means mouse is running during that frame
    """
    delta = np.diff(np.asarray((x, y)).T, axis = 0)
    dists = np.hypot(delta[:, 0], delta[:, 1])
    dists = np.insert(dists, 0, np.mean(dists)) ## if you insert 0, will cause inf due to division
    diff_t = np.diff(t, prepend=0)
    zero_data = np.where(diff_t == 0)[0]
    diff_t[zero_data] = 0.0000000001 ## when miiniscope drops frames and the same behavior times get used for nearby frames. This
    ## ensures there are no np.float64('inf') due to division by zero
    velocity = dists / diff_t ## in seconds
    velocity[velocity > 100000] = 0 ## can reset to zero to exclude these since their velocity is crazy high
    running = velocity > velocity_thresh
    return velocity, running


def adjust_behavior_start(x, y, t):
    """ 
    Used to remove the seconds at the beginning of the behavior for miniscope 
    mice since the session is started then the mouse is dropped into the maze.
    Args:
        x, y, t : np.array
            x, y, and time (in seconds) of mouse position data
        sampling_rate : float
            sampling rate of your behavior
    """
    x_cm, y_cm = ctb.convert_to_cm(x=x, y=y)
    velocity, _ = define_running_epochs(x=x_cm, y=y_cm, t=t)
    index = np.where(velocity > 500)[0]
    return index


def spatial_bins(x, y=None, bin_size=2, nbins=None, bins=None, weights=None, show_plot=True, **kwargs):
    """ 
    Create spatial bins of either x,y position data, or linearized position data.
    If radians, bin_size=0.16 for 2cm bins.
    """
    x_extrema = [np.min(x), np.max(x)]

    if nbins is None:
        nbins = int(np.round(np.diff(x_extrema)[0] / bin_size))
    
    if bins is not None:
        nbins = bins

    if y is not None:
        H, x_edges, y_edges = np.histogram2d(x, y, nbins, weights=weights)

        if show_plot:
            fig = pf.custom_graph_template(**kwargs)
            fig.add_trace(go.Heatmap(x=x_edges, y=y_edges, z=H))
            fig.show()
        return H, x_edges, y_edges
    
    else:
        H, x_edges = np.histogram(x, nbins, weights=weights)

        if show_plot:
            fig = pf.custom_graph_template(**kwargs)
            fig.add_trace(go.Bar(x=x_edges, y=H))
            fig.show()
        return H, x_edges


def minimum_activity_level(data, minimum_event_amount=0.2, **kwargs):
    """
    Used to determine whether a cell should be included in spatial analyses.
    Split the activity into x-second long bins (for example, 60s bins would yield 15).
    If a cell fires in more than 20% of those bins (for example, more than 3 bins), output will be True
    """
    binned_data = ctn.bin_activity(data, **kwargs)
    ## Determine proportion of events
    prop_events = np.sum(binned_data > 0, axis=1) / binned_data.shape[1]
    return prop_events >= minimum_event_amount


def make_place_field(neural_data, two_dim=True, binarize=False, **kwargs):
    if binarize:
        neural_data = neural_data > 0

    if two_dim:
        pf, x_edges, y_edges = spatial_bins(weights=neural_data, **kwargs)
        return pf, x_edges, y_edges
    else:
        pf, x_edges = spatial_bins(weights=neural_data, **kwargs)
        return pf, x_edges


def spatial_activity(neural_data, position_data, bin_size, binarized=True, fps=None):
    """
    Returns the number of events of all cells across spatial bins that can then be normalized by occupancy. Position_data is linearized position.
    Args:
        neural_data : xarray.DataArray
        position_data : numpy.ndarray
        bin_size : size of spatial bins
        binarized : bool
            whether to binarize calcium events or not
        fps : int 
            frames per second - your sampling rate for miniscope recordings
            if not None, will normalize occupancy by fps to convert occupancy to seconds
    Returns:
        population_activity : numpy.ndarray
            spatial bin x cells; number of events in each bin
        occupancy : numpy.ndarray
            amount of time spent in each spatial bin. if fps is not None, in seconds. Else in frames.
        bins : numpy.ndarray
            bin edges used for calculating spatial bins
    """
    bins = ctb.calculate_bins(x=position_data, bin_size=bin_size)
    population_activity = np.zeros((len(bins)-1, neural_data.shape[0]))
    occupancy = np.zeros((len(bins)-1))
    for idx, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
        binned_data = neural_data[:, (position_data >= start) & (position_data < end)]
        occupancy[idx] = binned_data.shape[1]
        if binarized:
            population_activity[idx, :] = np.sum(binned_data > 0, axis=1)
        else:
            population_activity[idx, :] = np.sum(binned_data, axis=1)
    if fps is not None:
        occupancy = occupancy / fps
    return population_activity, occupancy, bins


def get_tuning_curves(population_activity, occupancy):
    """ 
    Converts your spatially binned neuronal activity into rate maps through dividing by occupancy. 
    Epsilon is some very small error value to bring bins above zero for bayesian decoding.
    Args:
        population_activity : numpy.ndarray
            spatial bin x cells; number of events in each spatial bin
        occupancy : numpy.ndarray
            amount of time spent in each spatial bin
    Returns:
        tuning_curves : numpy.ndarray
            spatial bin x cells
    """
    epsilon = 1e-12
    tuning_curves = np.zeros((population_activity.shape[0], population_activity.shape[1]))
    for n in np.arange(0, population_activity.shape[1]):
        tuning_curves[:, n] = (population_activity[:, n] / occupancy) + epsilon
    return tuning_curves


def skaggs_information_content(tuning_curves, occupancy):
    mean_rate = np.mean(tuning_curves, axis=0)
    prob = occupancy / occupancy.sum()
    ## Handle log2(0)
    index = tuning_curves > 1e-12 ## epsilon value used in get_tuning_curves function
    ## Spatial information formula
    bits_per_event = [np.sum(prob[index[:, n]] * (tuning_curves[:, n][index[:, n]] / mean_rate[n]) * np.log2(tuning_curves[:, n][index[:, n]] / mean_rate[n])) for n in np.arange(0, tuning_curves.shape[1])]
    return np.array(bits_per_event)


def spatial_coherence_kernel(size):
    """ 
    Create a kernel for calculating the mean across spatial bins, excluding the center bin.
    Args:
        size : int
            must be an odd integer
    Returns:
        kernel : numpy.array
            a numpy array with 0 at the center and 1s on either side
    """
    sides = np.repeat(a=1, repeats=((size-1) / 2))
    return np.concatenate((sides, np.array([0]), sides)) / size


def calculate_spatial_coherence(tuning_curves, ksize):
    """
    Calculate spatial coherence for each cell in a population of cells.
    Population activity is a P x N matrix, where P is the number of spatial bins and N is the number of neurons.
    Args:
        ksize : int
            must be odd - determines size of kernel for averaging. Kernel in the form of [1, 0, 1], for example.
    """
    kernel = spatial_coherence_kernel(size=ksize)
    avg_of_avg = np.zeros((tuning_curves.shape[0], tuning_curves.shape[1]))
    spatial_coherence_values = np.zeros((tuning_curves.shape[1]))
    for n in np.arange(0, tuning_curves.shape[1]):
        avg_of_avg[:, n] = generic_filter(tuning_curves[:, n], function=np.mean, footprint=kernel, output=float, mode='wrap')
        spatial_coherence_values[n] = pearsonr(tuning_curves[:, n], avg_of_avg[:, n])[0]
    return avg_of_avg, spatial_coherence_values


def first_second_half_stability(ar, bin_size, lin_pos_col='lin_position'):
    """
    Separate neural activity into the first half and second half of the sesion.
    Args:
        data : xarray.DataArray
            aligned calcium and behavior
        bin_size : float
            size of linear position bins, either radians or degrees
        lin_pos_col : str
            name of position column, one of a_pos or lin_position (radians)
    Returns:
        cell_stability : numpy.ndarray
            array of Pearson's r values correlating spatially binned activity between the two halves
    """
    trials = np.unique(ar['trials'])
    data = ar[:, ar['trials'] == trials[0]]
    for trial in trials[1:]:
        trial_data = ar[:, ar['trials'] == trial]
        data = xr.concat([data, trial_data], dim='frame')
    median_trial = np.median(data['trials'])
    first_half = data[:, data['trials'] <= median_trial]
    second_half = data[:, data['trials'] > median_trial]
    bins = ctb.calculate_bins(x=data[lin_pos_col].values, bin_size=bin_size)

    first_pop_act = np.zeros((data.shape[0], len(bins)))
    second_pop_act = np.zeros((data.shape[0], len(bins)))
    for idx, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
        binned_data = first_half.values[:, (first_half[lin_pos_col] >= start) & (first_half[lin_pos_col] < end)]
        avg_activity = np.sum(binned_data > 0, axis=1) / binned_data.shape[1]
        first_pop_act[:, idx] = avg_activity

        binned_data = second_half.values[:, (second_half[lin_pos_col] >= start) & (second_half[lin_pos_col] < end)]
        avg_activity = np.sum(binned_data > 0, axis=1) / binned_data.shape[1]
        second_pop_act[:, idx] = avg_activity

    cell_stability = np.zeros((data.shape[0]))
    for cell in np.arange(0, data.shape[0]):
        cell_stability[cell] = pearsonr(first_pop_act[cell], second_pop_act[cell])[0]
    cell_stability
    return cell_stability


def odd_even_stability(data, bin_size, lin_pos_col='lin_position'):
    """
    Splits the data into odd and even forward trials (correct direction trials).
    Args:
        data : xarray.DataArray
            aligned calcium and behavior
        bin_size : float
            size of linear position bins, either radians or degrees
        lin_pos_col : str
            name of position column, one of a_pos or lin_position (radians)
    Returns:
        odd_even_ar : numpy.ndarray
            array of Pearson's r values correlating spatially binned activity between the odd and even trials
    """
    data['trials'] = data['trials'] + 1 ## since trials start at trial 0
    bins = ctb.calculate_bins(x=data[lin_pos_col].values, bin_size=bin_size)
    trials = np.unique(data['trials'])
    odd_data = data[:, data['trials'] == trials[0]]
    even_data = data[:, data['trials'] == trials[1]]
    for trial in trials[2:]:
        trial_data = data[:, data['trials'] == trial]
        if trial % 2 != 0:
            odd_data = xr.concat([odd_data, trial_data], dim='frame')
        else:
            even_data = xr.concat([even_data, trial_data], dim='frame')

    odd_pop_act = np.zeros((data.shape[0], len(bins)))
    even_pop_act = np.zeros((data.shape[0], len(bins)))
    for idx, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
        binned_data = odd_data.values[:, (odd_data[lin_pos_col] >= start) & (odd_data[lin_pos_col] < end)]
        avg_activity = np.sum(binned_data > 0, axis=1) / binned_data.shape[1]
        odd_pop_act[:, idx] = avg_activity

        binned_data = even_data.values[:, (even_data[lin_pos_col] >= start) & (even_data[lin_pos_col] < end)]
        avg_activity = np.sum(binned_data > 0, axis=1) / binned_data.shape[1]
        even_pop_act[:, idx] = avg_activity
    odd_even_ar = np.zeros((data.shape[0]))
    for cell in np.arange(0, data.shape[0]):
        odd_even_ar[cell] = pearsonr(odd_pop_act[cell], even_pop_act[cell])[0]
    return odd_even_ar


def shuffle_spatial_metrics(ar, lin_pos_col, bin_size, binarized=True, ksize=8, nshuffles=500, seed=24601):
    np.random.seed(seed)
    ## Preassign arrays for spatial metric outputs
    shuffled_ar_si = np.zeros((nshuffles, ar.shape[0]))
    shuffled_ar_sc = np.zeros((nshuffles, ar.shape[0]))
    for shuffle in np.arange(0, nshuffles):
        ## Shift position data forward by a random number of frames
        shuffled_data = np.array(())
        for trial in np.unique(ar['trials']):
            position = ar[lin_pos_col][ar['trials'] == trial].values
            random_shift = np.random.randint(0, position.shape[0])
            rolled_position = np.roll(position, random_shift)
            shuffled_data = np.concatenate((shuffled_data, rolled_position))
        ## Calculate spatial coherence and spatial information with the reordered data
        population_activity, occupancy, _ = spatial_activity(ar, shuffled_data, bin_size=bin_size, binarized=binarized)
        shuffled_tuning_curves = get_tuning_curves(population_activity, occupancy)
        shuffled_ar_si[shuffle, :] = skaggs_information_content(shuffled_tuning_curves, occupancy)
        _, spatial_coherence_values = calculate_spatial_coherence(shuffled_tuning_curves, ksize=ksize)
        shuffled_ar_sc[shuffle, :] = spatial_coherence_values
    return shuffled_ar_si, shuffled_ar_sc


def shuffle_stability_metrics(ar, bin_size, lin_pos_col='lin_position', nshuffles=500, seed=24601):
    np.random.seed(seed)
    ## Preassign arrays for spatial stability outputs
    shuffled_ar_first_second = np.zeros((nshuffles, ar.shape[0]))
    shuffled_ar_odd_even = np.zeros((nshuffles, ar.shape[0]))
    for shuffle in np.arange(0, nshuffles):
        ## Shift position data forward by a random number of frames
        shuffled_data = np.array(())
        for trial in np.unique(ar['trials']):
            position = ar[lin_pos_col][ar['trials'] == trial].values
            random_shift = np.random.randint(0, position.shape[0])  
            rolled_position = np.roll(position, random_shift)
            shuffled_data = np.concatenate((shuffled_data, rolled_position))
        ## Calculate trial stability of cells with the reordered positions
        ar_new = ar.copy()
        ar_new = ar_new.assign_coords(reordered_pos=('frame', shuffled_data))
        shuffled_ar_first_second[shuffle, :] = first_second_half_stability(ar_new, bin_size=bin_size, lin_pos_col='reordered_pos')
        shuffled_ar_odd_even[shuffle, :] = odd_even_stability(ar_new, bin_size=bin_size, lin_pos_col='reordered_pos')
    return shuffled_ar_first_second, shuffled_ar_first_second


def shuffle_mutual_info(ar, bin_size, lin_pos_col='lin_position', nshuffles=500, seed=24601):
    np.random.seed(seed)
    ## Preassign arrays for spatial stability outputs
    shuffled_mutual = np.zeros((nshuffles, ar.shape[0]))
    for shuffle in np.arange(0, nshuffles):
        ## Shift position data forward by a random number of frames
        shuffled_data = np.array(())
        for trial in np.unique(ar['trials']):
            position = ar[lin_pos_col][ar['trials'] == trial].values
            random_shift = np.random.randint(0, position.shape[0])  
            rolled_position = np.roll(position, random_shift)
            shuffled_data = np.concatenate((shuffled_data, rolled_position))
        ## Calculate trial stability of cells with the reordered positions
        ar_new = ar.copy()
        ar_new = ar_new.assign_coords(reordered_pos=('frame', shuffled_data))
        population_activity, _, bins = spatial_activity(ar_new.values, ar_new['reordered_pos'].values, bin_size=bin_size, binarized=False)
        ## Calculate mutual information
        discrete_bins = np.arange(0, bins.shape[0]-1)
        mi = [mutual_info_score((population_activity[:, uid] * 100).astype(int), discrete_bins) for uid in np.arange(0, population_activity.shape[1])]
        shuffled_mutual[shuffle, :] = mi
    return shuffled_mutual


def pf_relative_reward(tuning_curves, reward_one_pos, reward_two_pos, bins=None, proportion=True):
    """ 
    Create a distribution of place fields relative to reward locations.
    Args:
        tuning_curves : numpy.ndarray
            output from get_tuning_curves function
            must be cells x time
        reward_one_pos, reward_two_pos : float
            position of rewards
        bins : numpy.ndarray
            spatial bins used to create tuning curves. by default None
        probability : boolean
            whether to convert the distribution of place fields relative to rewards
            to a probability instead of count. by default True
    Returns:
        h_shift_one, h_shift_two : numpy.ndarray
            distribution of place fields relative to reward locations for reward one and reward two
    """
    ## Find the peak of each place field
    pf_peaks = np.max(tuning_curves, axis=1)
    ## Find the spatial bins where each peak occurred
    field_dist = np.zeros(pf_peaks.shape[0])
    for idx, peak in enumerate(pf_peaks):
        field_dist[idx] = bins[np.where(tuning_curves[idx, :] == peak)[0][0]]

    if bins is not None:
        H, xbin = np.histogram(field_dist, bins=bins[:-1])
    else:
        H, xbin = np.histogram(field_dist)
    mid_bin = int(xbin.shape[0] / 2)

    ## Get spatial bins relative to reward
    rel_rw_one = np.round(xbin - reward_one_pos, 3)
    rel_rw_two = np.round(xbin - reward_two_pos, 3)
    ## Get shifts to center data relative to reward locations
    shift_one = np.argmin(abs(rel_rw_one)) - mid_bin
    shift_two = np.argmin(abs(rel_rw_two)) - mid_bin
    if shift_one > 0:
        rel_rw_one = np.roll(rel_rw_one, shift=-shift_one)
        h_shift_one = np.roll(H, shift=-shift_one)
    else:
        rel_rw_one = np.roll(rel_rw_one, shift=abs(shift_one))
        h_shift_one = np.roll(H, shift=abs(shift_one))

    if shift_two > 0:
        rel_rw_two = np.roll(rel_rw_two, shift=-shift_two)
        h_shift_two = np.roll(H, shift=-shift_two)
    else:
        rel_rw_two = np.roll(rel_rw_two, shift=abs(shift_two))
        h_shift_two = np.roll(H, shift=abs(shift_two))
    
    if proportion:
        ## Convert to proportion to combine across mice in the future
        h_shift_one = h_shift_one / np.sum(h_shift_one) 
        h_shift_two = h_shift_two / np.sum(h_shift_two)
    return h_shift_one, h_shift_two, H, xbin, mid_bin, rel_rw_one, rel_rw_two


def tc_relative_reward(tuning_curves, reward_one_pos, reward_two_pos, bins, proportion=True, tc_metric='average'):
    """ 
    Create a single tuning curve across the session that then is used to look at spatial activity around reward locations.
    Args:
        tuning_curves : numpy.ndarray
            output from get_tuning_curves function
            must be cells x time
        reward_one_pos, reward_two_pos : float
            position of rewards
        bins : numpy.ndarray
            spatial bins used to create tuning curves. by default None
        proportion : boolean
            whether to convert the distribution of place fields relative to rewards
            to a proportion instead of count. by default True. Only done if tc_metric is not average activity.
        tc_metric : str
            one of average or above_zero
            if average, then will compute an average across all the tuning curves for each spatial bin.
            if above_zero, will sum all of the values where cells were active across spatial bins to create a histogram
    Returns:
        h_shift_one, h_shift_two : numpy.ndarray
            distribution of where cells are active relative to reward locations for reward one and reward two if tc_metric is above_zero
            average tuning curve relative to reward one and reward two if tc_metric is average

    """
    if tc_metric == 'average':
        tc = np.mean(tuning_curves, axis=0)
    elif tc_metric == 'above_zero':
        tc = np.sum(tuning_curves > 1e-10, axis=0) ## have to use a small value because of the epsilon in spatial_activity function for decoding
        tc = tc / np.sum(tc) ## normalize into proportion
    else:
        raise Exception(f'{tc_metric} not supported!')

    mid_bin = int(bins.shape[0] / 2)
    x_bins = bins.copy()

    ## Get spatial bins relative to reward
    rel_rw_one = np.round(bins - reward_one_pos, 3)
    rel_rw_two = np.round(bins - reward_two_pos, 3)
    ## Get shifts to center data relative to reward locations
    shift_one = np.argmin(abs(rel_rw_one)) - mid_bin
    shift_two = np.argmin(abs(rel_rw_two)) - mid_bin
    if shift_one > 0:
        rel_rw_one = np.roll(rel_rw_one, shift=-shift_one)
        h_shift_one = np.roll(tc, shift=-shift_one)
    else:
        rel_rw_one = np.roll(rel_rw_one, shift=abs(shift_one))
        h_shift_one = np.roll(tc, shift=abs(shift_one))

    if shift_two > 0:
        rel_rw_two = np.roll(rel_rw_two, shift=-shift_two)
        h_shift_two = np.roll(tc, shift=-shift_two)
    else:
        rel_rw_two = np.roll(rel_rw_two, shift=abs(shift_two))
        h_shift_two = np.roll(tc, shift=abs(shift_two))

    if (proportion) & (tc_metric != 'average'):
        ## Convert to proportion to combine across mice in the future
        h_shift_one = h_shift_one / np.sum(h_shift_one) 
        h_shift_two = h_shift_two / np.sum(h_shift_two)
    return h_shift_one, h_shift_two, tc, x_bins, mid_bin, rel_rw_one, rel_rw_two


def bayesian_decoding(tuning_curves, Q, bins, time_step):
    """ 
    Decode the animals location given tuning curves and binned spiking in time using Bayes Theorem.
    Translated from David Redish's MATLAB code.
    Args:
        tuning_curves : numpy.darray
            must be cells x spatial bin
        Q : numpy.ndarray
            must be cells x time
            histogram of number of events for each cell across the session, binned by time_step
        bins : numpy.ndarray
            edges of spatial bins used to create tuning curves
        time_step : float
            size of time bins in seconds
    Returns:
        pxs : numpy.ndarray
            time x spatial bin
            posterior probability distribution (posterior of x [position] given s [spiking]) for each time bin
            To get the decoded value, find the spatial bin with the maximum posterior probability post-hoc.
    """
    num_bins = bins.shape[0] ## spatial bins
    num_tbins = Q.shape[1] ## time bins

    ## Assume uniform occupancy
    Px = 1/num_bins

    pxs = np.full((num_tbins, num_bins), np.nan)
    for bin in np.arange(num_bins): ## for each spatial bin
        tempProd = np.nansum(np.log(np.matlib.repmat(tuning_curves[:, bin:bin+1], 1, num_tbins)**Q), axis=0) # for all time bins, get collective vote of population
        tempSum = np.exp(-time_step * np.nansum(tuning_curves[:, bin]))
        pxs[:, bin] = np.exp(tempProd) * tempSum * Px

    ## Normalize each time bin as long as the value is greater than zero
    for tbin in np.arange(num_tbins):
        denom = np.nansum(pxs[tbin, :])
        if denom > 0:
            pxs[tbin] /= denom 
    return pxs


def trial_raster(sdata, bin_size, binarized=True, lin_pos_col='lin_position', **kwargs):
    """ 
    Get trial activity for each spatial bin for each neuron.
    Args:
        sdata : xarray.DataArray
            aligned calcium and behavior
        bin_size : float
            size of spatial bin
        binarized : bool
            whether or not to binarize the event matrix when calculating events within the spatial bin
        lin_pos_col : str
            whether to use degrees or radians
        **kwargs
            additional arguments for subset_correct_dir_and_running
    Returns:
        population_act : numpy.ndarray
            trial x spatial bin x neuron array
            values are number of events in that trial in that spatial bin
        bins : numpy.ndarray
            spatial bin edges used
    """
    neural_data, position_data = ctn.subset_correct_dir_and_running(sdata, **kwargs)
    bins = ctb.calculate_bins(x=position_data, bin_size=bin_size)
    trials = np.unique(neural_data['trials'].values)
    population_act = np.zeros((trials.shape[0], bins[:-1].shape[0], neural_data.shape[0]))
    for trial_idx, trial in enumerate(trials):
        tdata = neural_data[:, neural_data['trials'] == trial] ## subset data by trial
        for spatial_idx, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
            binned_data = tdata[:, (tdata[lin_pos_col] >= start) & (tdata[lin_pos_col] < end)]

            if binarized:
                population_act[trial_idx, spatial_idx, :] = np.sum(binned_data > 0, axis=1)
            else:
                population_act[trial_idx, spatial_idx, :] = np.sum(binned_data, axis=1)

    return population_act, bins