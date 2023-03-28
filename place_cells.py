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
import plotly.graph_objects as go
import pylab as pl
import xarray as xr
from numpy.polynomial.polynomial import polyfit
from plotly.subplots import make_subplots
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from scipy.stats import pearsonr, zscore, spearmanr
import itertools

from random import randint
import multiprocessing as mp
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.signal import savgol_filter
from sklearn.impute import SimpleImputer

def define_running_epochs(aligned_behavior, velocity_thresh = 7):
    """
    Used to determine where a mouse is running above a given threshold.
    Args:
        aligned_behavior : pandas.DataFrame
        velocity_thresh : int
            any velocity value above this integer will be determined as running
    Returns:
        running : np.array
            array of boolean values where True means mouse is running during that frame
    """
    delta = np.diff(np.asarray((aligned_behavior.x, aligned_behavior.y)).T, axis = 0)
    dists = np.hypot(delta[:, 0], delta[:, 1])
    dists = np.insert(dists, 0, 0)
    velocity = dists / np.diff(aligned_behavior.t/1000, prepend = 0) ## in seconds
    running = velocity > velocity_thresh
    return running




































############ Modified from wmau PlaceFields.py code ############
def spatial_bin(
    x, y, bin_size_cm=20, show_plot=False, weights=None, ax=None, bins=None, one_dim=False, nbins=None,
):
    """
    Spatially bins the position data.
    Args:
        x,y: array-like
            Vector of x and y positions in cm.
        bin_size_cm: float
            Bin size in centimeters.
        show_plot: bool
            Flag for plotting.
        weights: array-like
            Vector the same size as x and y, describing weights for
            spatial binning. Used for making place fields (weights
            are booleans indicating timestamps of activity).
        ax: Axis object
            If you want to reference an exis that already exists. If None,
            makes a new Axis object.
    Returns:
        H: (nx,ny) array
            2d histogram of position.
        xedges, yedges: (n+1,) array
            Bin edges along each dimension.
    """
    ## Calculate the min and max of position.
    x_extrema = [min(x), max(x)]
    y_extrema = [min(y), max(y)]
    ## Calculate bins 
    if nbins is None:
        nbins = int(np.round(np.diff(x_extrema)[0] / bin_size_cm))
    ## Calculate bins if one dimensional
    if one_dim:
        if bins is None:
            bins = np.linspace(
                x_extrema[0], x_extrema[1],
                nbins
            )
        H, edges = np.histogram(x, bins, weights=weights)
        if show_plot:
            if ax is None:
                fig, ax = plt.subplots()
            ax.plot(H)
        return H, edges, bins
    else:
        if bins is None:
            ## Make bins.
            xbins = np.linspace(
                x_extrema[0], x_extrema[1], nbins
            )
            ybins = np.linspace(
                y_extrema[0], y_extrema[1], nbins
            )
            bins = [ybins, xbins]
        # #Do the binning.
        H, xedges, yedges = np.histogram2d(y, x, bins, weights=weights)
        if show_plot:
            if ax is None:
                fig, ax = plt.subplots()
            ax.imshow(H)
        return H, xedges, yedges, bins


def consecutive_dist(x, axis=0, zero_pad=False):
    """
    Calculates the the distance between consecutive points in a vector.
    Args:
        aligned_behavior : pandas.DataFrame
            Output from circletrack_behavior.load_and_align_behavior function
    Returns:
        dists: array-like, distances
    """
    ## Calculate differences
    delta = np.diff(x, axis=axis)
    dists = np.hypot(delta[:, 0], delta[:, 1])
    if zero_pad:
        dists = np.insert(dists, 0, 0)
    return dists


def cart2pol(x, y):
    """
    Cartesian to polar coordinates. For linearizing circular trajectory.
    Args:
        aligned_behavior : pandas.DataFrame
           Output from load_and_align_behavior function
    Returns:
        (phi, rho): tuple
            Angle (linearized distance) and radius (distance from center).
    """
    ## Calculate rho and phi
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (phi, rho)


def define_field_bins(placefield, field_threshold=0.5):
    field_bins = np.where(placefield >= max(placefield) * field_threshold)[0]

    return field_bins


def spatial_information(tuning_curve, occupancy):
    """
    Calculate spatial information in one neuron's activity.
    :parameters
    ---
    tuning_curve: array-like
        Activity (S or binary S) per spatial bin.
    occupancy: array-like
        Time spent in each spatial bin.
    :return
    ---
    spatial_bits_per_spike: float
        Spatial bits per spike.
    """
    # Make 1-D.
    tuning_curve = tuning_curve.flatten()
    occupancy = occupancy.flatten()

    # Only consider activity in visited spatial bins.
    tuning_curve = tuning_curve[occupancy > 0]
    occupancy = occupancy[occupancy > 0]

    # Find rate and mean rate.
    rate = tuning_curve / occupancy
    mrate = tuning_curve.sum() / occupancy.sum()

    # Get occupancy probability.
    prob = occupancy / occupancy.sum()

    # Handle log2(0).
    index = rate > 0

    # Spatial information formula.
    bits_per_spk = sum(
        prob[index] * (rate[index] / mrate) * np.log2(rate[index] / mrate)
    )

    return bits_per_spk


class PlaceFields:
    def __init__(
        self,
        t,
        x,
        y,
        neural_data,
        bin_size=20,
        circular=False,
        linearized=False,
        shuffle_test=True,
        fps=None,
        velocity_threshold=10,
        nbins=None,
    ):
        """
        Place field object.
        Args:
            t: array
                Time array in milliseconds.
            x, y: (t,) arrays
                Positions per sample. Should be in cm. If circular==True,
                x will be converted to radians, but you should also use
                circle_radius.
            neural_data: (n,t) array
                Neural activity (usually S).
            bin_size: int
                Bin size in cm.
            circular: bool
                Whether the x data is in radians (for circular tracks).
            shuffle_test: bool
                Flag to shuffle data in time to recompute spatial information.
            fps: int
                Sampling rate. If None, will try to compute based on supplied
                time vector.
            threshold: float
                Velocity to threshold whether animal is running or not (cm/s).
        """
        imp = SimpleImputer(missing_values=np.nan, strategy='constant',
                            fill_value=0)
        neural_data = imp.fit_transform(neural_data.T).T
        if bin_size is not None and nbins is not None:
            print('Warning! Both bin_size and nbins were assigned values. '
                  'nbins will take priority. Proceed with caution.')
        x_extrema = [min(x), max(x)]
        y_extrema = [min(y), max(y)]
        ## Calculate bins 
        if nbins is None:
            nbins = int(np.round(np.diff(x_extrema)[0] / bin_size))
        else:
            nbins = nbins
        ## Set data and meta data
        self.data = {
            "t": t,
            "x": x,
            "y": y,
            "neural": neural_data,
            "n_neurons": neural_data.shape[0],
        }
        self.meta = {
            "circular": circular,
            "linearized": True if circular else linearized,
            "bin_size": bin_size,
            "nbins": nbins,
            "velocity_threshold": velocity_threshold,
        }

        ## Get fps.
        if fps is None:
            self.meta["fps"] = self.get_fps()
        else:
            self.meta["fps"] = int(fps)

        ## Compute distance and velocity. Smooth the velocity.
        d = consecutive_dist(
            np.asarray((self.data["x"], self.data["y"])).T, zero_pad=True
        )
        self.data["velocity"] = d / np.diff(t/1000, prepend=0)
        self.data["running"] = self.data["velocity"] > self.meta["velocity_threshold"]

        ## If we're using circular position, convert data to radians.
        if self.meta["circular"]:
            x_extrema = [min(x), max(x)]
            y_extrema = [min(y), max(y)]
            width = np.diff(x_extrema)[0]
            height = np.diff(y_extrema)[0]

            self.meta["circle_radius"] = np.mean([width, height]) / 2
            center = [np.mean(x_extrema), np.mean(y_extrema)]

            # Convert to angles (linear position) and radii (distance from center).
            angles, radii = cart2pol(x - center[0], y - center[1])

            # Shift everything so that 12 o'clock (pi/2) is 0.
            angles += np.pi / 2
            self.data['x'] = np.mod(angles, 2 * np.pi)
            self.data['y'] = np.zeros_like(self.data['x'])
        
        ## Get occupancy bins
        (
            self.data["occupancy_map"],
            self.data["occupancy_bins"],
        ) = self.make_occupancy_map(show_plot=False)

        ## Calculate all place fields
        self.data['placefields'] = self.make_all_place_fields()

        ## Calculate normalized by occupancy place fields
        self.data['normalized_placefields'] = self.make_all_place_fields(normalize = True)

        ## Calculate spatial information
        self.data['spatial_information'] = []
        for pf in self.data['placefields']:
            data = spatial_information(pf, self.data['occupancy_map'])
            self.data['spatial_information'].append(data)

        ## Find place field centers
        self.data["placefield_centers"] = self.find_pf_centers()

        ## Significance test
        if shuffle_test:
            (
                self.data["spatial_info_pvals"],
                self.data["spatial_info_z"],
            ) = self.assess_spatial_sig_parallel()


    def get_fps(self):
        """
        Get sampling frequency by counting interframe interval.
        :return:
        """
        # Take difference.
        interframe_intervals = np.diff(self.data["t"])

        # Inter-frame interval in milliseconds.
        mean_interval = np.mean(interframe_intervals)
        fps = round(1 / (mean_interval))

        return int(fps)


    def make_occupancy_map(self, show_plot=True, ax=None):
        """
        Makes the occupancy heat cell_map of the animal.
        :parameters
        ---
        bin_size_cm: float, bin size in centimeters.
        show_plot: bool, flag for plotting.
        """
        temp = spatial_bin(
            self.data["x"][self.data["running"]],
            self.data["y"][self.data["running"]],
            bin_size_cm=self.meta["bin_size"],
            show_plot=show_plot,
            one_dim=self.meta["linearized"],
            nbins=self.meta["nbins"]
        )
        occupancy_map, occupancy_bins = temp[0], temp[-1]

        if show_plot:
            if ax is None:
                fig, ax = plt.subplots()

            ax.imshow(occupancy_map, origin="lower")

        return occupancy_map, occupancy_bins
    

    def make_place_field(
        self, neuron, show_plot=True, normalize_by_occ=False, ax=None, shuffle=False
    ):
        """
        Bins activity in space. Essentially a 2d histogram weighted by
        neural activity.
        :parameters
        ---
        neuron: int, neuron index in neural_data.
        bin_size_cm: float, bin size in centimeters.
        show_plot: bool, flag for plotting.
        :return
        ---
        pf: (x,y) array, 2d histogram of position weighted by activity.
        """
        if shuffle:
            random_shift = randint(300, self.data["neural"].shape[1])
            neural_data = np.roll(self.data["neural"][neuron], random_shift)
        else:
            neural_data = self.data["neural"][neuron]

        pf = spatial_bin(
            self.data["x"][self.data["running"]],
            self.data["y"][self.data["running"]],
            bin_size_cm=self.meta["bin_size"],
            nbins=self.meta["nbins"],
            show_plot=False,
            weights=neural_data[self.data["running"]],
            one_dim=self.meta["linearized"],
            bins=self.data["occupancy_bins"],
        )[0]

        # Normalize by occupancy.
        if normalize_by_occ:
            pf = pf / self.data["occupancy_map"]

        if show_plot:
            if ax is None:
                fig, ax = plt.subplots()

            if self.meta['circular']:
                ax.plot(pf)
            else:
                ax.imshow(pf, origin="lower")

        return pf


    def make_all_place_fields(self, normalize=False):
        """
        Compute the spatial rate maps of all neurons.
        :return:
        """
        pfs = []
        for neuron in range(self.data["neural"].shape[0]):
            pfs.append(self.make_place_field(neuron, show_plot=False, normalize_by_occ=normalize))

        return np.asarray(pfs)

    
    def find_pf_centers(self, normalize = False):
        if normalize:
            centers = [np.argmax(pf) for pf in self.data["normalized_placefields"]]
        else:
            centers = [np.argmax(pf) for pf in self.data["placefields"]]

        return np.asarray(centers)
    

    def make_snake_plot(self, order="sorted", neurons="all", normalize=True):
        if neurons == "all":
            neurons = np.asarray([int(n) for n in range(self.data["n_neurons"])])
        pfs = self.data["normalized_placefields"][neurons]

        if order == "sorted":
            order = np.argsort(self.data["placefield_centers"][neurons])

        if normalize:
            pfs /= np.nanmax(pfs, axis=1)[:, np.newaxis]

        fig, ax = plt.subplots()
        ax.imshow(pfs[order])
        ax.axis('tight')
        ax.set_xlabel('Position')
        ax.set_ylabel('Neuron #')

        return fig, ax


    def assess_spatial_sig(self, neuron, n_shuffles=500):
        shuffled_SIs = []
        for i in range(n_shuffles):
            shuffled_pf = self.make_place_field(neuron, show_plot=False, normalize_by_occ=False, shuffle=True)
            shuffled_SIs.append(
                spatial_information(shuffled_pf, self.data["occupancy_map"])
            )

        shuffled_SIs = np.asarray(shuffled_SIs)
        p_value = np.sum(self.data["spatial_information"][neuron] < shuffled_SIs) / n_shuffles

        SI_z = (self.data["spatial_information"][neuron] - np.mean(shuffled_SIs)) / np.std(
            shuffled_SIs
        )

        return p_value, SI_z


    def assess_spatial_sig_parallel(self):
        print("Doing shuffle tests. This may take a while.")
        neurons = tqdm([n for n in range(self.data["n_neurons"])])
        n_cores = mp.cpu_count()
        # with futures.ProcessPoolExecutor() as pool:
        #     results = pool.map(self.assess_spatial_sig, neurons)
        results = Parallel(n_jobs=n_cores)(
            delayed(self.assess_spatial_sig)(i) for i in neurons
        )

        pvals, SI_z = zip(*results)

        return np.asarray(pvals), np.asarray(SI_z)
    

    # def plot_dots(
    #     self, neuron, std_thresh=2, pos_color="k", transient_color="r", ax=None
    # ):
    #     """
    #     Plots a dot show_plot. Position samples with suprathreshold activity
    #     dots overlaid.
    #     :parameters
    #     ---
    #     neuron: int, neuron index in neural_data.
    #     std_thresh: float, number of standard deviations above the mean
    #         to show_plot "spike" dot.
    #     pos_color: color-like, color to make position samples.
    #     transient_color: color-like, color to make calcium transient-associated
    #         position samples.
    #     """
    #     # Define threshold.
    #     thresh = np.mean(self.data["neural"][neuron]) + std_thresh * np.std(
    #         self.data["neural"][neuron]
    #     )
    #     supra_thresh = self.data["neural"][neuron] > thresh

    #     # Plot.
    #     if ax is None:
    #         fig, ax = plt.subplots()

    #     ax.scatter(self.data["x"], self.data["y"], s=3, c=pos_color)
    #     ax.scatter(
    #         self.data["x"][supra_thresh],
    #         self.data["y"][supra_thresh],
    #         s=3,
    #         c=transient_color,
    #     )

    
