"""
Codes for PCA/ICA methods described in Detecting cell assemblies in large
neuronal populations, Lopes-dos-Santos et al (2013).
https://doi.org/10.1016/j.jneumeth.2013.04.010
This implementation was written in Feb 2019. Please e-mail me if you have
comments, doubts, bug reports or criticism (Vítor, vtlsantos@gmail.com /
vitor.lopesdossantos@pharm.ox.ac.uk).
"""

import warnings
import pandas as pd
import numpy as np
import xarray as xr
from scipy import stats
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, zscore
import plotly.express as px
import plotly.graph_objects as go
import pymannkendall as mk
from statsmodels.stats.multitest import multipletests
import pingouin as pg
import numpy.matlib
import pickle
from os.path import join as pjoin
## Austin's custom py files
import circletrack_behavior as ctb
import circletrack_neural as ctn
import place_cells as pc

__author__ = "Vítor Lopes dos Santos"
__version__ = "2019.1"


def toyExample(assemblies, nneurons=10, nbins=1000, rate=1.0):
    np.random.seed()

    actmat = np.random.poisson(rate, nneurons * nbins).reshape(nneurons, nbins)
    assemblies.actbins = [None] * len(assemblies.membership)
    for (ai, members) in enumerate(assemblies.membership):
        members = np.array(members)
        nact = int(nbins * assemblies.actrate[ai])
        actstrength_ = rate * assemblies.actstrength[ai]

        actbins = np.argsort(np.random.rand(nbins))[0:nact]

        actmat[members.reshape(-1, 1), actbins] = (
            np.ones((len(members), nact)) + actstrength_
        )

        assemblies.actbins[ai] = np.sort(actbins)

    return actmat


class toyassemblies:
    def __init__(self, membership, actrate, actstrength):
        self.membership = membership
        self.actrate = actrate
        self.actstrength = actstrength


def marcenkopastur(significance):
    nbins = significance.nbins
    nneurons = significance.nneurons
    tracywidom = significance.tracywidom

    # calculates statistical threshold from Marcenko-Pastur distribution
    q = float(nbins) / float(nneurons)  # note that silent neurons are counted too
    lambdaMax = pow((1 + np.sqrt(1 / q)), 2)
    lambdaMax += tracywidom * pow(nneurons, -2.0 / 3)  # Tracy-Widom correction

    return lambdaMax


def getlambdacontrol(zactmat_):
    significance_ = PCA()
    significance_.fit(zactmat_.T)
    lambdamax_ = np.max(significance_.explained_variance_)

    return lambdamax_


def binshuffling(zactmat, significance):
    np.random.seed(42)

    lambdamax_ = np.zeros(significance.nshu)
    for shui in range(significance.nshu):
        zactmat_ = np.copy(zactmat)
        for (neuroni, activity) in enumerate(zactmat_):
            randomorder = np.argsort(np.random.rand(significance.nbins))
            zactmat_[neuroni, :] = activity[randomorder]
        lambdamax_[shui] = getlambdacontrol(zactmat_)

    lambdaMax = np.percentile(lambdamax_, significance.percentile)

    return lambdaMax


def circshuffling(zactmat, significance):
    np.random.seed(42)

    lambdamax_ = np.zeros(significance.nshu)
    for shui in tqdm(range(significance.nshu)):
        zactmat_ = np.copy(zactmat)
        for (neuroni, activity) in enumerate(zactmat_):
            cut = int(np.random.randint(significance.nbins * 2))
            zactmat_[neuroni, :] = np.roll(activity, cut)
        lambdamax_[shui] = getlambdacontrol(zactmat_)

    lambdaMax = np.percentile(lambdamax_, significance.percentile)

    return lambdaMax


def runSignificance(zactmat, significance):
    if significance.nullhyp == "mp":
        lambdaMax = marcenkopastur(significance)
    elif significance.nullhyp == "bin":
        lambdaMax = binshuffling(zactmat, significance)
    elif significance.nullhyp == "circ":
        lambdaMax = circshuffling(zactmat, significance)
    else:
        print("ERROR !")
        print(
            "    nyll hypothesis method "
            + str(significance.nullhyp)
            + " not understood"
        )
        significance.nassemblies = np.nan

    nassemblies = np.sum(significance.explained_variance_ > lambdaMax)
    significance.nassemblies = nassemblies

    return significance


def extractPatterns(actmat, significance, method):
    nassemblies = significance.nassemblies

    if method == "pca":
        idxs = np.argsort(-significance.explained_variance_)[0:nassemblies]
        patterns = significance.components_[idxs, :]
    elif method == "ica":
        from sklearn.decomposition import FastICA

        ica = FastICA(n_components=nassemblies, random_state = 42)
        ica.fit(actmat.T)
        patterns = ica.components_
    else:
        print("ERROR !")
        print("    assembly extraction method " + str(method) + " not understood")
        patterns = np.nan

    if patterns is not np.nan:
        patterns = patterns.reshape(nassemblies, -1)

        # sets norm of assembly vectors to 1
        norms = np.linalg.norm(patterns, axis=1)
        patterns /= np.matlib.repmat(norms, np.size(patterns, 1), 1).T

    return patterns


def runPatterns(
    zactmat, method="ica", nullhyp="circ", nshu=1000, percentile=99, tracywidom=False
):
    """
    INPUTS

        zactmat:     activity matrix - numpy array (neurons, time bins)
                        should already be z-scored

        nullhyp:    defines how to generate statistical threshold for assembly detection.
                        'bin' - bin shuffling, will shuffle time bins of each neuron independently
                        'circ' - circular shuffling, will shift time bins of each neuron independently
                                                            obs: mantains (virtually) autocorrelations
                        'mp' - Marcenko-Pastur distribution - analytical threshold

        nshu:       defines how many shuffling controls will be done (n/a if nullhyp is 'mp')

        percentile: defines which percentile to be used use when shuffling methods are employed.
                                                                    (n/a if nullhyp is 'mp')

        tracywidow: determines if Tracy-Widom is used. See Peyrache et al 2010.
                                                (n/a if nullhyp is NOT 'mp')

    OUTPUTS

        patterns:     co-activation patterns (assemblies) - numpy array (assemblies, neurons)
        significance: object containing general information about significance tests
        zactmat:      returns zactmat

    """

    nneurons = np.size(zactmat, 0)
    nbins = np.size(zactmat, 1)

    silentneurons = np.var(zactmat, axis=1) == 0
    if any(silentneurons):
        warnings.warn(
            f"Silent neurons detected: " f"{np.where(silentneurons)[0].tolist()}"
        )
    actmat_didspike = zactmat[~silentneurons, :]

    # # z-scoring activity matrix
    # actmat_ = stats.zscore(actmat_, axis=1)
    #
    # # Impute missing values.
    # imp = SimpleImputer(missing_values=np.nan, strategy='constant',
    #                     fill_value=0)
    # actmat_ = imp.fit_transform(actmat_.T).T

    # running significance (estimating number of assemblies)
    significance = PCA()
    significance.fit(actmat_didspike.T)
    significance.nneurons = nneurons
    significance.nbins = nbins
    significance.nshu = nshu
    significance.percentile = percentile
    significance.tracywidom = tracywidom
    significance.nullhyp = nullhyp
    significance = runSignificance(actmat_didspike, significance)
    if np.isnan(significance.nassemblies):
        return

    if significance.nassemblies < 1:
        print("WARNING !")
        print("    no assembly detected!")
        patterns = []
    else:
        # extracting co-activation patterns
        patterns_ = extractPatterns(actmat_didspike, significance, method)
        if patterns_ is np.nan:
            return

        # putting eventual silent neurons back (their assembly weights are defined as zero)
        patterns = np.zeros((np.size(patterns_, 0), nneurons))
        patterns[:, ~silentneurons] = patterns_
        # zactmat = np.copy(actmat)
        # zactmat[~silentneurons, :] = actmat_didspike

    return patterns, significance, zactmat


def computeAssemblyActivity(patterns, zactmat, zerodiag=True):
    nassemblies = len(patterns)
    nbins = np.size(zactmat, 1)
    assemblyAct = np.zeros((nassemblies, nbins))
    for (assemblyi, pattern) in enumerate(patterns):
        projMat = np.outer(pattern, pattern)
        if zerodiag:
            np.fill_diagonal(projMat, 0)
        assemblyAct[assemblyi, :] = np.diag(zactmat.T @ projMat @ zactmat)
    return assemblyAct


def computeAssemblyActivity_legacy(patterns, zactmat, zerodiag=True):
    nassemblies = len(patterns)
    nbins = np.size(zactmat, 1)

    assemblyAct = np.zeros((nassemblies, nbins))
    for (assemblyi, pattern) in enumerate(patterns):
        projMat = np.outer(pattern, pattern)
        projMat -= zerodiag * np.diag(np.diag(projMat))
        for bini in range(nbins):
            assemblyAct[assemblyi, bini] = np.dot(
                np.dot(zactmat[:, bini], projMat), zactmat[:, bini]
            )

    return assemblyAct


################### Will's code starts here ##################


def find_assemblies(
    neural_data,
    method="ica",
    nullhyp="mp",
    n_shuffles=1000,
    percentile=99,
    tracywidow=False,
    compute_activity=True,
    use_bool=False,
):
    """
    Gets patterns and assembly activations in one go.

    :parameters
    ---
    neural_data: (neuron, time) array
        Neural activity (e.g., S).

    method: str
        'ica' or 'pca'. 'ica' is recommended.

    nullhyp: str
        defines how to generate statistical threshold for assembly detection.
            'bin' - bin shuffling, will shuffle time bins of each neuron independently
            'circ' - circular shuffling, will shift time bins of each neuron independently
                     obs: maintains (virtually) autocorrelations
             'mp' - Marcenko-Pastur distribution - analytical threshold

    nshu: float
        defines how many shuffling controls will be done (n/a if nullhyp is 'mp')

    percentile: float
        defines which percentile to be used use when shuffling methods are employed.
        (n/a if nullhyp is 'mp')

    tracywidow: bool
        determines if Tracy-Widom is used. See Peyrache et al 2010.
        (n/a if nullhyp is NOT 'mp')

    """
    spiking, _, bool_arr = get_transient_timestamps(neural_data, thresh_type="eps")
    if use_bool:
        actmat = bool_arr
    else:
        actmat = stats.zscore(neural_data, axis=1) 

    # Replace NaNs.
    imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
    actmat = imp.fit_transform(actmat.T).T

    patterns, significance, z_data = runPatterns(
        actmat,
        method=method,
        nullhyp=nullhyp,
        nshu=n_shuffles,
        percentile=percentile,
        tracywidom=tracywidow,
    )

    if compute_activity:
        activations = computeAssemblyActivity(patterns, actmat)
    else:
        activations = None

    assembly_dict = {
        "patterns": patterns,
        "significance": significance,
        "z_data": z_data,
        "orig_data": neural_data,
        "activations": activations,
    }

    return assembly_dict


def lapsed_activation(act_list, nullhyp="circ", n_shuffles=1000, percentile=99):
    """
    Computes activity of ensembles based on data from another day.

    :parameters
    ---
    S_list: list of (neurons, time) arrays. The first entry will be
    considered the template AND all arrays must be sorted by row
    (neuron) in the same order.
        Neural activity from all sessions.

        method: str
        'ica' or 'pca'. 'ica' is recommended.

    nullhyp: str
        defines how to generate statistical threshold for assembly detection.
            'bin' - bin shuffling, will shuffle time bins of each neuron independently
            'circ' - circular shuffling, will shift time bins of each neuron independently
                     obs: maintains (virtually) autocorrelations
             'mp' - Marcenko-Pastur distribution - analytical threshold

    n_shuffles: float
        defines how many shuffling controls will be done (n/a if nullhyp is 'mp')

    percentile: float
        defines which percentile to be used use when shuffling methods are employed.
        (n/a if nullhyp is 'mp')
    """
    # Get patterns.
    patterns, significance, _ = runPatterns(
        act_list[0], nullhyp=nullhyp, nshu=n_shuffles, percentile=percentile
    )

    if significance.nassemblies < 1:
        raise ValueError("No assemblies detected.")

    # Find assembly activations for the template session then the lapsed ones.
    activations = []
    for actmat in act_list:
        # Get activations.
        activations.append(computeAssemblyActivity(patterns, actmat))

    assemblies = {
        "activations": activations,
        "patterns": patterns,
        "significance": significance,
    }

    return assemblies


def get_transient_timestamps(
    neural_data, thresh_type="eps", do_zscore=True, std_thresh=3
):
    """
    Converts an array of continuous time series (e.g., traces or S)
    into lists of timestamps where activity exceeds some threshold.
    :parameters
    ---
    neural_data: (neuron, time) array
        Neural time series, (e.g., C or S).
    std_thresh: float
        Number of standard deviations above the mean to define threshold.
    :returns
    ---
    event_times: list of length neuron
        Each entry in the list contains the timestamps of a neuron's
        activity.
    event_mags: list of length neuron
        Event magnitudes.
    """
    # Compute thresholds for each neuron.
    neural_data = np.asarray(neural_data, dtype=np.float32)
    if thresh_type == "eps":
        thresh = np.repeat(np.finfo(np.float32).eps, neural_data.shape[0])
    else:
        if do_zscore:
            stds = np.std(neural_data, axis=1)
            means = np.mean(neural_data, axis=1)
            thresh = means + std_thresh * stds
        else:
            thresh = np.repeat(std_thresh, neural_data.shape[0])

    # Get event times and magnitudes.
    bool_arr = neural_data > np.tile(thresh, [neural_data.shape[1], 1]).T

    event_times = [np.where(neuron > t)[0] for neuron, t in zip(neural_data, thresh)]

    event_mags = [neuron[neuron > t] for neuron, t in zip(neural_data, thresh)]

    return event_times, event_mags, bool_arr


def make_bins(data, samples_per_bin, axis=1):
    """
    Make bins determined by how many samples per bin.
    :parameters
    ---
    data: array-like
        Data you want to bin.
    samples_per_bin: int
        Number of values per bin.
    axis: int
        Axis you want to bin across.
    """
    try:
        length = data.shape[axis]
    except:
        length = data.shape[0]

    bins = np.arange(samples_per_bin, length, samples_per_bin)

    return bins.astype(int)


def nan_array(size):
    arr = np.empty(size)
    arr.fill(np.nan)

    return arr

####################### AUSTIN'S CODE STARTS HERE ############################
def bin_transients(data, bin_size_in_seconds, fps=15, analysis_type = 'max', non_binary=False):
    """
    Bin data then either sum, average, or take the max value in the bin.
    Args:
        data : xarray, or numpy array
            Minian output (S matrix, usually).
        bin_size_in_seconds : int
            How big you want each bin, in seconds.
        fps : int
            Sampling rate (default takes into account 2x downsampling
            from minian).
        analysis_type : str
            Determines what is done to the binned data - must be one of ['sum', 'num_spikes', 'average_value', 'max', 'median']
    Returns:
        summed: (cell, bin) array
            Number of S per cell for each bin.
    """
    ## Convert input into array
    if type(data) is not np.ndarray:
        data = np.asarray(data)
    #data = np.round(data, 3)

    # Group data into bins.
    bins = make_bins(data, bin_size_in_seconds * fps)
    binned = np.split(data, bins, axis=1)
    if analysis_type == 'sum':
    ## Sum the number of "S" per bin.
        if non_binary:
            summed = [np.sum(bin, axis=1) for bin in binned]
        else:
            summed = [np.sum(bin > 0, axis=1) for bin in binned]
        return np.vstack(summed).T
    ## Get the number of spikes within a bin
    elif analysis_type == 'num_spikes':
        num_spikes = [np.count_nonzero(bin, axis=1) for bin in binned]
        return np.vstack(num_spikes).T
    ## Get the mean value of a bin
    elif analysis_type == 'average_value':
        average_value = [np.mean(bin, axis = 1) for bin in binned]
        return np.vstack(average_value).T
    ## Get the max value within a bin
    elif analysis_type == 'max':
        max_value = [np.nanmax(bin, axis = 1) for bin in binned]
        return np.vstack(max_value).T
    ## Get median value within a bin
    elif analysis_type == 'median':
        median_value = [np.nanmedian(bin, axis = 1) for bin in binned]
        return np.vstack(median_value).T


def bin_avg_transient(data, bin_size_seconds, fps = 15, analysis_type = 'average_value'):
    """
    Used if there is one average trace from a population that you want to bin.
    """
    if type(data) is not np.ndarray:
        data = np.asarray(data)
    ## Create bins
    bins = make_bins(data, bin_size_seconds * fps)
    binned = np.split(data, bins)
    ## Take average within a bin
    if analysis_type == 'average_value':
        average_values = [np.mean(bin, axis = 0) for bin in binned]
        return average_values

def calculate_ensemble_correlation_shared(assemblies_sess1, assemblies_sess2, test='pearson'):
    """
    Used to calculate the pearson/spearman correlation or cosine similarity between two sessions.
    Requires cells to be shared between two sessions.
    Args:
        assemblies_sess1/assemblies_sess2 : dict
            output of find_assemblies function
        test : str
            must be one of ['pearson', 'spearman', 'cosine_similarity']
    Returns:
        corr : pandas.DataFrame
            pd.DataFrame with columns ensemble_id1, ensemble_id2, statistic, and pvalue
            pvalue is NaN if cosine similarity is calculated
    """
    corr = []
    for e in range(0, assemblies_sess1['patterns'].shape[0]):
        for i in range(0, assemblies_sess2['patterns'].shape[0]):
            if test == 'pearson':
                res = pearsonr(assemblies_sess1['patterns'][e], assemblies_sess2['patterns'][i])
                tmp = [e, i, res[0], res[1]]
            elif test == 'spearman':
                res = spearmanr(assemblies_sess1['patterns'][e], assemblies_sess2['patterns'][i])
                tmp = [e, i, res[0], res[1]]
            elif test == 'cosine_similarity':
                res = (1 - spatial.distance.cosine(assemblies_sess1['patterns'][e], assemblies_sess2['patterns'][i]))
                tmp = [e, i, res, np.nan]
            else:
                raise Exception("Not a valid test! Must be 'pearson', 'spearman', or 'cosine_similarity'")
            corr.append(tmp)
    corr = pd.DataFrame(corr, columns = ['ensemble_id1', 'ensemble_id2', 'statistic', 'pvalue'])
    return corr


def identify_correlated_ensembles(corr, alpha=0.05, direction='positive'):
    """
    Identifies positively or negatively correlated ensembles calculated from the calculate_ensemble_correlation function.
    Args:
        corr : pandas.DataFrame
            output from calculate_ensemble_correlation function
        alpha : float
            alpha level for significance; default is 0.05
        direction : str
            must be one of ['positive', 'negative'] to identify positively or negatively correlated ensembles
    Returns:
        pos_corr or neg_corr : pandas.DataFrame
            pd.DataFrame with columns ensemble_id1, ensemble_id2, statistic, and pvalue
    """
    if direction == 'positive':
        pos_corr = corr.loc[(corr.pvalue < alpha) & (corr.statistic > 0)].reset_index(drop = True)
        return pos_corr
    elif direction == 'negative':
        neg_corr = corr.loc[(corr.pvalue < alpha) & (corr.statistic < 0)].reset_index(drop = True)
        return neg_corr
    else:
        raise Exception("Direction must be 'positive' or 'negative'!")
        

def bootstrap_ensemble_correlations(patterns_one, patterns_two, test='pearson', iterations=500):
    """
    Used to perform bootstrap analysis of either Pearson correlation test statistics or cosine similarity values.
    Args:
        assemblies_sess1, assemblies_sess2 : dict
            output of find_assemblies function
        test : str
            must be one of ['pearson', 'cosine_similarity']
        iterations : int
            number of times to bootstrap; by default 1000
        plot_distribution : boolean
            creates a histogram of the test statistic; by default False
    Returns: 
        df : pandas.DataFrame
            pd.DataFrame with columns ensemble_id1, ensemble_id2, statistic, pvalue, and iteration
            pvalue is NaN if cosine similarity is calculated
    """
    dictionary = {'ensemble_id1': [], 'ensemble_id2': [], 'statistic': [], 'pvalue': [], 'iteration': []}
    for e in np.arange(0, patterns_one.shape[0]):
        for n_times in np.arange(1, iterations+1):
            for i in np.arange(0, patterns_two.shape[0]):
                samples = np.random.choice(patterns_two[i], patterns_two.shape[1], replace=False)
                ## Use pearson correlation to get a distribution of test statistics
                if test == 'pearson':
                    res = pearsonr(patterns_one[e], samples)
                elif test == 'cosine_similarity':
                    res = (1 - spatial.distance.cosine(patterns_one['patterns'][e], samples))
                else:
                    raise Exception("Incorrect 'test' argument! Must be one of ['pearson', 'cosine_similarity']")
                dictionary['ensemble_id1'].append(e)
                dictionary['ensemble_id2'].append(i)
                dictionary['statistic'].append(res[0])
                dictionary['pvalue'].append(res[1])
                dictionary['iteration'].append(n_times)
    ## Convert to dataframe           
    df = pd.DataFrame(dictionary)
    return df


def test_bootstrap_correlations(corr_ensembles, bootstrap_correlations, percentile=95, direction='positive'):
    """
    Determine matched ensembles.
    Args:
        corr_ensembles: pandas.DataFrame
            pd.DataFrame with columns ensemble_id1, ensemble_id2, statistic, and pvalue from identify_correlated_ensembles
        bootstrap_correlations : pandas.DataFrame
            pd.DataFrame with columns ensemble_id1, ensemble_id2, statistic, and pvalue from bootstrap_ensemble_correlations
        percentile : int
            what percentile the test statistic has to be above to be kept; by default 95th percentile
        direction : str
            must be one of ['positive', 'negative']; must be the same as identify_correlated_ensembles
    Returns:
        matched : pandas.DataFrame
           pd.DataFrame with columns ensemble_id1, ensemble_id2, statistic, and pvalue
           These are the ensemble pairs that are above the given percentile
    """
    ## Find out what statistic value represents that percentile
    matched = pd.DataFrame()
    for ensemble_id in np.unique(bootstrap_correlations['ensemble_id2']):
        cutoff = np.percentile(bootstrap_correlations['statistic'][bootstrap_correlations['ensemble_id2'] == ensemble_id], percentile)
        ensemble_data = corr_ensembles[corr_ensembles['ensemble_id2'] == ensemble_id]
        if direction == 'positive':
            matched_loop = ensemble_data.loc[ensemble_data['statistic'] > cutoff]
        elif direction == 'negative':
            matched_loop = ensemble_data.loc[ensemble_data['statistic'] < (-cutoff)]
        else:
            raise Exception('Incorrect direction string! Must be one of positive, negative')
        matched = pd.concat([matched, matched_loop])
    return matched


def resolve_matched_ensembles(matched, column_ids=['ensemble_id1', 'ensemble_id2']):
    """ 
    Finds pairs of ensembles with the highest correlation between them. 
    Args:
        matched : pandas.DataFrame
            output from test_bootstrap_correlations function, rough set of matches for ensembles between two sessions
            one ensemble can be matched to many ensembles from this output
        column_ids : list
            list of string values for the column names to be looped through;
            by default ['ensemble_id1', 'ensemble_id2']
    """
    ensemble_pairs = pd.DataFrame()
    for ensemble_id in np.unique(matched[column_ids[0]]):
        ensemble_data = matched[matched[column_ids[0]] == ensemble_id]
        ensemble_pair = ensemble_data[ensemble_data['statistic'] == np.max(ensemble_data['statistic'])]
        ensemble_pairs = pd.concat([ensemble_pairs, ensemble_pair]).reset_index(drop=True)

    matched_ensembles = pd.DataFrame()
    for ensemble_id in np.unique(ensemble_pairs[column_ids[1]]):
        ensemble_data = ensemble_pairs[ensemble_pairs[column_ids[1]] == ensemble_id]
        final_matches = ensemble_data[ensemble_data['statistic'] == np.max(ensemble_data['statistic'])]
        matched_ensembles = pd.concat([matched_ensembles, final_matches]).reset_index(drop=True)
    return matched_ensembles


def align_activations_to_behavior(activations, aligned_behavior):
    """
    Ensure the length of activations is the same length as behavior dataframe.
    Args:
        activations : numpy.ndarray
            activation strength across frame
        aligned_behavior : pandas.DataFrame
            output from load_and_align_behavior function
    Returns:
        indexed_behavior : pandas.DataFrame
            behavior df that's the same length as activations
    """
    subtraction = len(aligned_behavior.index) - activations.shape[1]
    if subtraction > 0:
        indexed_behavior = aligned_behavior[:-subtraction]
    else:
        indexed_behavior = aligned_behavior
    return indexed_behavior


def ensemble_trends_linear_regression(act, x_bin_size = 1, analysis_type = 'average_value', alpha_old = 0.05, correction = 'sidak'):
    ## Bin activations
    binned_activations = []
    for ensemble in act:
        binned_activations.append(bin_transients(ensemble[np.newaxis], x_bin_size, fps = 15, analysis_type = analysis_type, non_binary = True)[0])
    binned_activations = np.vstack(binned_activations)
    ## Do multiple linear corrections
    linear_regression_dict = {}
    for ensemble_number, ensemble_data in enumerate(binned_activations):
        time_vector = np.arange(0, binned_activations.shape[1])
        linear_regression_dict[ensemble_number] = pg.linear_regression(time_vector, ensemble_data, as_dataframe = False)
    ## Correct for multiple testing
    numtests = len(linear_regression_dict)
    if correction == 'sidak':
        alpha_corrected = (1 - (1 - alpha_old)**(1/numtests))
    elif correction == 'bonferroni':
        alpha_corrected = (1 - (alpha_old / numtests))
    ## Loop trough all linear regression outputs, check if pvalue < alpha_corrected
    ensemble_trends = {}
    no_trend = {}
    decreasing = {}
    increasing = {}
    for i in np.arange(0, len(linear_regression_dict)):
        if (linear_regression_dict[i]['pval'][1] < alpha_corrected) & (linear_regression_dict[i]['coef'][1] < 0):
            decreasing[i] = linear_regression_dict[i]
        elif (linear_regression_dict[i]['pval'][1] < alpha_corrected) & (linear_regression_dict[i]['coef'][1] > 0):
            increasing[i] = linear_regression_dict[i]
        else:
            no_trend[i] = linear_regression_dict[i]
    ensemble_trends = {'no trend': no_trend, 'decreasing': decreasing, 'increasing': increasing}
    return ensemble_trends


def define_ensemble_trends_across_time(activations, z_threshold = None, x_bin_size = 1, analysis_type = 'max', 
                                       zscored = True, correction = 'sidak', alpha_old = 0.05):
    """"
    Find assembly "trends", whether they are increasing/decreasing in occurrence rate over the course of a session. 
    Use the Mann Kendall test to find trends.

    Args:
        activations : np.array
            Values from data['activations'], which is determined from the find_assemblies function.
        x_bin_size : int
            If x == 'time', bin size in seconds
            If x == 'trial', bin size in trials
        zscored : boolean
            Determines whether or not to zscore activations. By default True.
    """
    ## zscore activation strength data
    if zscored:
        data = zscore(activations, nan_policy = 'omit', axis = 1)
    else:
        data = activations

    ## Binarize activations
    activity = np.zeros_like(data)
    if z_threshold is not None:
        for i, unit in enumerate(data):
            activity[i, unit > z_threshold] = unit[unit > z_threshold]
    else:
        activity = data

    ## Bin by time
    binned_activations = []
    for unit in activity:
        binned_activations.append(bin_transients(unit[np.newaxis], x_bin_size, fps = 15, analysis_type = analysis_type, non_binary = True)[0])
    binned_activations = np.vstack(binned_activations)

    ## Group ensembles into increasing, decreasing, or no trend
    trends = {key: [] for key in ['no trend', 'decreasing', 'increasing']}
    slopes, tau = nan_array(activity.shape[0]), nan_array(activity.shape[0])
    ## If using sidak
    if type(correction) is str:
        pvals = []
        for i, unit in enumerate(binned_activations):
            mk_test = mk.original_test(unit, alpha = 0.05)
            pvals.append(mk_test.p)

            slopes[i] = mk_test.slope
            tau[i] = mk_test.Tau
        ## Correct pvalues for multiple comparisons
        corrected_pvals = multipletests(pvals, alpha = alpha_old, method = correction)[1]
        ## Loop through tuples of combined corrected pvalues and slopes
        for i, (corr_pval, slope) in enumerate(zip(corrected_pvals, slopes)):
            if corr_pval < alpha_old:
                direction = 'increasing' if slope > 0 else 'decreasing'
                trends[direction].append(i)
            else:
                trends['no trend'].append(i)
        return trends, binned_activations, slopes, tau


def caclulate_activity_for_trends(trial_activation, analysis_type):
    """
    Used in define_ensemble_trends_across_trials function to calculate either the max, mean, or median of the activations.
    Args:
        analysis_type : str
            one of ['max', 'mean', 'median']
    Returns:
        result : np.ndarray
    """
    if analysis_type == 'max':
        result = np.nanmax(trial_activation, axis=1)
    elif analysis_type == 'mean':
        result = np.nanmean(trial_activation, axis=1)
    elif analysis_type == 'median':
        result = np.nanmedian(trial_activation, axis=1)
    return result


def define_ensemble_trends_across_trials_legacy(activations, aligned_behavior, trials, trial_type = 'forward', 
                                         z_threshold = None, zscored = True, correction = 'sidak', alpha_old = 0.05, analysis_type = 'max'):
    """"
    Find assembly "trends", whether they are increasing/decreasing in occurrence rate over the course of a session. 
    Use the Mann Kendall test to find trends.

    Args:
        activations : np.array
            Values from data['activations'], which is determined from the find_assemblies function.
        trial_type : str
            one of ['forward', 'reverse', 'all']
        x_bin_size : int
            If x == 'time', bin size in seconds
            If x == 'trial', bin size in trials
        zscored : boolean
            Determines whether or not to zscore activations. By default True.
    """
    ## zscore data
    if zscored:
        data = zscore(activations, nan_policy = 'omit', axis = 1)
    else:
        data = activations
    ## Binarize activations
    activity = np.zeros_like(data)
    if z_threshold is not None:
        for i, unit in enumerate(data):
            activity[i, unit > z_threshold] = unit[unit > z_threshold]
    else:
        activity = data

    if trial_type == 'forward':
        ## Bin by trial, take max activation strength within a bin
        forward_trials, reverse_trials = ctb.forward_reverse_trials(aligned_behavior, trials)
        binned_activations = []
        for forward in forward_trials:
            trial_activation = activity[:, trials == forward]
            binned_activations.append(caclulate_activity_for_trends(trial_activation, analysis_type = analysis_type))
        binned_activations = np.transpose(binned_activations)
    elif trial_type == 'reverse':
        ## Bin by trial, take max activation strength within a bin
        forward_trials, reverse_trials = ctb.forward_reverse_trials(aligned_behavior, trials)
        binned_activations = []
        for reverse in reverse_trials:
            trial_activation = activity[:, trials == reverse]
            binned_activations.append(caclulate_activity_for_trends(trial_activation, analysis_type = analysis_type))
        binned_activations = np.transpose(binned_activations)
    elif trial_type == 'all':
        binned_activations = []
        for trial in np.unique(trials):
            trial_activation = activity[:, trials == trial]
            binned_activations.append(caclulate_activity_for_trends(trial_activation, analysis_type = analysis_type))
        binned_activations = np.transpose(binned_activations)

    ## Group ensembles into increasing, decreasing, or no trend
    trends = {key: [] for key in ['no trend', 'decreasing', 'increasing']}
    slopes, tau = nan_array(activity.shape[0]), nan_array(activity.shape[0])
    ## If using sidak
    if type(correction) is str:
        pvals = []
        for i, unit in enumerate(binned_activations):
            mk_test = mk.original_test(unit, alpha = 0.05)
            pvals.append(mk_test.p)

            slopes[i] = mk_test.slope
            tau[i] = mk_test.Tau
        ## Correct pvalues for multiple comparisons
        corrected_pvals = multipletests(pvals, alpha = alpha_old, method = correction)[1] 
        ## Loop through tuples of combined corrected pvalues and slopes
        for i, (corr_pval, slope) in enumerate(zip(corrected_pvals, slopes)):
            if corr_pval < alpha_old:
                direction = 'increasing' if slope > 0 else 'decreasing'
                trends[direction].append(i)
            else:
                trends['no trend'].append(i)
        return trends, binned_activations, slopes, tau
    

def define_ensemble_trends_across_trials(act, behav, trial_type='forward', analysis='max', zscored=True, correction='sidak', alpha_old=0.05, **kwargs):
    """ 
    Use Mann-Kendall test to determine if coordinated activity is increasing/decreasing across a session.
    Args:
        act : np.array 
            values from data['activations'], which is determined from find_assemblies function
        behav : pandas.DataFrame
            processed behavior output
        trial_type : str
            one of ['forward', 'reverse', 'all']; forward by default
        analysis : str
            one of ['max', 'mean', 'median']; max by default
        zscored : boolean
            will zscore data if true; by default true
        correction : str
            used for multiple testing correction, one of ['sidak']; by default sidak
        alpha_old : float
            set the alpha value prior to correction; by default 0.05
        **kwargs
            additional arguments for get_forward_reverse_trials function, which are positive_jump and wiggle
    Returns:
        trends, binned_act, slopes, tau
    """
    if zscored:
        activity = zscore(act, nan_policy='omit', axis=1)
    else:
        activity = act 
    
    if (trial_type == 'forward') | (trial_type == 'reverse'):
        forward_trials, reverse_trials = ctb.get_forward_reverse_trials(behav, **kwargs)
        if trial_type == 'forward':
            trials = forward_trials 
        elif trial_type == 'reverse':
            trials = reverse_trials 
    elif trial_type == 'all':
        trials = np.unique(behav['trials'])
    else:
        print('Error! Must be one of forward, reverse, or all.')
    
    binned_act = []
    for trial in trials:
        trial_act = activity[:, behav['trials'] == trial]
        binned_act.append(caclulate_activity_for_trends(trial_act, analysis_type=analysis))
    binned_act = np.transpose(binned_act)

    trends = {key: [] for key in ['no trend', 'decreasing', 'increasing']}
    slopes, tau = nan_array(activity.shape[0]), nan_array(activity.shape[0])
    ## If using sidak
    if type(correction) is str:
        pvals = []
        for i, unit in enumerate(binned_act):
            mk_test = mk.original_test(unit, alpha = 0.05)
            pvals.append(mk_test.p)

            slopes[i] = mk_test.slope
            tau[i] = mk_test.Tau
        ## Correct pvalues for multiple comparisons
        corrected_pvals = multipletests(pvals, alpha = alpha_old, method = correction)[1] 
        ## Loop through tuples of combined corrected pvalues and slopes
        for i, (corr_pval, slope) in enumerate(zip(corrected_pvals, slopes)):
            if corr_pval < alpha_old:
                direction = 'increasing' if slope > 0 else 'decreasing'
                trends[direction].append(i)
            else:
                trends['no trend'].append(i)
        return trends, binned_act, slopes, tau


def calculate_proportions_ensembles(trends, function = None):
    """
    Calculate the proportion of increasing and decreasing ensembles.
    Args:
        trends : dict
            dictionary where keys are session_ids (e.g. 'Training1') and values are dictionaries with values ['no trend', 'increasing', 'decreasing']
            trends is output of define_ensemble_trends function
        function : str
            either 'linear_regression' or None - depends on how fading ensembles were determined
    Returns:
        proportion_dict : dict
            dictionary where keys are session_ids and values are dictionaries containing proportions of increasing/decreasing ensembles
    """
    ## Create empty dictionary
    proportion_dict = {}
    if function == 'linear_regression':
        num_ensembles = len(trends['no trend']) + len(trends['decreasing']) + len(trends['increasing'])
        num_decreasing = len(trends['decreasing'])
        num_increasing = len(trends['increasing'])
        proportion_dict['prop_increasing'] = num_increasing / num_ensembles
        proportion_dict['prop_decreasing'] = num_decreasing / num_ensembles
    else:
        for key in trends:
            num_decreasing = len(trends[key]['decreasing'])
            num_increasing = len(trends[key]['increasing'])
            total = num_decreasing + num_increasing + len(trends[key]['no trend'])
            ## Append to dictionary
            proportion_dict[key] = {'prop_increasing': num_increasing / total, 'prop_decreasing': num_decreasing / total}
    return proportion_dict


def save_detected_ensembles(path, neural_dict, binarize=True, smooth_factor=None, nullhyp='circ', n_shuffles=500):
    """
    This function saves each output from the find_assemblies function above into a pickle file for future use.
    Args:
        path : str
            path to where you want the nc file to be stored
        mouse : str
            mouse name
        neural_dict : dictionary
            dictionary where keys are session identifiers (e.g. 'Training1', 'A1', etc) and values are xarray.DataArray
            output from import_mouse_neural_data from circletrack_neural.py
        nullhyp : str
            defines how to generate statistical threshold for assembly detection.
                'bin' - bin shuffling, will shuffle time bins of each neuron independently
                'circ' - circular shuffling, will shift time bins of each neuron independently
                        obs: maintains (virtually) autocorrelations
                'mp' - Marcenko-Pastur distribution - analytical threshold
        n_shuffles : float
            defines how many shuffling controls will be done (n/a if nullhyp is 'mp')
    """
    for key in tqdm(neural_dict):
        if binarize:
            neural_data = neural_dict[key].values
            neural_data = neural_data > 0
        else:
            neural_data = neural_dict[key].values
        ## Smooth data using a moving average
        if smooth_factor is not None:
            smoothed_data = ctn.moving_average(neural_data, ksize=smooth_factor)
        else:
            smoothed_data = neural_data
        assemblies = find_assemblies(smoothed_data, nullhyp=nullhyp, n_shuffles=n_shuffles)
        ## Save activations and patterns in the same nc file
        act = assemblies['activations']
        pat = assemblies['patterns']
        ## Get length of both arrays, take the difference, and subset the longer one to the difference value
        ## This accounts for any frame discrepancies
        len_ensembles = act.shape[1]
        len_frames = len(neural_dict[key].frame.values)
        difference = len_frames - len_ensembles
        if len_ensembles < len_frames:
            activations = xr.DataArray(act, dims = ['en_id', 'frame'], 
                                    coords = {'en_id': np.arange(act.shape[0]), 'frame': neural_dict[key].frame.values[:-difference]}, 
                                    name = 'activations')
            patterns = xr.DataArray(pat, dims = ['en_id', 'unit_id'],
                                    coords = {'en_id': np.arange(pat.shape[0]), 'unit_id': neural_dict[key].unit_id.values},
                                    name = 'patterns')
        elif len_ensembles > len_frames:
            activations = xr.DataArray(act[:, 0:-difference], dims = ['en_id', 'frame'], 
                                    coords = {'en_id': np.arange(act.shape[0]), 'frame': neural_dict[key].frame.values}, 
                                    name = 'activations')
            patterns = xr.DataArray(pat, dims = ['en_id', 'unit_id'],
                                    coords = {'en_id': np.arange(pat.shape[0]), 'unit_id': neural_dict[key].unit_id.values},
                                    name = 'patterns')
        else:
            activations = xr.DataArray(act, dims = ['en_id', 'frame'], 
                                    coords = {'en_id': np.arange(act.shape[0]), 'frame': neural_dict[key].frame.values}, 
                                    name = 'activations')
            patterns = xr.DataArray(pat, dims = ['en_id', 'unit_id'],
                                    coords = {'en_id': np.arange(pat.shape[0]), 'unit_id': neural_dict[key].unit_id.values},
                                    name = 'patterns')
        ## Save activations and patterns to netcdf files
        data_path = pjoin(path, key)
        ica_data = xr.merge([activations, patterns])
        ica_data.to_netcdf(data_path)



def load_session_assemblies_legacy(mouse, spath, format = 'pickle', session_id = None):
    """
    Load pickle files of detected assemblies.
    Args:
        mouse : str
            mouse name
        spath : str
            path to pkl file
        experimenter : str
            one of ['pkl', 'pickle']; accounts for different pickle file naming conventions
        session_id : str
            one of ['neutral_only', 'fear_only', 'neutral_fear', 'remaining_ensemble']
    Returns:
        assemblies : dict
            dictionary of session-specific assemblies
    """
    if format == 'pkl':
        ## Create file named based on specified session_id
        file_name = pjoin(spath, '{}_{}_ensembles.pkl'.format(mouse, session_id))
    elif format == 'pickle':
        ## Create file named based on specified session_id
        file_name = pjoin(spath, 'assemblies_{}_{}.pickle'.format(mouse, session_id))
    ## Open pkl file
    with open(file_name, 'rb') as handle:
        assemblies = pickle.load(handle)
    return assemblies


def load_session_assemblies(mouse, spath, session_id):
    """
    Load netCDF files of detected assemblies.
    Args:
        mouse : str
            mouse name
        spath : str
            path to pkl file
        session_id : str
            one of ['neutral_only', 'fear_only', 'neutral_fear', 'remaining_ensemble']
    Returns:
        assemblies : dict
            dictionary of session-specific activations and patterns from PCA/ICA output
    """
    ## Create file name
    file_name = pjoin(spath, '{}_{}.nc'.format(mouse, session_id))
    assemblies = xr.open_dataset(file_name)
    return assemblies

   
def ensemble_membership(assemblies):
    """
    Create a boolean array that tells you which cells are 2 SDs above the mean.
    Args:
        assemblies : dict
            output from find_assemblies function
    Returns:
        participants : list
            list of np.arrays of boolean values; true if a cell is 2 SD above the mean
    """
    participants = []
    for i in np.arange(0, assemblies['patterns'].shape[0]):
        ensemble_average = assemblies['patterns'][i].mean()
        ## two stand deviations above or below the mean
        ensemble_cutoff_pos = ensemble_average + (assemblies['patterns'][i].std() * 2)
        ensemble_cutoff_neg = ensemble_average - (assemblies['patterns'][i].std() * 2)
        participating_cells = (assemblies['patterns'][i] > ensemble_cutoff_pos) | (assemblies['patterns'][i] < ensemble_cutoff_neg)
        participants.append(participating_cells)
    return participants
        

def get_ensemble_members(patterns, member_type, std_size=2):
    """
    Gets which unit_ids are considered members of an ensemble, determined by std_size above the mean weight.
    Args:
        patterns : xarray.DataArray
        member_type : str
            one of ['pos', 'neg', 'both']
        std_size : float
            how many standard deviations above the mean, by default 2
    Returns:
        xarray.DataArray
    """
    ensemble_cutoff_pos = (patterns.mean() + (patterns.std() * std_size))
    ensemble_cutoff_neg = (patterns.mean() - (patterns.std() * std_size))
    if member_type == 'pos':
        participants = patterns > ensemble_cutoff_pos
    elif member_type == 'neg':
        participants = patterns < ensemble_cutoff_neg
    elif member_type == 'both':
        participants = (patterns > ensemble_cutoff_pos) | (patterns < ensemble_cutoff_neg)
    else:
        raise Exception('Incorrect member_type argument!')
    return patterns[participants]


def match_ensembles_between_sessions(mouse, crossregistration_path, assembly_path, session_one, session_two, 
                                    direction, test='pearson', iterations=100, alpha=0.05, percentile=99):
    """ 
    
    """
    
    mappings = pd.read_feather(pjoin(crossregistration_path, f'{mouse}.feat'))
    assembly_one = load_session_assemblies(mouse, assembly_path, session_one)
    assembly_two = load_session_assemblies(mouse, assembly_path, session_two)

    ## Get pairs of cells between the two sessions of interest
    pairs = mappings.loc[:, [f'{session_one}', f'{session_two}']].dropna()
    ## Select cells from the cells that are paired between the two sessions
    session_one_units = pairs.loc[:, f'{session_one}'].to_numpy()
    session_two_units = pairs.loc[:, f'{session_two}'].to_numpy()
    a_first = assembly_one.sel(unit_id=session_one_units)
    a_second = assembly_two.sel(unit_id=session_two_units)
    ## Calculate the correlation between weights for all ensmbles in both sessions
    corr = calculate_ensemble_correlation_shared(a_first, a_second, test=test)
    pos_corr = identify_correlated_ensembles(corr, alpha=alpha, direction=direction)
    ## Bootstrap
    df = bootstrap_ensemble_correlations(a_first['patterns'], a_second['patterns'], test=test, iterations=iterations)
    ## Match ensembles based on statistic value being above whatever percentile value of the bootstrap distribution
    matched = test_bootstrap_correlations(pos_corr, df, percentile=percentile, direction=direction)
    matched_ensembles = resolve_matched_ensembles(matched)
    return matched_ensembles


def determine_proportion_matched(mouse, mouse_ensembles, matched_ensembles, session_of_interest, ensemble_id_column='ensemble_id2'):
    results_dict = {'mouse': [], 'num_fading': [], 'num_fading_matched': [], 'num_nonfading': [], 'num_nonfading_matched': []}
    ## Compare fading ensembles from the second session with the matched ensembles
    fading_ensemble_list = mouse_ensembles[mouse][session_of_interest]['decreasing']
    nonfading_ensemble_list = mouse_ensembles[mouse][session_of_interest]['no trend']

    fading_ensembles = pd.DataFrame()
    for fading_ens in fading_ensemble_list:
        ensemble = matched_ensembles[matched_ensembles[ensemble_id_column] == fading_ens]
        fading_ensembles = pd.concat([fading_ensembles, ensemble])
    fading_ensembles['mouse'] = mouse

    nonfading_ensembles = pd.DataFrame()
    for nonfading in nonfading_ensemble_list:
        ensemble_non = matched_ensembles[matched_ensembles[ensemble_id_column] == nonfading]
        nonfading_ensembles = pd.concat([nonfading_ensembles, ensemble_non])
    nonfading_ensembles['mouse'] = mouse

    results_dict['mouse'].append(mouse)
    results_dict['num_fading'].append(len(fading_ensemble_list))
    results_dict['num_fading_matched'].append(len(fading_ensembles))
    results_dict['num_nonfading'].append(len(nonfading_ensemble_list))
    results_dict['num_nonfading_matched'].append(len(nonfading_ensembles))
    return pd.DataFrame(results_dict), fading_ensembles, nonfading_ensembles
       

def make_ensemble_raster(assemblies, bin_size=0.3, running_only=False, velocity_thresh=7, ensemble_ids=None):
    lin_position = assemblies['lin_position'].values
    filler = np.zeros_like(lin_position)
    if ensemble_ids is not None:
        activations = assemblies['activations'].values[ensemble_ids]
    else:
        activations = assemblies['activations'].values

    if running_only:
        _, running = pc.define_running_epochs(x=assemblies['x'].values, y=assemblies['y'].values,
                                              t=assemblies['behav_t'], velocity_thresh=velocity_thresh)
    else:
        running = np.ones_like(assemblies['frame'].values, dtype=bool)

    bin_edges = pc.spatial_bin(lin_position, filler, bin_size_cm=bin_size, nbins=None, one_dim=True)[1]
    rasters = nan_array((activations.shape[0], int(np.max(assemblies['trials']).values), len(bin_edges) - 1))

    for trial_number in np.arange(int(np.max(assemblies['trials']))):
        time_bins = assemblies["trials"] == trial_number
        positions_this_trial = assemblies['lin_position'][time_bins]
        filler = np.zeros_like(positions_this_trial)
        running_this_trial = running[time_bins]
        for n, neuron in enumerate(activations):
            activation = neuron[time_bins] * running_this_trial
            rasters[n, trial_number, :] = pc.spatial_bin(
                positions_this_trial,
                filler,
                bins=bin_edges,
                one_dim=True,
                weights=activation)[0]
    return rasters, bin_edges


def ensemble_types_across_sessions(mouse_list, dpath, behav_path, session_list, ms_to_s=True, x_bin_size=None, analysis_type='max', 
                                   alpha_old=0.05, trial_type='all', zscored=False, z_thresh=None):
    ## Create empty dictionaries
    mouse_trends = {}
    mouse_binned_activations = {}
    mouse_slopes = {}
    mouse_taus = {}
    mouse_ensembles = {}
    mouse_trial_times = {}
    ## Loop through each mouse
    for mouse in tqdm(mouse_list):
        ## Create empty dictionaries to store output
        determined_trends = {}
        binned_activations_dict = {}
        slopes_dict = {}
        tau_dict = {}
        trial_times = {}
        ## Load assemblies
        for session in session_list:
            ## Load specific session's assemblies
            assemblies = load_session_assemblies(mouse, dpath, session)
            ## Set activation values as act
            act = assemblies['activations'].values
            ## Load a specific session's behavior data
            if x_bin_size is None:
                behav_file = pjoin(behav_path, '{}_{}.feat'.format(mouse, session))
                aligned_behavior = pd.read_feather(behav_file)
                ## Get which timestamps are part of which trial for the aligned behavior data
                trials = aligned_behavior['trials']
                ## Get length of time for each trial
                time_diff = []
                for trial in np.unique(trials):
                    ## Subset aligned_behavior by a given trial
                    behavior = aligned_behavior.loc[trials == trial]
                    ## Get the first and last timestamp to determine the window
                    first_timestamp, last_timestamp = behavior['t'].to_numpy()[0], behavior['t'].to_numpy()[-1]
                    if ms_to_s:
                        ## Convert from ms to s
                        first_timestamp = first_timestamp / 1000
                        last_timestamp = last_timestamp / 1000
                    ## Append to time_diff list
                    time_diff.append(last_timestamp - first_timestamp)
                trial_times[session] = time_diff
            ## This is where the data gets binned either by even time intervals or by trials
            if x_bin_size is not None:
                trends, binned_activations, slopes, tau = define_ensemble_trends_across_time(act, z_threshold=z_thresh, x_bin_size=x_bin_size, analysis_type=analysis_type, 
                                                                                             zscored=zscored, alpha_old=alpha_old)  
            else:
                ## Define ensemble trends across trials to determine if activation strength is increasing/decreasing across the session
                trends, binned_activations, slopes, tau = define_ensemble_trends_across_trials(act, aligned_behavior, trial_type=trial_type, 
                                                                                               z_threshold=z_thresh, alpha_old=alpha_old, analysis_type=analysis_type)
            ## Save to dictionaries
            determined_trends[session] = trends
            binned_activations_dict[session] = binned_activations
            slopes_dict[session] = slopes
            tau_dict[session] = tau
        ## Determine the proportion of ensembles that are increasing, decreasing, or have no trend based on their activation strength across time
        proportion_dict = calculate_proportions_ensembles(determined_trends)
        ## Save to mouse dictionaries before looping to the next mouse
        mouse_trends[mouse] = proportion_dict
        mouse_binned_activations[mouse] = binned_activations_dict
        mouse_slopes[mouse] = slopes_dict
        mouse_taus[mouse] = tau_dict
        mouse_ensembles[mouse] = determined_trends
        mouse_trial_times[mouse] = trial_times
    return mouse_trends, mouse_binned_activations, mouse_slopes, mouse_taus, mouse_ensembles, mouse_trial_times