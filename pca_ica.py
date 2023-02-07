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
import numpy.matlib
import pickle
from os.path import join as pjoin
## Austin's custom py files
import circletrack_behavior as ctb
import circletrack_neural as ctn

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
    np.random.seed()

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
            Determines what is done to the binned data - must be one of ['sum', 'average_num_spikes', 'average_value', 'max']
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
    ## Take the mean of the number of spikes within a bin
    elif analysis_type == 'average_num_spikes':
        average_spikes = [np.mean(bin > 0, axis = 1) for bin in binned]
        return np.vstack(average_spikes).T
    ## Get the mean value of a bin
    elif analysis_type == 'average_value':
        average_value = [np.mean(bin, axis = 1) for bin in binned]
        return np.vstack(average_value).T
    ## Get the max value within a bin
    elif analysis_type == 'max':
        max_value = [np.nanmax(bin, axis = 1) for bin in binned]
        return np.vstack(max_value).T


def calculate_ensemble_correlation_shared(assemblies_sess1, assemblies_sess2, test = 'pearson'):
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


def identify_correlated_ensembles(corr, alpha = 0.05, direction = 'positive'):
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
        

def bootstrap_ensemble_correlations(assemblies_sess1, assemblies_sess2, test = 'pearson', 
                                    iterations = 1000, plot_distribution = False):
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
    for e in range(0, assemblies_sess1['patterns'].shape[0]):
        for n_times in range(1, iterations+1):
            for i in range(0, assemblies_sess2['patterns'].shape[0]):
                samples = np.random.choice(assemblies_sess2['patterns'][i], assemblies_sess2['patterns'].shape[1],
                                           replace = False)
                ## Use pearson correlation to get a distribution of test statistics
                if test == 'pearson':
                    res = pearsonr(assemblies_sess1['patterns'][e], samples)
                    dictionary['ensemble_id1'].append(e)
                    dictionary['ensemble_id2'].append(i)
                    dictionary['statistic'].append(res[0])
                    dictionary['pvalue'].append(res[1])
                    dictionary['iteration'].append(n_times)
                ## Use cosine similarity to get a distribution of similarity values
                elif test == 'cosine_similarity':
                    res = (1 - spatial.distance.cosine(assemblies_sess1['patterns'][e], samples))
                    dictionary['ensemble_id1'].append(e)
                    dictionary['ensemble_id2'].append(i)
                    dictionary['statistic'].append(res)
                    dictionary['pvalue'].append(np.nan)
                    dictionary['iteration'].append(n_times)
                else:
                    raise Exception("Incorrect 'test' argument! Must be one of ['pearson', 'cosine_similarity']")
    ## Convert to dataframe           
    df = pd.DataFrame(dictionary)
    ## Histogram
    if plot_distribution:
        fig = px.histogram(df, x = 'statistic', template = 'simple_white')
        fig.show()
    return df


def test_bootstrap_correlations(corr_ensembles, bootstrap_correlations, percentile = 95, direction = 'positive', test = 'correlations'):
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
        test : str
            must be one of ['correlations', 'cosine_similarity']
    Returns:
        matched : pandas.DataFrame
           pd.DataFrame with columns ensemble_id1, ensemble_id2, statistic, and pvalue
           These are the ensemble pairs that are above the given percentile
    """
    ## Find out what statistic value represents that percentile
    cutoff = np.percentile(bootstrap_correlations['statistic'], percentile)
    ## Get ensemble comparisons that fall above 95% on either side of the distribution
    if test == 'correlations':
        if direction == 'positive':
            matched = corr_ensembles.loc[corr_ensembles.statistic > cutoff]
        elif direction == 'negative':
            matched = corr_ensembles.loc[corr_ensembles.statistic < (-cutoff)]
        else:
            raise Exception("No direction given! Must be one of ['positive', 'negative']")
    elif test == 'cosine_similarity':
        if direction == 'positive':
            matched = corr_ensembles.loc[corr_ensembles.statistic > cutoff]
        elif direction == 'negative':
            matched = corr_ensembles.loc[corr_ensembles.statistic < (-cutoff)]
        else:
            raise Exception("No direction given! Must be one of ['positive', 'negative']")
    else:
        raise Exception("No test given! Must be one of ['correlations', 'cosine_similarity']")
    return matched


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


def define_ensemble_trends_across_time(activations, z_threshold = None, x_bin_size = 1, analysis_type = 'max', zscored = True, alpha = 'sidak'):
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

    ## Bin by time, sum activation every few seconds
    binned_activations = []
    for unit in activity:
        binned_activations.append(bin_transients(unit[np.newaxis], x_bin_size, fps = 15, analysis_type = analysis_type, non_binary = True)[0])
    binned_activations = np.vstack(binned_activations)

    ## Group ensembles into increasing, decreasing, or no trend
    trends = {key: [] for key in ['no trend', 'decreasing', 'increasing']}
    slopes, tau = nan_array(activity.shape[0]), nan_array(activity.shape[0])
    ## If using sidak
    if type(alpha) is str:
        pvals = []
        pval_thresh = 0.005
        for i, unit in enumerate(binned_activations):
            mk_test = mk.original_test(unit, alpha = 0.05)
            pvals.append(mk_test.p)

            slopes[i] = mk_test.slope
            tau[i] = mk_test.Tau
        ## Correct pvalues for multiple comparisons
        corrected_pvals = multipletests(pvals, alpha = pval_thresh, method = alpha)[1]
        ## Loop through tuples of combined corrected pvalues and slopes
        for i, (corr_pval, slope) in enumerate(zip(corrected_pvals, slopes)):
            if corr_pval < pval_thresh:
                direction = 'increasing' if slope > 0 else 'decreasing'
                trends[direction].append(i)
            else:
                trends['no trend'].append(i)
        return trends, binned_activations, slopes, tau


def define_ensemble_trends_across_trials(activations, aligned_behavior, trials, trial_type = 'forward', z_threshold = None, zscored = True, alpha = 'sidak'):
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
            binned_activations.append(np.nanmax(trial_activation, axis = 1))
        binned_activations = np.transpose(binned_activations)
    elif trial_type == 'reverse':
        ## Bin by trial, take max activation strength within a bin
        forward_trials, reverse_trials = ctb.forward_reverse_trials(aligned_behavior, trials)
        binned_activations = []
        for reverse in reverse_trials:
            trial_activation = activity[:, trials == reverse]
            binned_activations.append(np.nanmax(trial_activation, axis = 1))
        binned_activations = np.transpose(binned_activations)
    elif trial_type == 'all':
        binned_activations = []
        for trial in np.unique(trials):
            trial_activation = activity[:, trials == trial]
            binned_activations.append(np.nanmax(trial_activation, axis = 1))
        binned_activations = np.transpose(binned_activations)

    ## Group ensembles into increasing, decreasing, or no trend
    trends = {key: [] for key in ['no trend', 'decreasing', 'increasing']}
    slopes, tau = nan_array(activity.shape[0]), nan_array(activity.shape[0])
    ## If using sidak
    if type(alpha) is str:
        pvals = []
        pval_thresh = 0.005
        for i, unit in enumerate(binned_activations):
            mk_test = mk.original_test(unit, alpha = 0.05)
            pvals.append(mk_test.p)

            slopes[i] = mk_test.slope
            tau[i] = mk_test.Tau
        ## Correct pvalues for multiple comparisons
        corrected_pvals = multipletests(pvals, alpha = pval_thresh, method = alpha)[1]
        ## Loop through tuples of combined corrected pvalues and slopes
        for i, (corr_pval, slope) in enumerate(zip(corrected_pvals, slopes)):
            if corr_pval < pval_thresh:
                direction = 'increasing' if slope > 0 else 'decreasing'
                trends[direction].append(i)
            else:
                trends['no trend'].append(i)
        return trends, binned_activations, slopes, tau


def calculate_proportions_ensembles(trends):
    """
    Calculate the proportion of increasing and decreasing ensembles.
    Args:
        trends : dict
            dictionary where keys are session_ids (e.g. 'Training1') and values are dictionaries with values ['no trend', 'increasing', 'decreasing']
            trends is output of define_ensemble_trends function
    Returns:
        proportion_dict : dict
            dictionary where keys are session_ids and values are dictionaries containing proportions of increasing/decreasing ensembles
    """
    ## Create empty dictionary
    proportion_dict = {}
    for key in trends:
        num_decreasing = len(trends[key]['decreasing'])
        num_increasing = len(trends[key]['increasing'])
        total = num_decreasing + num_increasing + len(trends[key]['no trend'])
        ## Append to dictionary
        proportion_dict[key] = {'prop_increasing': num_increasing / total, 'prop_decreasing': num_decreasing / total}
    return proportion_dict


def save_detected_ensembles(path, mouse, neural_dict, binarize = True, smooth_factor = None, nullhyp = 'circ', n_shuffles = 500):
    """
    This function saves each output from the find_assemblies function above into a pickle file for future use.
    Args:
        path : str
            path to where you want the pickle files to be stored
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
            smoothed_data = ctn.moving_average(neural_data, ksize = smooth_factor)
        else:
            smoothed_data = neural_data
        assemblies = find_assemblies(smoothed_data, nullhyp = nullhyp, n_shuffles = n_shuffles)
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
        data_path = pjoin(path, '{}_{}.nc'.format(mouse, key))
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
    file_name = pjoin(spath, 'assemblies/{}_{}.nc'.format(mouse, session_id))
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
        
       