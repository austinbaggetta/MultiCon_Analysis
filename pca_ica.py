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
from scipy import stats
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, zscore
import plotly.express as px
import plotly.graph_objects as go

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
    np.random.seed()

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

        ica = FastICA(n_components=nassemblies)
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

####################### AUSTIN'S CODE STARTS HERE #######################


def calculate_ensemble_correlation(assemblies_sess1, assemblies_sess2, test = 'pearson'):
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


def assemblies_between_sessions(path, mouse, neural_type = 'spikes'):
    """
    Used to identify cell assemblies between all pairwise session comparisons.
    Args:
        path : str
            experiment directory
        mouse: str
            name of the mouse (e.g. 'mc01')
        neural_type: str
            one of ['traces', 'spikes', 'smoothed']
    Returns:
    
    """

        
       