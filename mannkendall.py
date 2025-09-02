import numpy as np
from scipy.stats import norm, rankdata
from collections import namedtuple


# Supporting Functions
# Data Preprocessing
def __preprocessing(x):
    x = np.asarray(x)
    dim = x.ndim
    
    if dim == 1:
        c = 1
        
    elif dim == 2:
        (n, c) = x.shape
        
        if c == 1:
            dim = 1
            x = x.flatten()
         
    else:
        print('Please check your dataset.')
        
    return x, c

	
# Missing Values Analysis
def __missing_values_analysis(x, method = 'skip'):
    if method.lower() == 'skip':
        if x.ndim == 1:
            x = x[~np.isnan(x)]
            
        else:
            x = x[~np.isnan(x).any(axis=1)]
    
    n = len(x)
    
    return x, n

	
# ACF Calculation
def __acf(x, nlags):
    y = x - x.mean()
    n = len(x)
    d = n * np.ones(2 * n - 1)
    
    acov = (np.correlate(y, y, 'full') / d)[n - 1:]
    
    return acov[:nlags+1]/acov[0]


# vectorization approach to calculate mk score, S
def __mk_score(x, n):
    s = 0

    demo = np.ones(n) 
    for k in range(n-1):
        s = s + np.sum(demo[k+1:n][x[k+1:n] > x[k]]) - np.sum(demo[k+1:n][x[k+1:n] < x[k]])
        
    return s

	
# original Mann-Kendal's variance S calculation
def __variance_s(x, n):
    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)

    # calculate the var(s)
    if n == g:            # there is no tie
        var_s = (n*(n-1)*(2*n+5))/18
        
    else:                 # there are some ties in data
        tp = np.zeros(unique_x.shape)
        demo = np.ones(n)
        
        for i in range(g):
            tp[i] = np.sum(demo[x == unique_x[i]])
            
        var_s = (n*(n-1)*(2*n+5) - np.sum(tp*(tp-1)*(2*tp+5)))/18
        
    return var_s


# standardized test statistic Z
def __z_score(s, var_s):
    if s > 0:
        z = (s - 1)/np.sqrt(var_s)
    elif s == 0:
        z = 0
    elif s < 0:
        z = (s + 1)/np.sqrt(var_s)
    
    return z


# calculate the p_value
def __p_value(z, alpha):
    # two tail test
    p = 2*(1-norm.cdf(abs(z)))  
    h = abs(z) > norm.ppf(1-alpha/2)

    if (z < 0) and h:
        trend = 'decreasing'
    elif (z > 0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'
    
    return p, h, trend


def __R(x):
    n = len(x)
    R = []
    
    for j in range(n):
        i = np.arange(n)
        s = np.sum(np.sign(x[j] - x[i]))
        R.extend([(n + 1 + s)/2])
    
    return np.asarray(R)


def __K(x,z):
    n = len(x)
    K = 0
    
    for i in range(n-1):
        j = np.arange(i,n)
        K = K + np.sum(np.sign((x[j] - x[i]) * (z[j] - z[i])))
    
    return K

	
# Original Sens Estimator
def __sens_estimator(x):
    idx = 0
    n = len(x)
    d = np.ones(int(n*(n-1)/2))

    for i in range(n-1):
        j = np.arange(i+1,n)
        d[idx : idx + len(j)] = (x[j] - x[i]) / (j - i)
        idx = idx + len(j)
        
    return d


def sens_slope(x):
    """
    This method proposed by Theil (1950) and Sen (1968) to estimate the magnitude of the monotonic trend. Intercept calculated using Conover, W.J. (1980) method.
    Input:
        x:   a one dimensional vector (list, numpy array or pandas series) data
    Output:
        slope: Theil-Sen estimator/slope
        intercept: intercept of Kendall-Theil Robust Line
    Examples
    --------
      >>> import numpy as np
	  >>> import pymannkendall as mk
      >>> x = np.random.rand(120)
      >>> slope,intercept = mk.sens_slope(x)
    """
    res = namedtuple('Sens_Slope_Test', ['slope','intercept'])
    x, c = __preprocessing(x)
#     x, n = __missing_values_analysis(x, method = 'skip')
    n = len(x)
    slope = np.nanmedian(__sens_estimator(x))
    intercept = np.nanmedian(x) - np.median(np.arange(n)[~np.isnan(x.flatten())]) * slope  # or median(x) - (n-1)/2 *slope
    
    return res(slope, intercept)


def seasonal_sens_slope(x_old, period=12):
    """
    This method proposed by Hipel (1994) to estimate the magnitude of the monotonic trend, when data has seasonal effects. Intercept calculated using Conover, W.J. (1980) method.
    Input:
        x:   a vector (list, numpy array or pandas series) data
		period: seasonal cycle. For monthly data it is 12, weekly data it is 52 (12 is the default)
    Output:
        slope: Theil-Sen estimator/slope
        intercept: intercept of Kendall-Theil Robust Line, where full period cycle consider as unit time step
    Examples
    --------
      >>> import numpy as np
	  >>> import pymannkendall as mk
      >>> x = np.random.rand(120)
      >>> slope,intercept = mk.seasonal_sens_slope(x, 12)
    """
    res = namedtuple('Seasonal_Sens_Slope_Test', ['slope','intercept'])
    x, c = __preprocessing(x_old)
    n = len(x)
    
    if x.ndim == 1:
        if np.mod(n,period) != 0:
            x = np.pad(x,(0,period - np.mod(n,period)), 'constant', constant_values=(np.nan,))

        x = x.reshape(int(len(x)/period),period)
    
#     x, n = __missing_values_analysis(x, method = 'skip')
    d = []
    
    for i in range(period):
        d.extend(__sens_estimator(x[:,i]))
        
    slope = np.nanmedian(np.asarray(d))
    intercept = np.nanmedian(x_old) - np.median(np.arange(x_old.size)[~np.isnan(x_old.flatten())]) / period * slope
    
    return res(slope, intercept)

	
def original_test(x_old, alpha = 0.05):
    """
    This function checks the Mann-Kendall (MK) test (Mann 1945, Kendall 1975, Gilbert 1987).
    Input:
        x: a vector (list, numpy array or pandas series) data
        alpha: significance level (0.05 default)
    Output:
        trend: tells the trend (increasing, decreasing or no trend)
        h: True (if trend is present) or False (if trend is absence)
        p: p-value of the significance test
        z: normalized test statistics
        Tau: Kendall Tau
        s: Mann-Kendal's score
        var_s: Variance S
        slope: Theil-Sen estimator/slope
        intercept: intercept of Kendall-Theil Robust Line
    Examples
    --------
	  >>> import numpy as np
      >>> import pymannkendall as mk
      >>> x = np.random.rand(1000)
      >>> trend,h,p,z,tau,s,var_s,slope,intercept = mk.original_test(x,0.05)
    """
    res = namedtuple('Mann_Kendall_Test', ['trend', 'h', 'p', 'z', 'Tau', 's', 'var_s', 'slope', 'intercept'])
    x, c = __preprocessing(x_old)
    x, n = __missing_values_analysis(x, method = 'skip')
    
    s = __mk_score(x, n)
    var_s = __variance_s(x, n)
    Tau = s/(.5*n*(n-1))
    
    z = __z_score(s, var_s)
    p, h, trend = __p_value(z, alpha)
    slope, intercept = sens_slope(x_old)

    return res(trend, h, p, z, Tau, s, var_s, slope, intercept)