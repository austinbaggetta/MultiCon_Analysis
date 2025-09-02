import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from scipy.ndimage import convolve
from scipy.stats import zscore

class DiffusionMetrics:
    def __init__(self, data=None, fps=None):
        """
        Create DiffusionMetrics class.
        Args:
            data : np.ndarray
            fps : float
                frames per second, given as a fraction (e.g. 1/30)
        """
        self.data = None
        self.msd_data = None

        if data is not None:
            self.raw_data = np.asarray(data)
        else:
            self.raw_data = None

        if fps is not None:
            self.time_int = fps 
        else:
            self.time_int = None
    

    def smooth_data(self, ksize, type='moving_average'):
        """
        Smooth data using different smoothing algorithms.
        Args:
            ksize : int
                Determiens kernel size for smoothing
            type : str
                which type of smoothing algorithm you want to use
        Returns:
            self.data : smoothed data
        """
        if self.data is not None:
            d = self.data 
        else:
            d = self.raw_data

        if type == 'moving_average':
            kernel = np.ones(ksize)/ksize
            result = np.empty([d.shape[0], d.shape[1]])
            for i in np.arange(0, d.shape[0]):
                result[i] = convolve(input=d[i], output='float', weights=kernel, mode='nearest')
            self.data = result
        else:
            raise Exception('Incorrect smooth type chosen!')
        
    
    def zscore_data(self, axis=1):
        if self.data is not None:
            d = self.data 
        else:
            d = self.raw_data

        output = zscore(d, axis=axis)
        self.data = output[~np.isnan(output).any(axis=axis), :] ## remove NaN values along axis
    

    def calculate_msd(self, msd_input=None, time_diff=25):
        """
        Calculate the mean squared displacement (MSD) of your data.
        Args:
            time_diff : int
                Creates a list of time lags, used as indices, to calculate the MSD against. 
                Actual time values are dependent on the sampling of your data.
        Returns:
            self.msd_data : dict
                dictionary containing lag values and msd values
        """
        if self.data is not None:
            d = self.data 
        else:
            d = self.raw_data

        if msd_input is not None:
            d = msd_input 
            
        time_lags = np.arange(0, time_diff)
        msd = {'lag_values': [], 'msd_values': []}
        for lag in tqdm(time_lags):
            lag_vector = []
            for position in np.arange(lag, d.shape[1]):
                lag_vector.append((d[:, position] - d[:, position - lag])**2)
            sum_values = np.sum(lag_vector)
            assert (d.shape[1] - lag) == len(lag_vector)
            r = sum_values / (d.shape[1] - lag)
            msd['lag_values'].append(lag)
            msd['msd_values'].append(r)
        msd['lag_values'] = np.asarray(msd['lag_values'])
        msd['msd_values'] = np.asarray(msd['msd_values'])
        self.msd_data = msd

    
    def plot_msd_curve(self, plot_title=None, height=500, width=500, template='simple_white', save_path=None, **kwargs):
        xvalues = self.msd_data['lag_values'] * self.time_int
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xvalues, 
                                 y=self.msd_data['msd_values'],
                                 **kwargs))
        fig.update_layout(template=template,
                          height=height,
                          width=width)
        fig.update_xaxes(title='Time Lag (s)')
        fig.update_yaxes(title='Mean Squared Displacement')
        fig.update_layout(title = {'text': plot_title,
                                   'xanchor': 'center',
                                   'y': 0.9,
                                   'x': 0.5})
        fig.show()
        if save_path is not None:
            fig.write_image(save_path)
            

    
