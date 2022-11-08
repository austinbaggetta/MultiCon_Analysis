import os
import pandas as pd
from os.path import join as pjoin


## This file contains miscellaneous functions that are used after an experiment is completed
def create_timestamps_csv(dpath, spath):
    """
    Create timestamps.csv for minian processing.
    Args:
        dpath : str
            path to a specific mouse, as the function will loop through the dates and timestamps of this mouse
        spath: str
            save path; where to save timestamps.csv to
    Returns:
        timestamps : pandas.DataFrame
            timestamps is a pd.DataFrame saved to csv; you can specify the name of the csv file in spath
    """
    ## Create empty dataframe
    timestamps = pd.DataFrame()
    ## Loop through dpath
    for date in os.listdir(dpath):
        tpath = pjoin(dpath, date)
        times = []
        ## Loop through folders in tpath
        for t in os.listdir(tpath):
            times.append(t)
        ## Add times vector to specific date
        timestamps['{}'.format(date)] = times 
    ## Save timestamps as a csv file
    timestamps.to_csv(spath, index = False)