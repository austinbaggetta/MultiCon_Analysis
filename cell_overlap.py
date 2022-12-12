import pandas as pd
import numpy as np
import itertools
import datetime

def calculate_overlap(mappings):
    """
    Calculates the cell percent overlap between all pairs of sessions.
    Args:
        mappings : pandas.DataFrame
            mappings is the output from Minian cross-registration function calculate_mappings, which calculates pairwise cell ids
    Returns:
        overlap_summary : pandas.DataFrame
            pd.DataFrame with columns session_id1, session_id2, total (total # cells), pairs (# of cell pairs between two sessions), overlap (percent)
    """
    ## Create empty data frame
    overlap_summary = pd.DataFrame(columns = ['session_id1', 'session_id2', 'total', 'pairs', 'overlap'])
    # Loop through comparisons of same day
    for d1 in mappings.session.columns:
        total = len(mappings.session[d1].dropna().unique())
        pairs = len(mappings.session[[d1, d1]].dropna(how = 'any').drop_duplicates())
        tmp = {'session_id1': d1,
               'session_id2' : d1, 
               'total' : total, 
               'pairs' : pairs, 
               'overlap' : np.nan} ## sets 100% overlap to NaN to allow for better heatmap visualization
        new = pd.DataFrame(data = tmp, index = [0])
        overlap_summary = pd.concat([overlap_summary, new], ignore_index = True)
    ## Loop through all combinations of sessions
    for d1,d2 in itertools.combinations(mappings.session.columns, r = 2):
        a = len(mappings.session[d1].dropna().unique())
        b = len(mappings.session[d2].dropna().unique())
        pairs = len(mappings.session[[d1, d2]].dropna(how = 'any'))
        tmp = {'session_id1': d1,
               'session_id2' : d2, 
               'total' : (a + b - pairs), 
               'pairs' : pairs, 
               'overlap' : (pairs/(a + b - pairs))*100}
        new = pd.DataFrame(data = tmp, index = [0])
        tmp2 =  {'session_id1': d2,
                 'session_id2' : d1, 
                 'total' : (a + b - pairs), 
                 'pairs' : pairs, 
                 'overlap' : (pairs/(a + b - pairs))*100}
        new2 = pd.DataFrame(data = tmp2, index = [0])
        overlap_summary = pd.concat([overlap_summary, new, new2], ignore_index = True)
    return overlap_summary



def within_context_overlap(overlap_summary, sessions):
    """
    Creates a data frame of within context sessions which can subsequently be manipulated.
    Args:
        overlap_summary : pandas.DataFrame
            output from calculate_overlap function 
        sessions : list
            list of sessions as strings, for example ['1', '2', '3', '4', '5'] if mouse was in one context for 5 days
    Returns:
        overlap : pandas.DataFrame
            pd.DataFrame of all session pairwise comparisons
    """
    overlap = pd.DataFrame()
    for i in itertools.combinations(sessions, r = 2):
        data = overlap_summary.loc[(overlap_summary.session_id1 == i[0]) & (overlap_summary.session_id2 == i[1])]
        overlap = pd.concat([overlap, data], ignore_index = True)
    return overlap


def dates_to_days(data, start_date, days):
    """
    Used to convert dates (e.g. '2022_06_08') to days (e.g. 1) for intuitive axes during plotting.
    Args:
        data : pandas.DataFrame
            usually pairwise comparison dataframes, if 'session_id1', etc. are dates
        start_date : str
            start date of the experiment (e.g. '2022_06_08')
        days : int
            number of days the experiment was for
    Returns:
        df : pandas.DataFrame
            dates will be changed to integer days
    """
    ## Start date of experiment
    start = datetime.datetime.strptime(start_date, '%Y_%m_%d')
    ## Set DayID to 1
    DayID = 1
    ## Create a date range from the start of the experiment to the end 
    dates = pd.date_range(start, periods = days)
    ## Initialize empty dictionary
    day_dict = {}
    ## Loop through all dates, add each date as a key to day_dict
    for date in dates:
        day_dict[date.strftime('%Y_%m_%d')] = DayID
        DayID += 1
    df = data.replace(day_dict)
    return df


def between_context_overlap(mouse, overlap, session_ids = ['A5', 'B5', 'C5', 'D5', 'R_A5'], session_type = 'pre'):
    """ 
    Get the overlap values between specific pairs of sessions.
    Args:
        mouse : str
            name of mouse
        overlap : pandas.DataFrame
            pd.DataFrame of cell overlap values determined from calculate_overlap function
        session_ids : list
            list of specific sessions you want to get the overlap values for. The function will compare session_ids[0] to session_ids[1], 1 to 2, etc
        session_type : str
            one of ['pre', 'behavior', 'post']
    """
    if session_type == 'pre':
        ## A to B
        AtoB = overlap.overlap.loc[(overlap.session_id1 == session_ids[0]) & (overlap.session_id2 == session_ids[1])].reset_index()
        AtoB.insert(2, 'mouse', mouse)
        AtoB.insert(3, 'context', 'a_b')
        ## B to C
        BtoC = overlap.overlap.loc[(overlap.session_id1 == session_ids[1]) & (overlap.session_id2 == session_ids[2])].reset_index()
        BtoC.insert(2, 'mouse', mouse)
        BtoC.insert(3, 'context', 'b_c')
        ## C to D
        CtoD = overlap.overlap.loc[((overlap.session_id1 == session_ids[2]) & (overlap.session_id2 == session_ids[3])) | ((overlap.session_id1 == session_ids[2]) & (overlap.session_id2 == session_ids[4]))].reset_index()
        CtoD.insert(2, 'mouse', mouse)
        CtoD.insert(3, 'context', 'c_d_ra')
        output = pd.concat([AtoB, BtoC, CtoD], ignore_index = True)
        return output