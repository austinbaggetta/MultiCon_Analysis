import pandas as pd
import numpy as np
import itertools

def calculate_overlap(mappings):
    """
    Mappings is the output from Minian cross-registration function calculate_mappings, which calculates pairwise cell identities.
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
               'overlap' : np.nan}
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
    
    overlap_summary : dataframe
        output from calculate_overlap function above
    sessions : list
        list of sessions, for example ['1', '2', '3', '4', '5'] if mouse was in one context for 5 days
    context : str
        demarcates what context the mouse was in, e.g. 'A'
    """
    overlap = pd.DataFrame()
    for i in itertools.combinations(sessions, r = 2):
        data = overlap_summary.loc[(overlap_summary.session_id1 == i[0]) & (overlap_summary.session_id2 == i[1])]
        overlap = pd.concat([overlap, data], ignore_index = True)
    return overlap