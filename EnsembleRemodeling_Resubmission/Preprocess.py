import os
from os.path import join as pjoin
import sys
sys.path.append('..')
import circletrack_neural as ctn
## Set parameters
path = '../../EnsembleRemodeling_Resubmission/circletrack_data'
mouse_list = ['mc03', 'mc06', 'mc07', 'mc09', 'mc11']   
session_dict = {'mc03': ['Training1', 'Training2', 'Training3', 'Training4', 'Reversal1', 'Reversal4', 'Training_Reversal'],
                'mc06': ['Training1', 'Training2', 'Training3', 'Training4', 'Reversal1', 'Reversal2', 'Reversal3', 'Reversal4', 'Training_Reversal'],
                'mc07': ['Training1', 'Training2', 'Training3', 'Training4', 'Reversal1', 'Reversal2', 'Reversal3', 'Reversal4', 'Training_Reversal'],
                'mc09': ['Training1', 'Training2', 'Training3', 'Training4', 'Reversal1', 'Reversal2', 'Reversal3', 'Reversal4', 'Training_Reversal'],
                'mc11': ['Training1', 'Training2', 'Training3', 'Training4', 'Reversal1', 'Reversal2', 'Reversal3', 'Reversal4', 'Training_Reversal']}
cohort_number = 'cohort1'
session_length = '30min'
## Loop through all mice
for mouse in mouse_list:
    dpath = pjoin(path, 'Results/{}/'.format(mouse))
    session_dates = os.listdir(dpath)
    session_dates.sort()
    for i, date in enumerate(session_dates):
        ctn.minian_to_netcdf(path, mouse, date, session_id = session_dict[mouse][i], cohort_number = cohort_number, session_length = session_length,
                             sampling_rate = 1/15, down_sample_factor = 2)
