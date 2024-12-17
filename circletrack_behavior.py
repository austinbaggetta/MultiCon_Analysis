import pandas as pd
import numpy as np
import xarray as xr
import yaml
import os
import re
import glob
import pingouin as pg
import plotly.graph_objects as go
from os.path import join as pjoin
from scipy.stats import norm


def get_file_list(rpath):
    """
    Get list of circle_track.csv files by setting rpath.
    Args:
        rpath : str
            path to where circle_track.csv file is; can use **/** in path if all directories follow a general structure
            (e.g '../LargeScale/MultiCon_Verification/Data/**/**/**/circle_track.csv')
    Returns:
        file_list: list
    """
    ## Create list of files
    file_list = glob.glob(rpath)
    return file_list


def get_mouse(file_list, str2match):
    """
    Used to extract mouse name from file string.
    Args:
        file_list : list
            contains a list of paths as strings from get_file_list function
        str2match : str
            regex pattern to extract mouse name (e.g. '(mc[0-9]+)')
    Returns: str
        mouse name (e. g. mc01)
    """
    ## Use a for loop to extract mouse name from every file in file_list
    mouseID = re.search(str2match, file_list)
    return mouseID.group()


def combine(file_list, mouse_list):
    """
    Combine file_list with mouse_list
    Args:
        file_list : list
            contains a list of paths as strings from get_file_list function
        mouse_list : list
            contains a list of mouse names as strings from get_mouse function within a for loop
    Returns:
        combined : pandas.DataFrame
            columns filepath and mouse
    """
    files = pd.DataFrame(file_list, columns = ['filepath'])
    mice = pd.DataFrame(mouse_list, columns = ['mouse'])
    combined = pd.concat([files, mice], axis = 1)
    return combined


def subset_combined(combined_df, mouse):
    """
    Subsets combined file_list and mouse_list based on mouse argument
    Args: 
        combined_df : pandas.DataFrame
            pd.DataFrame from combine function
        mouse : str
            mouse name (e.g. 'mc01')
    Returns:
        file_list : pandas.DataFrame
            contains paths for the circle_track.csv data for the mouse of interest
    """
    ## Subsets combined file_list and mouse_list based on mouse string
    files = combined_df.loc[combined_df.mouse == mouse]
    file_list = files.filepath
    return file_list


def crop_data(data):
    """
    Crop data from START to TERMINATE
    Args:
        data : pandas.DataFrame
            circle_track.csv data with three columns
    Returns:
        df : pandas.DataFrame
    """
    ## Get data from START to TERMINATE
    if any(data.event == 'TERMINATE'):
        df = data[np.where(data['event'] == 'START')[0][0]:np.where(data['event'] == 'TERMINATE')[0][0]+1]
    else:
        df = data[np.where(data['event'] == 'START')[0][0]:]
    return df


def get_rewarding_ports(data):
    """
    Get rewarding ports for that session.
    Args:
        data : pandas.DataFrame
            circle_track.csv data with three columns
    Returns:
        reward_first, reward_second : str
            returns string of which ports were the rewarding ports in that session
    """
    ## In more recent circle track experiments, initialization was added to easily get rewarding ports
    ## Previous experiments require assessing from rewarding licks
    if any(data.event == 'initializing'):
        reward_ports = data.loc[data.event == 'initializing'].reset_index(drop=True)
        return reward_ports
    else:
        all_rewards = data.data.loc[data.event == 'REWARD'].reset_index(drop=True)
        reward_ports = pd.DataFrame(all_rewards.unique(), columns=['data'])
        if reward_ports.empty:
            reward_first = None
            reward_second = None
        elif len(reward_ports) == 1:
            reward_first = reward_ports.iloc[0, 0]
            reward_second = None
        else:
            reward_first = reward_ports.iloc[0, 0]
            reward_second = reward_ports.iloc[1, 0]
        return reward_first, reward_second
    
    
def normalize_timestamp(data):
    """
    Convert timestamps to seconds
    Args:
        data : pandas.DataFrame
            circle_track.csv data with three columns
    Returns:
        data : pandas.DataFrame
            timestamp column converted to seconds
    """
    time = data.loc[data.event == 'START', 'timestamp'].reset_index()
    start_time = float(time.timestamp[0])
    data.timestamp = (pd.to_numeric(data.timestamp) - start_time)
    return data


def get_correct_direction(a, roi_bin_size=50, reward_direction=1):
    """
    Determines whether the mouse is moving in the correct direction.
    Args:
        a : np.array, pd.Series, list
            Angular position of the mouse (in degrees)
        roi_bin_size : int
            how large the angular bin sizes should be to track direction
        reward_direction : int
            set as 1 or -1 dependingn on the maze.yml for that circle track
    Returns:
        direction_boolean : np.array
            whether or not the mouse was in the correct direction, labels every frame
    """
    direction_boolean = np.full(len(a), False)
    rois = np.arange(0, 360, roi_bin_size)
    roi_last = None
    for angle_idx, ang in enumerate(a):
        dif = ang - rois
        idx = (np.abs(dif)).argmin()
        cur_roi = rois[idx]

        if cur_roi != roi_last:
            try:
                prev_roi = rois[idx + np.sign(reward_direction)]
            except IndexError:
                prev_roi = rois[(idx + np.sign(reward_direction)) % len(rois)]
            
            if roi_last == prev_roi:
                direction_boolean[angle_idx] = True
            else:
                direction_boolean[angle_idx] = False
            roi_last = cur_roi
        else:
            direction_boolean[angle_idx] = direction_boolean[angle_idx-1]
    return direction_boolean


def set_track_and_maze(maze_number, track='condos'):
    """
    Creates a dataframe of the angular positions of all ports in a specified maze. Needed for dprime calculations.
    Args:
        track : str
            determines what circle track setup you are using; options are one of ['condos', 'clear']
        maze_number: str
            determines what maze you are using; options are one of ['maze1', 'maze2', 'maze3', 'maze4'] for condos
            one of ['maze1'] for track == 'clear' (since there is only one maze)
    Returns:
        reward_ports : pandas.DataFrame
            pd.DataFrame of each ports angular position
    """
    if track == 'condos':
        ## Set angular positions of all ports from the condos (4 tracks Brian and I built)
        reward_ports = pd.DataFrame({'reward1': [90, 90, 90, 90], 'reward2': [20, 25, 27, 45],
                                     'reward3': [340, 347, 345, 0], 'reward4': [296, 305, 302, 315],
                                     'reward5': [253, 265, 259, 270], 'reward6': [210, 220, 215, 225],
                                     'reward7': [169, 178, 171, 180], 'reward8': [128, 135, 131, 135]})
        ## Name index based on maze
        reward_ports.index = ['maze1', 'maze2', 'maze3', 'maze4']
    elif track == 'clear':
        ## Set angular positions of all ports from clear circle track (Phil's original design)
        reward_ports = pd.DataFrame({'reward1': [], 'reward2': [], 'reward3': [], 'reward4': [],
                                     'reward5': [], 'reward6': [], 'reward7': [], 'reward8': []})
        ## Name index based on maze; only one maze
        reward_ports.index = ['maze1']
    else:
        raise Exception("No track set! Must be either 'condos' or 'clear'!")
    ## Choose reward ports for specified maze
    if maze_number == 'maze1':
        ports = reward_ports.loc[reward_ports.index == 'maze1']
    elif maze_number == 'maze2':
        ports = reward_ports.loc[reward_ports.index == 'maze2']
    elif maze_number == 'maze3':
        ports = reward_ports.loc[reward_ports.index == 'maze3']
    elif maze_number == 'maze4':
        ports = reward_ports.loc[reward_ports.index == 'maze4']
    else:
        raise Exception("No maze set! Must be either ['maze1', 'maze2', 'maze3', 'maze4']")
    ## Return output
    return ports


def linearize_trajectory(df, angle_type, shift_factor, inner_d=25, outer_d=30, desired_unit='cm'):
    """
    Linearizes circular position into physical length units.
    Args:
        df : pandas.DataFrame
            behavior data that contains angular position
        angle_type : str
            one of ['degrees', 'radians']
        d1 : float
            diameter between inner walls of the circular track
        d2 : float
            diameter between outer walls of the circular track
        unit : str
            one of ['cm', 'in']
        shift_factor : float
            used to shift the 0 position; by default None, which implies no shifting
            if using degrees shift_factor is in degrees - if in radians, use np.pi
    Returns:
        linearized_position | result : list
    """
    if desired_unit == 'cm':
        ## convert inches to cm
        d1 = inner_d * 2.54
        d2 = outer_d * 2.54
    elif desired_unit == 'in':
        d1 = inner_d
        d2 = outer_d
    ## Take the average of the two since the mouse is between the inner and outer wall
    r = np.mean([(d1/2), (d2/2)])
    if angle_type == 'degrees':
        linearized_position = []
        for i in np.arange(0, len(df)):
                linearized_position.append(((df.a_pos.iloc[i] + shift_factor) / 360) * 2 * np.pi * r)
        return linearized_position
    elif angle_type == 'radians':
        angles = []
        for i in np.arange(0, len(df)):
                angles.append((df.a_pos.iloc[i] * (np.pi/180)) + shift_factor) ## np.pi/180 converts degrees to radians
        result = np.mod(angles, 2 * np.pi)
        return result


def bin_linearized_position(linearized_trajectory, angle_type='radians', bin_num=8):
    """
    Create a certain number of bins, then determine which bin the data from linearized_trajectory is in.
    Args:
        linearized_trajectory : list
            output from linearize_trajectory function
        angle_type : str
            one of ['degrees', 'radians']; by default radians
        bin_num : int
            number of bins; will end up being bin_num - 1
    Returns:
        binned : list
            list where each linearized position is labeled by what bin it is a part of
    """
    if angle_type == 'radians':
        bins = np.linspace(0, 2 * np.pi, (bin_num+1))
        bins = bins[::-1]
        binned = np.digitize(linearized_trajectory, bins)
    elif angle_type == 'degrees':
        bins = np.linspace(0, np.max(linearized_trajectory), (bin_num))
        bins = bins[::-1]
        binned = np.digitize(linearized_trajectory, bins)
    return binned


def get_trials(angular_position, jump_val=310, angle_accumulation=-314, min_trial_length=480, convert_to_rad=False):
    """
    Labels each frame as part of a trial.
    A trial is determined as:
        a) Whether the mouse passed the zero location in the correct direction
        b) Whether it accumulated a full rotation around the circle in the correct direction
        c) Is longer than 40 frames, which helps get rid of small blips
    Args:
        angular_position : np.array or list
        jump_val : int
            threshold for determining whether the mouse passed the zero location.
            either in degrees or radians
        angle_accumulation : int
            how many degrees/radians are needed for the cumulative sum of the trial
        min_trial_length: int
            how many frames the minimum trial must be. Can help get rid of blips where the mouse is sitting at the zero location.
        convert_to_rad : bool
            whether or not you want to convert your angular position data from degrees to radians. If in radians, must account for
            this in the jump_val and angle_accumulation values
    Returns:
        trials : np.array
            numpy array the same length as your angular position data labeling each value as part of a trial
    """
    trials = np.full(len(angular_position), np.nan)
    if convert_to_rad:
        lin_pos = angular_position * (np.pi/180)
    else:
        lin_pos = angular_position

    prev_idx = 0
    diffs = np.diff(lin_pos)
    for trial, jump in enumerate(np.where(diffs > jump_val)[0]):
        ## Account for first trial
        if trial == 0:
            trials[prev_idx:jump+1] = trial
            prev_idx = jump+1
            prev_trial = trial ## create prev_trial variable to account for some "trials" in the loop not meeting trial criteria
        else:
            if (np.cumsum(diffs[prev_idx:jump])[-1] <= angle_accumulation) and (len(diffs[prev_idx:jump]) > min_trial_length):
                trials[prev_idx:jump+1] = prev_trial + 1
                prev_idx = jump+1
                prev_trial = prev_trial + 1
            else:
                ## If the data points don't meet the above criteria, set those data points as part of the previous trial
                trials[prev_idx:jump+1] = prev_trial
                prev_idx = jump+1
        
        ## Account for last trial
        if trial == len(np.where(diffs > jump_val)[0]) - 1:
            trials[prev_idx:] = prev_trial + 1
    return trials


def label_lick_trials(aligned_behavior, lick_tmp, trials):
    """
    Labels a dataframe containing lick information with what trial those licks occurred during.
    Args:
        aligned_behavior : pandas.DataFrame
            output from load_and_align_behavior function - df with at least columns timestamp, trial
        lick_tmp : pandas.DataFrame
            output from get_licks function - df with columns timestamp, event (LICK), data (which port)
        trials : numpy.ndarray
            array the same length as aligned_behavior.timestamp - labels each frame with what trial that frame is
    Returns:
        lick_data : pandas.DataFrame
            df with same columns as lick_tmp, but now with trials
    """
    ## Create a trial column filled with NaN
    lick_tmp.insert(3, 'trial', np.nan)
    ## Create empty dataframe
    lick_data = pd.DataFrame()
    for trial in np.unique(trials):
        ## Subset aligned_behavior by a given trial
        behavior = aligned_behavior.loc[trials == trial]
        ## Get the first and last timestamp to determine the window
        first_timestamp = behavior.timestamp.to_numpy()[0]
        last_timestamp = behavior.timestamp.to_numpy()[-1]
        ## Create a temporary df by subsetting lick_tmp with the df values between the first and last timestamp
        licks = lick_tmp[(lick_tmp.timestamp >= first_timestamp) & (lick_tmp.timestamp < last_timestamp)]
        ## Loop through each row in licks and set the NaN value to the trial value
        for i, row in licks.iterrows():
            licks.at[i,'trial'] = trial
        ## Combine
        lick_data = pd.concat([lick_data, licks])
    return lick_data


def get_forward_reverse_trials(behav, percent_correct=90):
    """
    After determining number of trials, separate trials into trials in the forward (correct) direction or reverse (incorrect) direction.
    Args:
        behav : pandas.DataFrame
            preprocessed behavior dataframe with trials and correct_dir column
        percent_correct : int
            how much of the trial has to be in the correct direction
    Returns:
        forward, backward : np.array
            array with values indicating forward and reverse trials
    """
    forward, reverse = [], []
    for trial in behav['trials'].unique():
        tdata = behav[behav['trials'] == trial]
        if (np.sum(tdata['correct_dir']) / tdata.shape[0] * 100) > percent_correct:
            forward.append(trial)
        else:
            reverse.append(trial)
    return np.asarray(forward), np.asarray(reverse)


def dprime_metrics(data, mouse, day, reward_ports, reward_index='one', forward_reverse='all', go_trials=2, nogo_trials=6, **kwargs):
    """
    Calculates hits, misses, false alarms, correct rejections, and dprime.
    Args:
        lick_data : pandas.DataFrame
            output from label_lick_trials function - df with columns timestamp, event, data, and trial
        trials : numpy.ndarray
            array the same length as aligned_behavior.timestamp - labels each frame with what trial that frame is
            output from get_trials function
        reward_one, reward_two : str
            output from get_rewarding_ports function - name of rewarding ports (reward1, reward5, for example)
        reward_index : str
            determines whether the reward list starts from 0 or from 1, one of ['zero', 'one']
    Returns:
        signal : dictionary
            dictionary with keys hits, miss, FA, CR, dprime   
    """
    ## Create nonreward list
    if reward_index == 'zero':
        nonreward_list = [x for x in np.arange(0, 8)]
    elif reward_index == 'one':
        nonreward_list = [x for x in np.arange(1, 9)]
    
    for port in reward_ports:
        nonreward_list.remove(port)
    
    if len(reward_ports) > 1:
        reward_one, reward_two = reward_ports[0], reward_ports[1]

    signal = {'mouse': [], 'day': [], 'trial': [], 'hits': [], 'miss': [], 'FA': [], 'CR': [], 'dprime': []}
    if forward_reverse == 'forward':
        forward_trials, _ = get_forward_reverse_trials(data, **kwargs)
        trial_list = forward_trials
    elif forward_reverse == 'reverse':
        _, reverse_trials = get_forward_reverse_trials(data, **kwargs)
        trial_list = reverse_trials 
    elif forward_reverse == 'all':
        trial_list = np.unique(data['trials'])
    for trial in trial_list:
        trial_data = data.loc[(data['trials'] == trial) & (data['lick_port'] != -1)].reset_index(drop=True)
        nonreward_ports = {nonreward_list[0]: [], nonreward_list[1]: [], nonreward_list[2]: [], nonreward_list[3]: [], nonreward_list[4]: [], nonreward_list[5]: []}
        ## Loop through all rows of trial data
        correct_licks = 0
        incorrect_licks = 0
        reward_one_licks = 0
        reward_two_licks = 0
        for row in np.arange(0, trial_data.shape[0]):
            if (trial_data.loc[row, 'lick_port'] == reward_one):
                reward_one_licks += 1
            elif (trial_data.loc[row, 'lick_port'] == reward_two):
                reward_two_licks += 1
            else:
                for key in nonreward_ports:
                    if (trial_data.loc[row, 'lick_port'] == key):
                        nonreward_ports[key].append(1)
        
        if reward_one_licks > 0:
            correct_licks += 1
        else:
            correct_licks = correct_licks 

        if reward_two_licks > 0:
            correct_licks += 1
        else:
            correct_licks = correct_licks

        for key in nonreward_ports:
            if np.sum(nonreward_ports[key]) > 0:
                incorrect_licks += 1
            else:
                incorrect_licks = incorrect_licks
        # Get rates for hits, misses, false alarms, and correct rejections.
        hit_rate = correct_licks / go_trials
        miss_rate = (go_trials - correct_licks) / go_trials
        FA_rate = incorrect_licks / nogo_trials
        CR_rate = (nogo_trials - incorrect_licks) / nogo_trials
        ## Adjust values to correct d' of infinity or -infinity
        hit_for_dprime = (correct_licks+0.5) / (go_trials+1)
        FA_for_dprime = (incorrect_licks+0.5) / (nogo_trials+1)
        ## Append to dict
        signal['mouse'].append(mouse)
        signal['day'].append(day)
        signal['trial'].append(trial)
        signal['hits'].append(hit_rate)
        signal['miss'].append(miss_rate)
        signal['FA'].append(FA_rate)
        signal['CR'].append(CR_rate)
        signal['dprime'].append(norm.ppf(hit_for_dprime) - norm.ppf(FA_for_dprime))
    return signal


def aggregate_metrics(metrics, bin_size=5, variable_of_interest='CR'):
    """
    Bins your hits, misses, FA, CR, and dprime according to bin_size.
    """
    binned_metrics = pd.DataFrame()
    for mouse in np.unique(metrics['mouse']):
        for day in np.unique(metrics['day']):
            df = pd.DataFrame(columns=['mouse', 'group', 'day', 'binned_trial', variable_of_interest])
            day_data = metrics[(metrics['mouse'] == mouse) & (metrics['day'] == day)].reset_index(drop=True)
            binned_data = bin_data(day_data[variable_of_interest], bin_size=bin_size)
            trial_nums = list(np.arange(1, len(binned_data)+1) * bin_size)
            df['binned_trial'] = trial_nums
            df[variable_of_interest] = binned_data
            df['mouse'] = mouse
            df['group'] = np.unique(day_data['group'])[0]
            df['day'] = day
            binned_metrics = pd.concat([binned_metrics, df], ignore_index=True)
    return binned_metrics


def bin_data(data, bin_size=2):
    """
    Used to bin percent correct if calculated on a trial by trial basis.
    Args:
        data : list
        bin_size : int
    Returns:
        binned_data : list
    """
    bins = np.arange(0, len(data), bin_size)
    binned = np.split(data, bins)
    return [np.nanmean(bin) for bin in binned if bin.size > 0]


def lick_accuracy(df, port_list, lick_threshold, by_trials=False):
    """
    Used to calculate lick accuracy of a given lick within a bout of licks.
    Args:
        df : pandas.DataFrame
            preprocessed behavior containing columns for trials, lick_ports
        port_one, port_two : int
            which ports were rewarded (e.g. 5)
        lick_threshold : int
            which lick you want to look at in a bout of licks
        by_trials : boolean
            if True, will calculate percent correct within a trial. By default False.
    Returns:
        percent_correct : float or list
            returns a single value when not calculated on a trial by trial basis
    """
    if by_trials:
        percent_correct = []
        for trial in np.unique(df['trials']):
                count = 0
                lick_port = np.nan
                trial_behav = df[df['trials'] == trial]
                licks = trial_behav[trial_behav['lick_port'] != -1].reset_index(drop=True)
                if licks.empty:
                    percent_correct.append(np.nan)
                else:
                    for idx, _ in licks.iterrows():
                        if lick_port != licks.loc[idx, 'lick_port']:
                            count = 1
                        else:
                            count += 1
                        
                        if count < lick_threshold - 1:
                            licks.loc[idx, 'threshold_reached'] = False
                        elif count == lick_threshold:
                            licks.loc[idx, 'threshold_reached'] = True
                        else:
                            licks.loc[idx, 'threshold_reached'] = False

                        lick_port =  licks.loc[idx, 'lick_port']

                    count_licks = licks[['lick_port', 'threshold_reached']].groupby(['lick_port'], as_index=False).agg({'threshold_reached': 'sum'})
                    if count_licks['threshold_reached'].dropna().sum() == 0:
                        percent_correct.append(np.nan)
                    else:
                        reward_licks = 0
                        for reward_port in port_list:
                            try:
                                reward_licks = np.nansum([reward_licks, count_licks['threshold_reached'][count_licks['lick_port'] == reward_port].values[0]])
                            except:
                                pass
                        percent_correct.append(reward_licks / count_licks['threshold_reached'].dropna().sum() * 100)
                
    else:
        count = 0
        lick_port = np.nan         
        if type(df) == xr.DataArray:
            licks = pd.DataFrame({'lick_port': df['lick_port'][df['lick_port'] != -1].values})
        else:
            licks = df[df['lick_port'] != -1].reset_index(drop=True)
        if licks.empty:
            percent_correct = np.nan 
        else:
            for idx, _ in licks.iterrows():
                if lick_port != licks.loc[idx, 'lick_port']:
                    count = 1
                else:
                    count += 1
                
                if count < lick_threshold - 1:
                    licks.loc[idx, 'threshold_reached'] = False
                elif count == lick_threshold:
                    licks.loc[idx, 'threshold_reached'] = True
                else:
                    licks.loc[idx, 'threshold_reached'] = False

                lick_port =  licks.loc[idx, 'lick_port']

            count_licks = licks[['lick_port', 'threshold_reached']].groupby(['lick_port'], as_index=False).agg({'threshold_reached': 'sum'})
            if count_licks['threshold_reached'].dropna().sum() == 0:
                percent_correct = np.nan 
            else:
                reward_licks = 0
                for reward_port in port_list:
                    try:
                        reward_licks = np.nansum([reward_licks, count_licks['threshold_reached'][count_licks['lick_port'] == reward_port].values[0]])
                    except:
                        pass
                percent_correct = reward_licks / count_licks['threshold_reached'].dropna().sum() * 100
    return percent_correct


def performance_drop(accuracy, day_list, replace=False):
    """
    Calculate the difference in lick accuracy between a given day and the day after it.
    Args:
        accuracy : pd.DataFrame
            df with columns mouse, day, and percent_correct
        day_list : list
            list containing day(s) of interest, e.g. [5, 10, 15] will calculate the difference
            between days 5-6, 10-11, and 15-16
        replace : bool
            replace any negative performance drop (mouse got better next session) with zero
    Returns:
        performance : pd.DataFrame
            df with columns mouse, drop (difference), and day
    """
    performance = pd.DataFrame()
    for day in day_list:
        first_day = accuracy.loc[accuracy['day'] == day, 'percent_correct'].to_numpy()
        second_day = accuracy.loc[accuracy['day'] == day + 1, 'percent_correct'].to_numpy()
        drop = first_day - second_day
        df = pd.DataFrame({'mouse': np.unique(accuracy['mouse']),
                           'drop': drop,
                           'day': day})
        performance = pd.concat([performance, df])
    if replace:
        performance.loc[performance['drop'] < 0, 'drop'] = 0
    return performance


def fix_lick_ports(behav, reward_one, reward_two):
    """
    Used to fix lick port identity when water == True in cohort0's behavior dataframe.
    Args:
        behav : pandas.DataFrame
        reward_one, reward_two : numpy.int64
    Returns:
        behav : pandas.DataFrame
    """
    r1_min, r1_max = np.min(behav['lin_position'][behav['lick_port'] == reward_one]), np.max(behav['lin_position'][behav['lick_port'] == reward_one])
    r2_min, r2_max = np.min(behav['lin_position'][behav['lick_port'] == reward_two]), np.max(behav['lin_position'][behav['lick_port'] == reward_two])
    for idx in np.arange(0, len(behav)):
        if (behav.loc[idx, 'water'] == True) & (behav.loc[idx, 'lick_port'] == -1):
            if (behav.loc[idx, 'lin_position'] >= r1_min) | (behav.loc[idx, 'lin_position'] <= r1_max):
                behav.loc[idx] = behav.loc[idx].replace(to_replace={-1: reward_one})
            elif (behav.loc[idx, 'lin_position'] >= r2_min) | (behav.loc[idx, 'lin_position'] <= r2_max):
                behav.loc[idx] = behav.loc[idx].replace(to_replace={-1: reward_two})
    return behav


def normalized_probe_metric(lick_array, reward_one, reward_two):
    """
    Calculate the average lick accuracy metric between -1 and 1. Ports next to a rewarded port are
    given a value of 0, whereas ports two spaces away are given a value of -1.
    
    ** This assumes that ports are orthogonal to each other, doesn't work otherwise. **

    Args:
        lick_data : np.array
            array of lick port values where a mouse licked during the probe
        reward_one, reward_two : int
            port id of rewarded ports 
    Returns:
        mean value between -1 and 1
    """
    if (type(reward_one)) and (type(reward_two)) == str:
        reward_one = int(reward_one[-1])
        reward_two = int(reward_two[-1])
    
    port_dict = {port: np.nan for port in np.arange(1, 9)}
    port_dict[reward_one], port_dict[reward_two] = [1, 1]

    if (reward_one == 3) and (reward_two == 7):
        port_dict[(reward_one + 1)], port_dict[(reward_one - 1)] = [0, 0]
        port_dict[(reward_one + 2)], port_dict[(reward_one - 2)] = [-1, -1]
        port_dict[(reward_two + 1)], port_dict[(reward_two - 1)] = [0, 0]
    elif (reward_one == 4) and (reward_two == 8):
        port_dict[(reward_one + 1)], port_dict[(reward_one - 1)] = [0, 0]
        port_dict[(reward_one + 2)], port_dict[(reward_one - 2)] = [-1, -1]
        port_dict[(reward_one - 3)], port_dict[(reward_two - 1)] = [0, 0]
    elif (reward_one == 2) and (reward_two == 6):
        port_dict[(reward_one + 1)], port_dict[(reward_one - 1)] = [0, 0]
        port_dict[(reward_two + 1)], port_dict[(reward_two - 1)] = [0, 0]
        port_dict[(reward_two + 2)], port_dict[(reward_two - 2)] = [-1, -1]
    elif (reward_one == 1) and (reward_two == 5):
        port_dict[(reward_two + 1)], port_dict[(reward_two -1)] = [0, 0]
        port_dict[(reward_two + 2)], port_dict[(reward_two - 2)] = [-1, -1]
        port_dict[(reward_two + 3)], port_dict[(reward_one + 1)] = [0, 0]
    else:
        raise Exception('Incorrect port values assigned!')
    
    value_list = []
    for value in lick_array:
        value_list.append(port_dict[value])
    return np.mean(value_list)


def days_to_criteria(lick_df, mouse, criteria_val, pc_column='percent_correct'):
    """
    Calculate number of days to reach criteria for a mouse.
    Args:
        lick_df : pandas.DataFrame
            dataframe containing mouse, day, percent_correct
        mouse : str
            mouse name
        criteria_val : int/float
            percent correct values less than or equal to
    Returns:
        an array with number of days to reach criteria in each context
    """
    sub_df = lick_df[(lick_df['mouse'] == mouse) & (lick_df[pc_column] >= criteria_val)].reset_index(drop=True)
    return np.concatenate((np.asarray([sub_df['day'][0]]), np.diff(sub_df['day'])))


def find_center(x, y):
    """
    Find center of circle.
    Args:
        x, y : float
    """
    x_extrema = [min(x), max(x)]
    y_extrema = [min(y), max(y)]
    return (np.mean(x_extrema), np.mean(y_extrema))


def rotate(p, origin, degrees=0):
    """
    Rotates a point about a given origin.
    Args:
        p : tuple
            x, y position of your point. Can also give multiple points in the case of a polygon.
        origin : tuple
            x, y position of point you want to rotate about
        degrees : float
            amount of degrees you want to rotate point
    Returns:
        rotated tuple
    """
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


def rotate_ports(input_maze, output_maze, reward_one, reward_two):
    """
    Used to rotate port identies to check for licking preference based on external cue.
    Args:
        input_maze : str
            which maze the mouse is coming from
        output_maze : str
            which maze the mouse went to
        reward_one, reward_two : int
            interger values for input_mazes' reward ports
    Returns:
        the equivalent ports of reward_one, reward_two in the output_maze based on cue rotation but sorted
    """
    ports = []
    if input_maze == 'maze1':
        mazes = {'maze2': {1: 7, 2: 8, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6},
                 'maze3': {1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 1, 8: 2},
                 'maze4': {1: 5, 2: 6, 3: 7, 4: 8, 5: 1, 6: 2, 7: 3, 8: 4}}
    elif input_maze == 'maze2':
        mazes = {'maze1': {1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 1, 8: 2},
                 'maze3': {1: 5, 2: 6, 3: 7, 4: 8, 5: 1, 6: 2, 7: 3, 8: 4},
                 'maze4': {1: 7, 2: 8, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}}
    elif input_maze == 'maze3':
        mazes = {'maze1': {1: 7, 2: 8, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6},
                 'maze2': {1: 5, 2: 6, 3: 7, 4: 8, 5: 1, 6: 2, 7: 3, 8: 4},
                 'maze4': {1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 1, 8: 2}}
    elif input_maze == 'maze4':
        mazes = {'maze1': {1: 5, 2: 6, 3: 7, 4: 8, 5: 1, 6: 2, 7: 3, 8: 4},
                 'maze2': {1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 1, 8: 2},
                 'maze3': {1: 7, 2: 8, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}}
    else:
        raise Exception('Incorrect input maze!')
    ports.append(mazes[output_maze][reward_one])
    ports.append(mazes[output_maze][reward_two])
    ports = sorted(ports)
    return ports


def pick_context_day(df, col_name='session', day_index=-1, contexts=['A', 'B', 'C', 'D', 'A2']):
    """
    Pick a specific day within a context.
    Args:
        df : pandas.DataFrame
        col_name : str
            column of interest you want to index from. By default session (as session is usually A, B, C, etc)
        day : int
            value of what index you are interested in. First would be 0, second would be 1, last would be -1. By default -1.
        contexts : list
            list of context identifying strings, e.g. ['A', 'B', 'C', 'D', 'A2']
    Returns:
        index_list : list
            list of indices of whatever day you want from that context
    """
    index_list = []
    for context in contexts:
        c_data = df[df[col_name] == context].reset_index()
        index_list.append(c_data.loc[c_data.index[day_index], 'index'])
    return index_list


def relative_port_distance(reward_one, reward_two, maze):
    """
    Determine minimum port angle distance from wall cue.
    Args:
        reward_one, reward_two : int
            reward port values
        maze : str
            maze string
    Returns:

    """
    port_angles = [0, 45, 90, 135, 180, 135, 90, 45] ## reference maze is maze4
    if maze == 'maze1':
        shifted_angles = np.roll(port_angles, shift=4)
    elif maze == 'maze2':
        shifted_angles = np.roll(port_angles, shift=2)
    elif maze == 'maze3':
        shifted_angles = np.roll(port_angles, shift=-2)
    elif maze == 'maze4':
        shifted_angles = port_angles
    else:
        raise Exception('Incorrect maze value!')

    ## Get minimum angle value, -1 for index because reward ports are labeled 1-8
    return np.min([shifted_angles[reward_one-1], shifted_angles[reward_two-1]])


def port_lick_chisquared(behav, num_ports=8, probe=True):
    """
    Determine if observed licking at ports is significantly different than expected by chance.
    Args:
        behav : pandas.DataFrame
            behavior dataframe with probe and lick_port info
        num_ports : int
            number of ports in circle track; by default 8
        probe : bool
            Whether to look at licking only during probe or outside of probe; by default True
    Returns:
        expected, observed, stats : pandas.DataFrame
            read pinguoin documentation on pg.chi2_independence
    """
    if probe:
        data = behav[behav['probe']]
    else:
        data = behav[~behav['probe']]

    ## Calculate expected value at a port E(P) = 1/n_p x total_licks
    total_licks = data[data['lick_port'] != -1].shape[0]
    expected_value = (1/num_ports) * total_licks ## 8 water ports by default
    
    actual_licks, expected_licks = pd.DataFrame(), pd.DataFrame()
    actual_licks['lick_port'] = data['lick_port'][data['lick_port'] != -1].reset_index(drop=True)
    expected_licks['lick_port'] = np.repeat(np.arange(1, num_ports+1), repeats=round(expected_value))
    actual_licks['type'] = 'actual'
    expected_licks['type'] = 'expected'
    lick_df = pd.concat([expected_licks, actual_licks])
    expected, observed, stats = pg.chi2_independence(lick_df, x='lick_port', y='type')
    return expected, observed, stats


def convert_to_cm(behav=None, x=None, y=None, pixels_per_cm=5.5380577427821525, update_dataframe=False):
    """
    Takes the x and y values in pixels from the preprocessed behavior data and converts them to centimeters.
    Pixels_per_cm was determined from a video recording of a hand being tracked along the edge of the maze.
    Args:
        behav : pandas.DataFrame
            output from 02_process_circletrack; by default None
        x, y : array
            x and y position in pixels from processed output
        pixels_per_cm : float
            code to get this value is in CellOverlap.ipynb in MultiCon_Imaging2
        update_dataframe = bool
            whether or not you insert the values as columns into behav
    """
    if x is None and y is None:
        x_cm = behav['x'] / pixels_per_cm
        y_cm = behav['y'] / pixels_per_cm
    elif behav is None:
        x_cm = np.array(x) / pixels_per_cm 
        y_cm = np.array(y) / pixels_per_cm
    else:
        raise Exception('All three arguments cannot be none!')
    
    if update_dataframe:
        behav.insert(0, 'x_cm', x_cm)
        behav.insert(1, 'y_cm', y_cm)
        return behav
    else:
        return x_cm, y_cm


def front_back_ports(reward_list, zero_start=False, nports=8):
    """
    For a reward port, gets the port directly before and directly after it.
    Args:
        reward_list : list
            list of reward port values as integers, e.g. [1, 5]
        zero_start : boolean
            whether the first reward port is reward1 or reward0; by default False (reward1)
        nports : int
            number of reward ports; by default 8
    Returns:
        front_ports, back_ports : lists
            list of the reward ports directly before the reward ports in reward list
            list of the reward ports directly after the reward ports in reward list
    """
    front_ports = []
    back_ports = []
    if zero_start:
        ports = np.arange(0, nports)
    else:
        ports = np.arange(1, nports + 1)

    for port_value in ports:
        if port_value in reward_list:
            port_list = ports
            if (port_value == port_list[0]):
                port_list = np.roll(port_list, shift=1)
                front_ports.append(port_list[port_value - 1])
                back_ports.append(port_list[port_value + 1])
            elif (port_value == port_list[-1]):
                port_list = np.roll(port_list, shift=-1)
                front_ports.append(port_list[port_value - 3])
                back_ports.append(port_list[port_value - 1])
            else:
                port_list = np.roll(port_list, shift=0)
                front_ports.append(port_list[port_value - 2])
                back_ports.append(port_list[port_value])
    return front_ports, back_ports


def label_lick_bout(df):
    """
    Used to label each bout of licking.
    Args:
        df : pandas.DataFrame
            preprocessed circletrack or lineartrack behavior
    Returns:
        licks : pandas.DataFrame
            dataframe with lick_bout as an additional column
    """
    licks = df[df['lick_port'] != -1].reset_index(drop=True)
    current_port = licks.loc[0, 'lick_port']
    current_bout = 1
    for idx, row in licks.iterrows():
        if row['lick_port'] == current_port:
            licks.loc[idx, 'lick_bout'] = current_bout
        else:
            current_bout += 1
            current_port = row['lick_port']
            licks.loc[idx, 'lick_bout'] = current_bout
    return licks


def calculate_bins(x, bin_size=0.4):
    """
    Used to calculate bins of a given variable based on bin size.
    """
    min_x, max_x = np.min(x), np.max(x)
    bins = np.arange(min_x, max_x + bin_size, bin_size)
    return bins


