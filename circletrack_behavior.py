import pandas as pd
import numpy as np
import xarray as xr
import yaml
import os
import re
import glob
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


def get_rewarding_ports(data, processed=False):
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
    if processed:
        rewards = data.loc[data['water'] == True]
        reward_first, reward_second = np.unique(rewards['lick_port'])[0], np.unique(rewards['lick_port'])[1]
        return reward_first, reward_second
    else:
        if any(data.event == 'initializing'):
            reward_ports = data.loc[data.event == 'initializing'].reset_index(drop = True)
            reward_first = reward_ports.data[0]
            reward_second = reward_ports.data[1]
            return reward_first, reward_second
        else:
            all_rewards = data.data.loc[data.event == 'REWARD'].reset_index(drop = True)
            reward_ports = pd.DataFrame(all_rewards.unique(), columns = ['data'])
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


def get_licks(data):
    """
    Get lick data within a session
    Args:
        data : pandas.DataFrame
            circle_track.csv data with three columns
    Returns:
        lick_tmp : pandas.DataFrame
            dataframe with lick events
    """
    ## Get licks
    lick_tmp = data.loc[(data.event == 'LICK') | (data.event == 'REWARD')].replace(to_replace='REWARD', value='LICK').reset_index(drop = True)
    return lick_tmp
    
    
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


def calculate_lick_accuracy(lick_tmp, reward_first, reward_second):
    """
    Calculate lick accuracy for a session.
    Selects the first lick in a bout of licks, calculates the number of first licks per rewarding port.
    Args:
        lick_tmp : pandas.DataFrame
            output from get_licks function
        reward_first, reward_second : str
            rewarding port numbers (e.g. 'reward1')
            can either be manually input or determined from get_rewarding_ports function
    Returns:
        percent_correct : float
            lick accuracy for that session in percentage points
    """
    ## Shift data column, corresponds to licks at reward port
    lick_tmp.insert(loc = 3, 
                    column = 'shifted',
                    value = lick_tmp.data.shift(periods = 1, fill_value = 'first'))
    ## Create boolean of shift to determine first licks
    lick_tmp.insert(loc = 4,
                    column = 'first',
                    value = (lick_tmp.data != lick_tmp.shifted))
    ## Create first lick data frame
    first = lick_tmp.loc[lick_tmp['first']].reset_index(drop = True)
    ## Summarize first lick data
    licks = first.groupby('data', as_index = False).count()
    ## Calculate first lick percent correct, or lick accuracy
    percent_correct = ((licks.event[(licks.data == reward_first) | (licks.data == reward_second)].dropna().sum()) / licks.event.dropna().sum()) * 100
    return percent_correct


def get_total_rewards(file_list, mouse):
    """
    Calculate total rewards for a mouse across all sessions.
    Can be combined with a for loop to get entire cohort's total rewards.
    Args:
        file_list : pandas.DataFrame
            combined file_list and mouse_list from combine function
        mouse : str
            mouse name (e.g. 'mc01')
    Returns:
        reward_data : pandas.DataFrame
            pd.DataFrame with columns total_rewards, mouse, and day
    """
    ## Subset file_list based on mouse
    files = subset_combined(file_list, mouse)
    ## Initialize empty dictionary
    reward_data = {'total_rewards': [], 'mouse': [], 'day': []}
    ## Loop through files
    DayID = 1
    for path in files:
        data = pd.read_csv(path)
        ## Crop data
        cropped_data = crop_data(data)
        ## Get total rewards
        total_rewards = len(cropped_data.loc[cropped_data.event == 'REWARD'])
        ## append to dictionary
        reward_data['total_rewards'].append(total_rewards)
        reward_data['mouse'].append(mouse)
        reward_data['day'].append(DayID)
        ## Add to DayID
        DayID = DayID + 1
    ## Convert to data frame
    reward_data = pd.DataFrame(reward_data)
    return reward_data


def get_lick_accuracy(file_list, mouse):
    """
    Calculate first lick accuracy by dividing the suum of first licks at both rewarding ports by total number of first licks.
    Can be combined with a for loop to get lick accuracy for an entire cohort.
    Args: 
        file_list : pandas.DataFrame
            combined file_list and mouse_list from combine function
        mouse : str
            mouse name (e.g. 'mc01')
    Returns:
        lick_data : pandas.DataFrame
            pd.DataFrame with columns percent_correct, mouse, and day
    """
    ## Subset file_list based on mouse
    files = subset_combined(file_list, mouse)
    ## Initialize empty data frame
    lick_data = {'percent_correct': [], 'mouse': [], 'day': []}
    ## Loop through files
    DayID = 1
    for path in files:
        data = pd.read_csv(path)
        ## Get rewarding ports
        [reward_first, reward_second] = get_rewarding_ports(data)
        ## Crop data
        cropped_data = crop_data(data)
        ## Get licks/rewards
        lick_tmp = get_licks(cropped_data)
        ## Calculate percent correct
        percent_correct = calculate_lick_accuracy(lick_tmp, reward_first, reward_second)
        ## append to dictionary
        lick_data['percent_correct'].append(percent_correct)
        lick_data['mouse'].append(mouse)
        lick_data['day'].append(DayID)
        ## Add to DayID
        DayID = DayID + 1
    ## Convert to data frame
    lick_data = pd.DataFrame(lick_data)
    return lick_data


def get_location_data(file_list, mouse, include_licks = False):
    """
    Get x, y,, and angular position of the mouse across all sessions.
    Args: 
        file_list : pandas.DataFrame
            combined file_list and mouse_list from combine function
        mouse : str
            mouse name (e.g. 'mc01')
        include_licks : boolean
            if True, will include licks and rewards; by default is false
    Returns:
        location_data : pandas.DataFrame
            pd.DataFrame with columns x_pos, y_pos, a_pos, mouse, day
    """
    ## Subset file_list based on mouse
    files = subset_combined(file_list, mouse)
    ## Initialize empty data frame
    location_data = {'timestamp': [], 'x_pos': [], 'y_pos': [], 'a_pos': [], 'mouse': [], 'day': [], 'lick': [], 'reward': []}
    ## Loop through files
    DayID = 1
    for path in files:
        data = pd.read_csv(path)
        ## Crop data
        cropped_data = crop_data(data)
        ## Normalize timestamp
        norm_data = normalize_timestamp(cropped_data)
        ## Separate positional elements
        if include_licks:
            location = norm_data.loc[norm_data.event == 'LOCATION']
            licks_rewards = norm_data.loc[(norm_data.event == 'LICK') | (norm_data.event == 'REWARD')]
        else:
            ## Get location data
            location = norm_data.loc[norm_data.event == 'LOCATION']
            ## append to dictionary
            for i in range(1, len(location)):
                location_data['timestamp'].append(location.timestamp.to_numpy()[i])
                location_data['x_pos'].append(float(re.search(r'X([0-9]+)', location.data.to_numpy()[i]).group(1)))
                location_data['y_pos'].append(float(re.search(r'Y([0-9]+)', location.data.to_numpy()[i]).group(1)))
                location_data['a_pos'].append(float(re.search(r'A([0-9]+)', location.data.to_numpy()[i]).group(1)))
                location_data['mouse'].append(mouse)
                location_data['day'].append(DayID)
                location_data['lick'].append(False)
                location_data['reward'].append(False)
        ## Add to DayID
        DayID = DayID + 1
    ## Convert to data frame
    location_data = pd.DataFrame(location_data)
    return location_data


def get_direction_information(location_data, lagn = 15):
    """
    Determine direction based on angular position.
    Args:
        location_data : pandas.DataFrame
            output from get_location_data function
        lagn : float
            number of frames to calculate the difference between angular positions; 
            by default set to 15 frames since data is sampled at 30 frames per second;
            can be changed to account for temporal downsampling to align miniscope and behavior data
    Returns:
        location_data : pandas.DataFrame
            adds a column titled direction with either correct, wrong, or NA direction (NA if offset is 0);
            direction is correct if difference between angular positions is negative
    """
    ## Calculate difference between angular position rows
    location_data.insert(3, 'a_offset', value = location_data.a_pos.diff(periods = lagn))
    ## Create conditions
    conditions = [
        (location_data.a_offset.lt(0)),
        (location_data.a_offset.gt(0)),
        (location_data.a_offset == 0),
        (location_data.a_offset.isnull())
    ]
    choices = ['correct', 'wrong', 'NA', 'NA']
    ## Determine direction
    location_data['direction'] = np.select(conditions, choices)
    return location_data


def direction_percentage(location_data):
    """
    Calculates percentage of time spent in the correct direction
    Args:
        location_data : pandas.DataFrame
            output from get_location_data and get_direction_information functions
    Returns:
        direction_percent : pandas.DataFrame
            pd.DataFrame with columns direction_percentage, day, and mouse
    """
    direction_percentage = {'direction_percentage': [], 'day': [], 'mouse': []}
    ## Loop through each day
    for day in range(1, len(location_data['day'].unique())+1):
        position_data = location_data.loc[location_data.day == day] ## Subset based on day
        ## Calculate correct direction percentage
        percentage = len(position_data.loc[position_data.direction == 'correct']) / (len(position_data.loc[position_data.direction == 'correct']) + len(position_data.loc[position_data.direction == 'wrong'])) * 100
        ## Append values to empty dictionary
        direction_percentage['direction_percentage'].append(percentage)
        direction_percentage['day'].append(day)
        direction_percentage['mouse'].append(position_data['mouse'].unique()[0])
    ## Convert to dataframe
    direction_percentage = pd.DataFrame(direction_percentage)
    return direction_percentage


def set_track_and_maze(track, maze_number):
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


def align_behavior_frames(df, session='20min', sampling_rate=1/15, plot_frame_usage=False):
    """
    Takes timestamps matrix associated with a behavior recording and a regularly spaced time vector the expected length of the session. 
    For each timeframe in 'time', the closest frame from timestamp column is acquired. 
    Args:
        data : pandas.DataFrame
            minian timestamps from preprocessing; argument set in load_and_align_minian function
        time : list
            regularly spaced time vector the expected length of the session; argument set in load_and_align_minian function
        plot_frame_usage : boolean
            if True, creates a plot of frame usage; by default set to False
    Returns:
        lined_up_timeframes : list
            vector of lined up frames to use to align recording to the time vector.
    """
    if session == '20min':
        frame_count = (
            20 * 60 / sampling_rate
        )  # 20min x 60sec/min / sampling_rate 
    elif session == '30min':
        frame_count = (
            30 * 60 / sampling_rate
        )  # 30min x 60sec/min / sampling_rate
    else:
        raise Exception(
            "Invalid 'session' argument. Must be one of: ['20min', '30min']"
        )
    time = np.arange(0, frame_count * sampling_rate, sampling_rate)
    arg_mins = [np.abs(df['t'] - (t)).argmin() for t in time]
    lined_up_timeframes = np.array(df.loc[arg_mins, 'frame'])
    behav = df.loc[arg_mins]
    if plot_frame_usage:
        duplicated_timeframes = np.unique(lined_up_timeframes, return_counts=True)
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=duplicated_timeframes[0],
                y=duplicated_timeframes[1],
                mode="lines+markers",
                marker_size=5,
            )
        )
        fig.update_layout(
            template="simple_white",
            xaxis_title="Time (ms)",
            yaxis_title="Frequency",
            title_text="Number of times each behavior frame was re-used",
        )
        fig.show(config={"scrollZoom": True})
    return behav

def align_behavior_frames_legacy(df, time, plot_frame_usage=False):
    """
    Takes timestamps matrix associated with a behavior recording and a regularly spaced time vector the expected length of the session. 
    For each timeframe in 'time', the closest frame from timestamp column is acquired. 
    Args:
        data : pandas.DataFrame
            minian timestamps from preprocessing; argument set in load_and_align_minian function
        time : list
            regularly spaced time vector the expected length of the session; argument set in load_and_align_minian function
        plot_frame_usage : boolean
            if True, creates a plot of frame usage; by default set to False
    Returns:
        lined_up_timeframes : list
            vector of lined up frames to use to align recording to the time vector.
    """
    arg_mins = [np.abs(df['timestamp'] - (t)).argmin() for t in time]
    lined_up_timeframes = np.array(df.loc[arg_mins, 'frame_number'])
    ## Plot frame usage
    if plot_frame_usage:
        duplicated_timeframes = np.unique(lined_up_timeframes, return_counts=True)
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=duplicated_timeframes[0],
                y=duplicated_timeframes[1],
                mode="lines+markers",
                marker_size=5,
            )
        )
        fig.update_layout(
            template="simple_white",
            xaxis_title="Time (ms)",
            yaxis_title="Frequency",
            title_text="Number of times each behavior frame was re-used",
        )
        fig.show(config={"scrollZoom": True})
    ## Return data frame
    return lined_up_timeframes


def load_and_align_behavior(path, mouse, date, session = '20min', sampling_rate = 1/15, downsample = False, downsample_factor = 2, plot_frame_usage = False):
    """
    Args:
        path : str
            directory path
        mouse : str
            name of the mouse (e.g. 'mc01')
        date : str
            date of session
        session : str
            one of ['20min', '30min']; by default '20min'
        sampling_rate : float
            sampling rate of behavior; by default 1/15 (15 frames per second) because calcium imaging data is downsampled to 15fps
        downsample : boolean
            determines whether or not you want to downsample the data; by default False
        downsample_factor : float
            what factor you want to downsample your data by; by default 2
        plot_frame_usage : boolean
            creates a plot showing the number of times a frame was used; by default False
    Returns:
        aligned_behavior: pandas.DataFrame
            pd.DataFrame with the columns timestamp, x_pos, y_pos, a_pos, and frame_number
    """
    # create time vector based on expected length of session
    if session == '20min':
        frame_count = (
            20 * 60 / sampling_rate
        )  # 20min x 60sec/min / sampling_rate 
    elif session == '30min':
        frame_count = (
            30 * 60 / sampling_rate
        )  # 30min x 60sec/min / sampling_rate
    else:
        raise Exception(
            "Invalid 'session' argument. Must be one of: ['20min', '30min']"
        )
    time = np.arange(0, frame_count * sampling_rate, sampling_rate)
    # load the specified type of neural activity
    rpath = pjoin(path, 'Data/{}/{}/'.format(mouse, date))
    timestamp = os.listdir(rpath) ## get timestamp associated with that day
    rpath = pjoin(rpath, timestamp[0]) 
    ## Read circle_track.csv
    data = pd.read_csv(pjoin(rpath, 'circle_track.csv'))
    ## Crop data
    cropped_data = crop_data(data)
    ## Normalize timestamp
    norm_data = normalize_timestamp(cropped_data)
    ## Get location
    location = norm_data[(norm_data.event == 'LOCATION') | (norm_data.event == 'START')]
    location = location.reset_index(drop = True)
    ## Set START to next frame's x,y,a value
    location.loc[0, 'data'] = location.loc[1, 'data']
    ## Initialize empty dictionary
    location_data = {'timestamp': [], 'x_pos': [], 'y_pos': [], 'a_pos': []}
    for i in range(len(location)):
        location_data['timestamp'].append(location.loc[i, 'timestamp'])
        location_data['x_pos'].append(float(re.search(r'X([0-9]+)', location.loc[i, 'data']).group(1)))
        location_data['y_pos'].append(float(re.search(r'Y([0-9]+)', location.loc[i, 'data']).group(1)))
        location_data['a_pos'].append(float(re.search(r'A([0-9]+)', location.loc[i, 'data']).group(1)))
    ## Convert to pd.DataFrame
    df = pd.DataFrame(location_data)
    df['frame_number'] = range(0, len(df))
    ## Get aligned frames
    lined_up_timeframes = align_behavior_frames_legacy(df, time, plot_frame_usage = plot_frame_usage)
    ## Select aligned frames
    aligned_behavior = df.loc[lined_up_timeframes]
    ## Downsample if minian output is downsampled
    if downsample:
        aligned_behavior = aligned_behavior[::downsample_factor]
    return aligned_behavior
    


def import_mouse_behavior_data(path, mouse, key_file, session, plot_frame_usage = False):
    """
    Import all data for one mouse. Requires a yml file that contains session identifier keys.
    Args:
        path : str
            path to experiment directory
        mouse : str
            mouse name
        key_file : str
            name of yaml file that contains mouse as key and inner dictionary with context as key and date as value (e.g. A1 : 2022_06_08)
        session : str
            one of ['20min', '30min']
        plot_frame_usage : boolean
            plots how many times each frame was used in the aligned output; by default false
    Returns:
        sessions : dict
            key is context, value is xarray.DataArray from minian output
    """
    ## Initialize sessions
    sessions = {}
    ## Load keys
    key_path = pjoin(path, key_file)
    with open(key_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    ## Select keys for a specific mouse
    keys = data_loaded[mouse]
    swapped_keys = {y[0]: x for x, y in keys.items()}
    for date in swapped_keys:
        sessions[swapped_keys[date]] = load_and_align_behavior(path, mouse, date, session = session, plot_frame_usage = plot_frame_usage)
    return sessions


def linearize_trajectory(df, angle_type, shift_factor, inner_d = 25, outer_d = 30, unit = 'cm'):
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
    if unit == 'cm':
        ## convert inches to cm
        d1 = inner_d * 2.54
        d2 = outer_d * 2.54
    elif unit == 'in':
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


def bin_linearized_position(linearized_trajectory, angle_type = 'radians', bin_num = 8):
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
        binned = np.digitize(linearized_trajectory, bins)
    elif angle_type == 'degrees':
        bins = np.linspace(0, np.max(linearized_trajectory), (bin_num+1))
        binned = np.digitize(linearized_trajectory, bins)
    return binned


def get_trials(df, shift_factor, angle_type = 'radians', counterclockwise = True):
    """
    Labels timestamps with trail numbers.
    Args:
        df : pandas.DataFrame
        counterclockwise : boolean
            determines whether session was run with the mouse running counterclockwise; by default True
    """
    ## Linearize then bin position into one of 8 bins
    position = linearize_trajectory(df, shift_factor = shift_factor, angle_type = angle_type)
    binned_position = bin_linearized_position(position, angle_type = angle_type)
    bins = np.unique(binned_position)
    ## For each bin, get timestamps when the mouse was in that bin
    indices = [np.where(binned_position == this_bin)[0] for this_bin in bins]
    if counterclockwise:  ## reverse the order of the bins.
        indices = indices[::-1]
    ## Preallocate trial vector
    trials = np.full(binned_position.shape, np.nan)
    trial_start = 0
    last_idx = 0
    ## We need to loop through bins rather than simply looking for border crossings because a mouse can backtrack, which we wouldn't want to count.
    ## For a large number of trials...
    for trial_number in range(500):
        ## For each bin...
        for this_bin in indices:
            ## Find the first timestamp that comes after the first timestamp in the last bin for that trial. 
            ## Argmax is supposedly faster than np.where.
            last_idx = this_bin[np.argmax(this_bin > last_idx)]
        ## After looping through all the bins, remember the last timestamp where there was a bin transition.
        trial_end = last_idx
        ## If the slice still has all NaNs, label it with the trial number.
        if np.all(np.isnan(trials[trial_start:trial_end])):
            trials[trial_start:trial_end] = trial_number
        ## If not, finish up and exit the loop.
        else:
            trials[np.isnan(trials)] = trial_number - 1
            break
        ## The start of the next trial is the end of the last.
        trial_start = trial_end
    return trials


def forward_reverse_trials(aligned_behavior, trials, positive_jump = 350, wiggle = 2):
    """
    After determining number of trials, separate trials into trials in the forward (correct) direction or reverse (incorrect) direction.
    Args:
        aligned_behavior : pandas.DataFrame
            output from load_and_align_behavior function (aligns behavior timestamps to an evenly spaced time vector)
            contains x_pos, y_pos, and a_pos (angular position in degrees)
        trials : numpy.ndarray
            array of every timestamp labeled as a trial number; output from get_trials function
        positive_jump : int
            The linearized position has a large jump when the angular position resets, so we want our difference 
            in angular position to be less than this positive jump
        wiggle : int
            Since there isn't any temporal smoothing, it is possible for small jumps in angular position in the 
            incorrect direction due to noise. We want our difference between successive angular positions to
            be greater than this noise.
    Returns:
        forward_trials, backward_trials : list
            list of trials that are in the forward (correct) direction or reverse (incorrect) direction
    """
    forward_trials = []
    backward_trials = []
    for trial in np.unique(trials):
        ## Take the difference between each angular position within a given trial to determine direction
        if type(aligned_behavior) == xr.core.dataarray.DataArray:
            diff = aligned_behavior.a_pos[trials == trial].diff(dim='frame')
        else:
            diff = aligned_behavior.a_pos[trials == trial].diff()
        ## If there are NOT any difference values above the wiggle value (noise) and below the positive jump, include as forward
        if not any(diff[(diff > wiggle) & (diff < positive_jump)]):
            forward_trials.append(trial)
        else:
            backward_trials.append(trial)
    return forward_trials, backward_trials


def calculate_trial_length(aligned_behavior, angle_type='radians', shift_factor=0, forward_reverse=False, recalc_trials=False):
    """
    Calculate the number of seconds that occurs in each trial.
    Args:
        aligned_behavior : pandas DataFrame
            output from aligning behavior
        angle_type : str
            what angle type you want the linearize_position to use
        shift_factor : float
            how to shift the zero position
        forward_reverse : boolean
            will separate trials into forward and reverse trials
        recalc_trials : boolean
            will use function get_trials to determine trial structure - not needed if trials already exist from preprocessing
    Returns:
        time_diff : list
            list of times corresponding to the time difference between the beginning and end of the trial
        time_diff_forward, time_diff_reverse : list
            list of times corresponding to the time difference between the beginning and end of the trial separated by forward and reverse trials
    """
    ## Calculate trials, label each frame as a specific trial
    if recalc_trials:
        trials = get_trials(aligned_behavior, shift_factor = shift_factor, angle_type = angle_type, counterclockwise = True)
    else:
        trials = aligned_behavior['trials']
    if forward_reverse:
        ## Calculate forward and reverse trials
        forward_trials, reverse_trials = forward_reverse_trials(aligned_behavior, trials, positive_jump = 350, wiggle = 2)
        time_diff_forward = []
        time_diff_reverse = []
        ## Loop through forward trials
        for f_trial in forward_trials:
            behavior = aligned_behavior.loc[trials == f_trial]
            ## Get the first and last timestamp to determine the window
            first_timestamp, last_timestamp = behavior['t'].to_numpy()[0], behavior['t'].to_numpy()[-1]
            time_diff_forward.append(last_timestamp - first_timestamp)
        ## Loop through reverse trials
        for r_trial in reverse_trials:
            behavior = aligned_behavior.loc[trials == r_trial]
            ## Get the first and last timestamp to determine the window
            first_timestamp, last_timestamp = behavior['t'].to_numpy()[0], behavior['t'].to_numpy()[-1]
            time_diff_reverse.append(last_timestamp - first_timestamp)
        ## Comebine for both to do histogram plotting
        time_diff = time_diff_forward + time_diff_reverse
        return time_diff_forward, time_diff_reverse
    else:
        time_diff = []
        for trial in np.unique(trials):
            ## Subset aligned_behavior by a given trial
            behavior = aligned_behavior.loc[trials == trial]
            ## Get the first and last timestamp to determine the window
            first_timestamp, last_timestamp = behavior['t'].to_numpy()[0], behavior['t'].to_numpy()[-1]
            time_diff.append(last_timestamp - first_timestamp)
        return time_diff


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


### Adding new functions here - these functions definitely work on preprocessed behavior
def dprime_metrics(data, reward_one, reward_two, reward_index='one', forward_reverse='all', **kwargs):
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
    # data['lick_port'] = data['lick_port'].astype(str)
    ## Create nonreward list
    if reward_index == 'zero':
        nonreward_list = [x for x in np.arange(0, 8)]
    elif reward_index == 'one':
        nonreward_list = [x for x in np.arange(1, 9)]
    nonreward_list.remove(reward_one)
    nonreward_list.remove(reward_two)
    signal = {'hits': [], 'miss': [], 'FA': [], 'CR': [], 'dprime': []}
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
        go_trials = 2 ## two rewarding ports
        nogo_trials = 6 ## 8 ports minus 2 rewarding ports
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
        signal['hits'].append(hit_rate)
        signal['miss'].append(miss_rate)
        signal['FA'].append(FA_rate)
        signal['CR'].append(CR_rate)
        signal['dprime'].append(norm.ppf(hit_for_dprime) - norm.ppf(FA_for_dprime))
    return signal


def aggregate_metrics(signal, bin_size = 5):
    """
    Bins your hits, misses, FA, CR, and dprime according to bin_size.
    """
    aggregated_data = {'hits': [], 'miss': [], 'FA': [], 'CR': [], 'dprime': []}
    for key in signal:
        bins = np.arange(0, len(signal[key]), bin_size)
        binned = np.split(signal[key], bins)
        avg_value = [np.nanmean(bin) for bin in binned if bin.size > 0] ## if bin.size > 0 removes any empty array in binned
        aggregated_data[key] = avg_value
    return aggregated_data


def trial_averages(mouse_trial_times, session_list, forward = False):
    """
    Calculate the average trial time for a session across mice.
    """
    if not forward:
        trial_times_df = pd.DataFrame(mouse_trial_times).T
        avg_times = {}
        for session in session_list:
            times = trial_times_df[session].apply(pd.Series)
            avg_times[session] = np.nanmean(times, axis = 0)
    else:
        trial_times_df = pd.DataFrame(mouse_trial_times).T
        avg_times = {}
        for session in session_list:
            times = trial_times_df[session].apply(pd.Series)
            avg_times[session] = np.nanmean(times, axis = 0)
    return avg_times

def calculate_trial_times(aligned_behavior, forward_trials):
    trial_length_dict = {'trial': [], 'trial_length': []}
    for trial in forward_trials:
        behavior = aligned_behavior[aligned_behavior.trial == trial]
        first, last = behavior.timestamp.to_numpy()[0], behavior.timestamp.to_numpy()[-1]
        trial_length = last - first
        trial_length_dict['trial'].append(trial)
        trial_length_dict['trial_length'].append(trial_length)
    return trial_length_dict


def probe_lick_accuracy(df, port_one, port_two):
    """
    Used to calculate percent correct during the probe. Doesn't count only first licks, counts all licks.
    Args:
        df : pandas.DataFrame
            preprocessed behavior output stored as feather file
        port_one, port_two : str
            name of rewarding ports
    Returns:
        percent_correct : float
            lick accuracy for that session
    """
    licks = df[df['lick_port'] != -1].reset_index(drop=True)
    count_licks = licks.groupby('lick_port', as_index=False).agg(licks=('lick_port', 'count'))
    percent_correct = ((count_licks['licks'][(count_licks['lick_port'] == port_one) | (count_licks['lick_port'] == port_two)].dropna().sum()) / 
                        count_licks['licks'].dropna().sum()) * 100
    return percent_correct


def lick_accuracy(df, port_one, port_two, by_trials=False):
    """
    Used to calculate the first lick percent correct.
    Args:
        df : pandas.DataFrame
            preprocessed behavior containing columns for trials, lick_ports
        port_one, port_two : int
            which ports were rewarded (e.g. 5)
        by_trials : boolean
            if True, will calculate the first lick percent correct within a trial. By default False.
    Returns:
        percent_correct : float or list
            returns a single value when not calculated on a trial by trial basis
    """
    if by_trials:
        percent_correct = []
        for trial in np.unique(df['trials']):
                trial_behav = df[df['trials'] == trial]
                licks = trial_behav[trial_behav['lick_port'] != -1].reset_index(drop=True)
                licks.insert(loc=6, 
                            column='shifted', 
                            value=licks['lick_port'].shift(periods=1, fill_value='first'))
                licks.insert(loc=7, 
                            column='first', 
                            value=licks['lick_port'] != licks['shifted'])
                licks_df = licks[licks['first']].reset_index(drop=True)
                count_licks = licks_df.groupby('lick_port', as_index=False).agg(first_licks=('lick_port', 'count'))
                percent_correct.append(((count_licks['first_licks'][(count_licks['lick_port'] == port_one) | (count_licks['lick_port'] == port_two)].dropna().sum()) / 
                                        count_licks['first_licks'].dropna().sum()) * 100)
    else:
        licks = df[df['lick_port'] != -1].reset_index(drop=True)
        licks.insert(loc=6, 
                    column='shifted', 
                    value=licks['lick_port'].shift(periods=1, fill_value='first'))
        licks.insert(loc=7, 
                    column='first', 
                    value=licks['lick_port'] != licks['shifted'])
        licks_df = licks[licks['first']].reset_index(drop=True)
        count_licks = licks_df.groupby('lick_port', as_index=False).agg(first_licks=('lick_port', 'count'))
        percent_correct = ((count_licks['first_licks'][(count_licks['lick_port'] == port_one) | (count_licks['lick_port'] == port_two)].dropna().sum()) / 
                            count_licks['first_licks'].dropna().sum()) * 100
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


def get_forward_reverse_trials(aligned_behavior, positive_jump = 350, wiggle = 2):
    """
    After determining number of trials, separate trials into trials in the forward (correct) direction or reverse (incorrect) direction.
    Args:
        aligned_behavior : pandas.DataFrame
            output from load_and_align_behavior function (aligns behavior timestamps to an evenly spaced time vector)
            contains x_pos, y_pos, and a_pos (angular position in degrees)
        positive_jump : int
            The linearized position has a large jump when the angular position resets, so we want our difference 
            in angular position to be less than this positive jump
        wiggle : int
            Since there isn't any temporal smoothing, it is possible for small jumps in angular position in the 
            incorrect direction due to noise. We want our difference between successive angular positions to
            be greater than this noise.
    Returns:
        forward_trials, backward_trials : list
            list of trials that are in the forward (correct) direction or reverse (incorrect) direction
    """
    forward_trials = []
    backward_trials = []
    for trial in np.unique(aligned_behavior['trials']):
        ## Take the difference between each angular position within a given trial to determine direction
        diff = aligned_behavior.a_pos[aligned_behavior['trials'] == trial].diff()
        ## If there are NOT any difference values above the wiggle value (noise) and below the positive jump, include as forward
        if not any(diff[(diff > wiggle) & (diff < positive_jump)]):
            forward_trials.append(trial)
        else:
            backward_trials.append(trial)
    return forward_trials, backward_trials


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


def days_to_criteria(lick_df, mouse, criteria_val):
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
    sub_df = lick_df[(lick_df['mouse'] == mouse) & (lick_df['percent_correct'] >= criteria_val)].reset_index(drop=True)
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










