import pandas as pd
import numpy as np
import os
import re
import glob
from os.path import join as pjoin


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
        reward_ports = data.loc[data.event == 'initializing'].reset_index(drop = True)
        reward_first = reward_ports.data[0]
        reward_second = reward_ports.data[1]
        return reward_first, reward_second
    else:
        all_rewards = data.data.loc[data.event == 'REWARD'].reset_index(drop = True)
        reward_ports = pd.DataFrame(all_rewards.unique(), columns = ['data'])
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


def align_behavior_frames(data, time, plot_frame_usage=False):
    """
    Takes timestamps matrix associated with a miniscope recording and a regularly spaced time vector the
    expected length of the session. For each timeframe in 'time', the closest frame from minian_timestamps is acquired.
    Some frames are used more than once. Returns vector of lined up frames to use to align recording to the time vector.
    """
    lined_up_timeframes = np.array(
        [
            np.abs(minian_timestamps["Time Stamp (ms)"] - (t * 1000)).argmin()
            for t in time
        ]
    )

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
            title_text="Number of times each miniscope frame was re-used",
        )
        fig.show(config={"scrollZoom": True})

    return lined_up_timeframes


def load_and_align_behavior(dpath, mouse, date, timestamp, session = '20min', sampling_rate=1 / 15):
    """
    Parameters:
    ==========
    dpath : str
        directory path
    mouse : str
        name of the mouse (e.g. 'mc01')
    date : str
        date of session
    timestamp : str
        timestamp associated with session
    session : str
        one of ['20min'], may incorporate other options if length of sessions decrease from 20min
    """
    # create time vector based on expected length of session
    if "20min" in session:
        frame_count = (
            20 * 60 / sampling_rate
        )  # 20min x 60sec/min / sampling_rate (usually 1/15 because of temporal downsampling during Minian processing)
    else:
        raise Exception(
            "Invalid 'session' argument. Must be one of: ['20min']"
        )
    time = np.arange(0, frame_count * sampling_rate, sampling_rate)
    ## Load behavior data
    rpath = pjoin(dpath, '/Data/{}/{}/{}/circle_track.csv'.format(mouse, date, timestamp))
    mouse_behavior = pd.read_csv(rpath)
    
    
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
    time = float(data.timestamp.loc[data.event == 'START'])
    data.timestamp = (data.timestamp - time)
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


def get_location_data(file_list, mouse):
    """
    Get x, y,, and angular position of the mouse across all sessions.
    Args: 
        file_list : pandas.DataFrame
            combined file_list and mouse_list from combine function
        mouse : str
            mouse name (e.g. 'mc01')
    Returns:
        location_data : pandas.DataFrame
            pd.DataFrame with columns x_pos, y_pos, a_pos, mouse, day
    """
    ## Subset file_list based on mouse
    files = subset_combined(file_list, mouse)
    ## Initialize empty data frame
    location_data = {'x_pos': [], 'y_pos': [], 'a_pos': [], 'mouse': [], 'day': []}
    ## Loop through files
    DayID = 1
    for path in files:
        data = pd.read_csv(path)
        ## Crop data
        cropped_data = crop_data(data)
        ## Normalize timestamp
        norm_data = normalize_timestamp(cropped_data)
        ## Get location
        location = data.loc[data.event == 'LOCATION']
        ## append to dictionary
        for i in range(1, len(location)):
            location_data['x_pos'].append(float(re.search(r'X([0-9]+)', location.data.iloc[i]).group(1)))
            location_data['y_pos'].append(float(re.search(r'Y([0-9]+)', location.data.iloc[i]).group(1)))
            location_data['a_pos'].append(float(re.search(r'A([0-9]+)', location.data.iloc[i]).group(1)))
            location_data['mouse'].append(mouse)
            location_data['day'].append(DayID)
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
        direction_percent : float
            percent of time moving in correct direction
    """
    direction_percent = len(location_data.loc[location_data.direction == 'correct']) / (len(location_data.loc[location_data.direction == 'correct']) + len(location_data.loc[location_data.direction == 'wrong'])) * 100
    return direction_percent