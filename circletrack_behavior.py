import pandas as pd
import numpy as np
import os
import re
import glob

## Get list of circle_track.csv files by setting rpath
## Example rpath: r'/media/caishuman/csstorage3/Austin/CircleTrack/LargeScale/MultiCon_Verification/Data/**/**/**/*.csv'
def get_file_list(rpath):
    ## Create list of files
    file_list = glob.glob(rpath)
    return file_list

## Used to extract mouse name from file string
## Use a for loop to extract mouse name from every file in file_list
## Example str2match: r'(mcv[0-9]+)'
def get_mouse(file_list, str2match):
    mouseID = re.search(str2match, file_list)
    return mouseID.group()

## Combine the whole file_list with mouse_list
## Combines both into a data frame with filepath and mouse column names
def combine(file_list, mouse_list):
    files = pd.DataFrame(file_list, columns = ['filepath'])
    mice = pd.DataFrame(mouse_list, columns = ['mouse'])
    combined = pd.concat([files, mice], axis = 1)
    return combined

## Subsets combined file_list and mouse_list based on mouse string
def subset_combined(data, mouse):
    files = data.loc[data.mouse == mouse]
    file_list = files.filepath
    return file_list

## Crop data from START to TERMINATE
def crop_data(data):
    ## Get data from START to TERMINATE
    if any(data.event == 'TERMINATE'):
        df = data[np.where(data['event'] == 'START')[0][0]:np.where(data['event'] == 'TERMINATE')[0][0]+1]
    else:
        df = data[np.where(data['event'] == 'START')[0][0]:]
    return df

## Get rewarding ports
def get_rewarding_ports(data):
    ## In more recent circle track experiments, initialization was added to easily get rewarding ports
    ## Previous experiments require assessing from rewarding licks
    if any(data.event == 'initializing'):
        reward_ports = data.loc[data.event == 'initializing'].reset_index(drop = True)
        reward_first = reward_ports.data[0]
        reward_second = reward_ports.data[1]
        return(reward_first, reward_second)
    else:
        all_rewards = data.data.loc[data.event == 'REWARD'].reset_index(drop = True)
        reward_ports = pd.DataFrame(all_rewards.unique(), columns = ['data'])
        reward_first = reward_ports.iloc[0, 0]
        reward_second = reward_ports.iloc[1, 0]
        return reward_first, reward_second

## Get licks
def get_licks(data):
    lick_tmp = data.loc[(data.event == 'LICK') | (data.event == 'REWARD')].replace(to_replace='REWARD', value='LICK').reset_index(drop = True)
    return lick_tmp

## Normalize timestamp to get seconds
def normalize_timestamp(data):
    time = float(data.timestamp.loc[data.event == 'START'])
    data.timestamp = (data.timestamp - time)
    return data

## Calculate lick accuracy
def calculate_lick_accuracy(lick_tmp, reward_first, reward_second):
    ## lick_tmp is a data frame from the output of the get_licks function
    ## reward_first and reward_second can either be manually input, or can be determined from get_rewarding_ports function
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

## Calculate total rewards for a mouse across all of that mouse's sessions
## Can be combined with a for loop to get entire cohort's total rewards
def get_total_rewards(file_list, mouse):
    ## Subset file_list based on mouse
    files = subset_combined(file_list, mouse)
    ## Initialize empty data frame
    reward_data = {'total_rewards': [], 'mouse': [], 'day': []}
    ## Loop through files
    DayID = 1
    for path in files:
        data = pd.read_csv(path)
        ## Crop data
        cropped_data = crop_data(data)
        ## Get rewards
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

## Calculate first lick accuracy
## Selects the first lick in a bout of licks, calculates the number of first licks per rewarding port
## Subsequently divides the sum of the first licks at the two rewarding ports by the total number of first licks
def get_lick_accuracy(file_list, mouse):
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

## Get x, y, and angular position of the mouse across the sessions
def get_location_data(file_list, mouse):
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

## Determine direction based on angular position
def get_direction_information(location_data, lagn = 15):
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

## Calculate percentage of time spent in the correct direction
## Takes location_data input after get_direction_information was used
def direction_percentage(location_data):
    direction_percent = len(location_data.loc[location_data.direction == 'correct']) / (len(location_data.loc[location_data.direction == 'correct']) + len(location_data.loc[location_data.direction == 'wrong'])) * 100
    return direction_percent