library(tidyverse)
library(ggExtra)
library(gridExtra)
library(reshape2)
library(plotly)
library(ggpubr)
library(viridis)

get_file_list = function(rpath){
  ## Create list of files
  file_list <- list.files(path = rpath, 
                          recursive = T, 
                          full.names = T, 
                          pattern = "circle_track.csv")
  return(file_list)
}

get_mouse = function(file, str2match){
  ## Get mouse
  mouseID = str_match(file, str2match)[,2]
  return(mouseID)
}

combine = function(file_list, mouse_list){
  ## Convert to data frame
  file_list = as.data.frame(file_list)
  mouse_list = as.data.frame(mouse_list)
  combined = cbind(file_list, mouse_list)
  colnames(combined) = c('filepath', 'mouse')
  return(combined)
}

subset_combined = function(data, mouse){
  files = data[data$mouse == mouse,]
  files = files$filepath
  return(files)
}

crop_data = function(data){
  ## Get data from START to TERMINATE
  if (any(data$event == 'TERMINATE', na.rm = TRUE)){
    data = data[which(data$event == 'START'):which(data$event == 'TERMINATE'),]
  } else {
    data = data[which(data$event == 'START'):length(data$event),]
  }
}

## Get rewarding ports function
get_rewarding_ports = function(data){
  if (any(data$event == 'initializing')){
    reward_ports = data[data$event == 'initializing',]
    return(reward_ports)
  } else {
    all_rewards = data$data[data$event == "REWARD"]
    reward_ports = unique(all_rewards)
    reward_ports = as.data.frame(reward_ports)
    colnames(reward_ports) = 'data'
    return(reward_ports)
  }
}

## Below function is the body of the get_first_lick_percent function
calculate_percent_correct = function(lick_tmp){
  ## First lick ratio calculations
  lick_tmp$shifted = c(
    "first",
    lick_tmp$data[-(length(lick_tmp$data))])
  lick_tmp$first = lick_tmp$data != lick_tmp$shifted
  first = lick_tmp[lick_tmp$first,]
  licks = first %>%
    group_by(data) %>%
    summarize(
      N_licks = n()
    )
  all_rewards = lick_tmp$data[lick_tmp$event == "REWARD"]
  rewards = unique(all_rewards)
  reward_first = rewards[1]
  reward_second = rewards[2]
  licks$data[(licks$data == reward_first | licks$data == reward_second)] <- "c"
  Percent_Correct = sum(licks$N_licks[licks$data == "c"], na.rm = T) / 
    sum(licks$N_licks, na.rm = T)
  Percent_Correct = as.numeric(Percent_Correct * 100)
  return(Percent_Correct)
}

## First lick accuracy for heatmap creation
## Removes finding the rewarding ports within the function
calculate_percent_correct_heatmap = function(lick_tmp, reward_first, reward_second){
  ## First lick ratio calculations
  lick_tmp$shifted = c(
    "first",
    lick_tmp$data[-(length(lick_tmp$data))])
  lick_tmp$first = lick_tmp$data != lick_tmp$shifted
  first = lick_tmp[lick_tmp$first,]
  licks = first %>%
    group_by(data) %>%
    summarize(
      N_licks = n()
    )
  licks$data[(licks$data == reward_first | licks$data == reward_second)] <- "c"
  Percent_Correct = sum(licks$N_licks[licks$data == "c"], na.rm = T) / 
    sum(licks$N_licks, na.rm = T)
  Percent_Correct = as.numeric(Percent_Correct * 100)
  return(Percent_Correct)
}

## The below functions take in the combined outputs from above as the path argument
get_lick_info <- function(path, mouse){
  ## Subset combined file_list and mouse_list
  files = subset_combined(path, mouse)
  ## Initialize data frame
  lick_data <- data.frame()
  ## Loop through directory
  DayID = 1
  for (i in 1:length(files)){
      data = files[i] %>%
        map_df(~read_csv(.))
      ## Get data from START to TERMINATE
      data = crop_data(data)
      ## Normalize time stamps to start
      time = as.numeric(data$timestamp[data$event == 'START'])
      data$timestamp = as.numeric(data$timestamp) - time[1]
      ## Get licks and rewards
      lick_info = data[data$event == "LICK" | data$event == "REWARD",]
      lick_info$event[lick_info$event == "REWARD"] <- "LICK"
      ## Get mouse ID
      Mouse = mouse
      ## rbind outputs for licks
      Mouse_lick = rep(Mouse, times = nrow(lick_info))
      Day = rep(DayID, times = nrow(lick_info))
      Day = as.numeric(Day)
      tmp = cbind(Mouse_lick, Day, lick_info)
      lick_data <- rbind(lick_data, tmp)
      DayID = DayID + 1
    }
  return(lick_data)
}

get_total_rewards <- function(path, mouse){
  ## Subset combined file_list and mouse_list
  files = subset_combined(path, mouse)
  ## Initialize empty data frames
  reward_data <- data.frame()
  ## Loop through directory
  DayID = 1
  for (i in 1:length(files)){
      data = files[i] %>%
        map_df(~read_csv(.))
      ## Get data from START to TERMINATE
      data = crop_data(data)
      ## Get rewards and licks
      lick_tmp = data[data$event == "LICK" | data$event == "REWARD",]
      total_rewards = length(lick_tmp$event[lick_tmp$event == "REWARD"])
      total_rewards = as.numeric(total_rewards)
      ## Get mouse ID
      Mouse = mouse
      ## Get day
      Day = as.numeric(DayID)
      ## rbind outputs for rewards
      tmp = cbind(Mouse, Day, total_rewards)
      reward_data <- rbind(reward_data, tmp)
      DayID = DayID + 1
    }
  return(reward_data)
}

get_first_lick_percent <- function(path, mouse){
  ## Subset combined file_list and mouse_list
  files = subset_combined(path, mouse)
  ## Initialize empty data frames
  first_lick <- data.frame()
  ## Loop through directory
  DayID = 1
  for (i in 1:length(files)){
        data = files[i] %>%
          map_df(~read_csv(.))
        ## Get data from START to TERMINATE
        data = crop_data(data)
        ## Get licks and rewards
        lick_tmp = data[data$event == "LICK" | data$event == "REWARD",]
        ## First lick ratio calculations
        lick_tmp$shifted = c(
          "first",
          lick_tmp$data[-(length(lick_tmp$data))])
        lick_tmp$first = lick_tmp$data != lick_tmp$shifted
        first = lick_tmp[lick_tmp$first,]
        licks = first %>%
          group_by(data) %>%
          summarize(
            N_licks = n()
          )
        all_rewards = lick_tmp$data[lick_tmp$event == "REWARD"]
        rewards = unique(all_rewards)
        reward_first = rewards[1]
        reward_second = rewards[2]
        licks$data[(licks$data == reward_first | licks$data == reward_second)] <- "c"
        Percent_Correct = sum(licks$N_licks[licks$data == "c"], na.rm = T) / 
          sum(licks$N_licks, na.rm = T)
        Percent_Correct = as.numeric(Percent_Correct * 100)
        ## Get mouse ID
        Mouse = mouse
        Day = DayID
        ## rbind outputs for first licks
        tmp = cbind(Mouse, Day, Percent_Correct)
        first_lick <- rbind(first_lick, tmp)
        DayID = DayID + 1
    } 
  return(first_lick)
}

get_location_info <- function(path, mouse, lagn = 15){
  ## Subset combined file_list and mouse_list
  files = subset_combined(path, mouse)
  ## Initialize empty data frames
  location_data <- data.frame()
  ## Loop through directory
  DayID = 1
  for (i in 1:length(files)){
      data = files[i] %>%
        map_df(~read_csv(.))
      ## Get data from START to TERMINATE
      if (any(data$event == 'TERMINATE', na.rm = TRUE)){
        data = data[which(data$event == 'START'):which(data$event == 'TERMINATE'),]
      } else {
        data = data[which(data$event == 'START'):length(data$event),]
      }
      ## Normalize time stamps to start
      time = as.numeric(data$timestamp[data$event == 'START'])
      if (length(time) != 1){
        data$timestamp = as.numeric(data$timestamp) - time[1]
      } else if (length(time) == 1){
        data$timestamp = as.numeric(data$timestamp) - time
      }
      ## Get LOCATION, LICK, and REWARD
      location = data[data$event == 'LOCATION',]
      ## Split x, y, and angle
      location = location %>%
        mutate(
          x_pos = as.numeric(str_match(location$data, pattern = 'X(.*?)Y')[,2]),
          y_pos = as.numeric(str_match(location$data, pattern = 'Y(.*?)A')[,2]),
          a_pos = str_extract(location$data, pattern = 'A[0-9]*')
        )
      location$a_pos = str_remove(location$a_pos, 'A')
      location$a_pos = as.numeric(location$a_pos)
      ## Offset a_pos
      location$a_offset = c(rep(NA, times = lagn), diff(location$a_pos, lag = lagn))
      location$direction = case_when(location$a_offset < 0 ~ "Correct",
                                     location$a_offset > 0 ~ "Wrong",
                                     location$a_offset == 0 ~ "NA")
      ## Get mouse ID
      Mouse = mouse
      ## rbind outputs for location
      Mouse = rep(Mouse, times = nrow(location))
      Day = rep(DayID, times = nrow(location))
      Day = as.numeric(Day)
      tmp = cbind(Mouse, Day, location)
      location_data <- rbind(location_data, tmp)
      DayID = DayID + 1
    }
  return(location_data)
}

get_direction_prop <- function(location_data){
   direction_prop <- location_data %>%
      group_by(Day) %>%
      summarize(
        prop_corr_dir = sum(direction == 'Correct', na.rm = T) /
          (sum(direction == 'Correct', na.rm = T) + sum(direction == 'Wrong', na.rm = T))
      )
   direction_prop = direction_prop
  return(direction_prop)
}

get_probe_info <- function(path, mouse){
  ## Subset combined file_list and mouse_list
  files = subset_combined(path, mouse)
  ## Initialize empty data frames
  probe_data <- data.frame()
  ## Loop through directory
  DayID = 1
  for (i in 1:length(files)){
      data = files[i] %>%
        map_df(~read_csv(.))
      ## Get probe length
      probe = str_extract(data$event, 'probe length: [0-9]*')
      probe = probe[!is.na(probe)]
      probe = str_remove(probe, 'probe length: ')
      probe = as.numeric(probe)
      if (probe > 0){
        ## Get rewarded ports
        rewards = data[data$event == "REWARD",]
        ports = unique(rewards$data)
        reward_one = ports[1]
        reward_two = ports[2]
        ## Get data from START to TERMINATE
        if (any(data$event == 'TERMINATE', na.rm = TRUE)){
          data = data[which(data$event == 'START'):which(data$event == 'TERMINATE'),]
        } else {
          data = data[which(data$event == 'START'):length(data$event),]
        }
        ## Normalize time stamps to start
        time = as.numeric(data$timestamp[data$event == 'START'])
        if (length(time) != 1){
          data$timestamp = as.numeric(data$timestamp) - time[1]
        } else if (length(time) == 1){
          data$timestamp = as.numeric(data$timestamp) - time
        }
        ## Check licking behavior during probe
        data = data[data$timestamp < probe,]
        licks = data[data$event == 'LICK',]
        licks = licks[!is.na(licks$event),]
        ## Calculate lick percent correct during probe
        Percent_Correct = ((sum(licks$data == reward_one, na.rm = TRUE) + 
                              sum(licks$data == reward_two, na.rm = TRUE))
                             / length(licks$data))
        Percent_Correct = as.numeric(Percent_Correct * 100)
        ## Number of licks at rewarded vs non-rewarded ports
        rewarded_ports = (sum(licks$data == reward_one, na.rm = T) + 
                            sum(licks$data == reward_two, na.rm = T))
        total_licks = length(licks$data)
        nonrewarded_ports = total_licks - rewarded_ports
        ## Get mouse ID
        Mouse = mouse
        ## rbind outputs for location
        probe_length = probe
        Day = as.numeric(DayID)
        tmp = cbind(Mouse, Day, Percent_Correct, rewarded_ports, 
                    nonrewarded_ports, probe_length)
        probe_data = rbind(probe_data, tmp)
        DayID = DayID + 1
      } else {
        print('No probe!')
        Mouse = mouse
        Day = as.numeric(DayID)
        Percent_Correct = NA
        rewarded_ports = NA
        nonrewarded_ports = NA
        probe_length = 0
        tmp2 = cbind(Mouse, Day, Percent_Correct, rewarded_ports, 
                    nonrewarded_ports, probe_length)
        probe_data = rbind(probe_data, tmp2)
        DayID = DayID + 1
      }
    } 
  return(probe_data)
}

get_time_between_rewards <- function(path, mouse){
  ## Subset combined file_list and mouse_list
  files = subset_combined(path, mouse)
  ## Initialize empty data frames
  reward_time <- data.frame()
  ## Loop through directory
  DayID = 1
  for (i in 1:length(files)){
      data = files[i] %>%
        map_df(~read_csv(.))
      ## Get data from START to TERMINATE
      if (any(data$event == 'TERMINATE', na.rm = TRUE)){
        data <- data[which(data$event == 'START'):which(data$event == 'TERMINATE'),]
      } else {
        data <- data[which(data$event == 'START'):length(data$event),]
      }
      ## Get rewards
      rewards = data[data$event == "REWARD",]
      rewards$timestamp = as.numeric(rewards$timestamp)
      rewards$time_diff = c(NA, diff(rewards$timestamp))
      median_time = as.numeric(median(rewards$time_diff, na.rm = TRUE))
      ## Get mouse ID
      Mouse = mouse
      Day = as.numeric(DayID)
      ## rbind outputs for rewards
      tmp = cbind(Mouse, Day, median_time)
      reward_time <- rbind(reward_time, tmp)
      DayID = DayID + 1
    }
  return(reward_time)
}

set_track_and_maze = function(x, y){
  if (x == '4track'){
    reward_ports = data.frame('reward1' = c(90, 90, 90, 90), 
                              'reward2' = c(20, 25, 27, 45),
                              'reward3' = c(340, 347, 345, 0),
                              'reward4' = c(296, 305, 302, 315),
                              'reward5' = c(253, 265, 259, 270),
                              'reward6' = c(210, 220, 215, 225),
                              'reward7' = c(169, 178, 171, 180),
                              'reward8' = c(128, 135, 131, 135))
    row.names(reward_ports) = c('maze1', 'maze2', 'maze3', 'maze4')
  } else if (x == 'clear'){
    reward_ports = data.frame()
    row.names(reward_ports) = 'maze1'
  }
  ## Select reward ports based on maze number (for clear maze, maze == 'maze1')
  if (y == 'maze1'){
    reward_ports = reward_ports['maze1',]
  } else if (y == 'maze2'){
    reward_ports = reward_ports['maze2',]
  } else if (y == 'maze3'){
    reward_ports = reward_ports['maze3',]
  } else if (y == 'maze4'){
    reward_ports = reward_ports['maze4',]
  }
  return(reward_ports)
}

get_d_prime = function(path, mouse, track, maze, window = 20, lick_thresh,
                      port1, port2){
  ## Subset combined file_list and mouse_list
  files = subset_combined(path, mouse)
  ## Create empty data frame
  dprime_data = data.frame()
  ## Loop through files
  DayID = 1
  for (i in 1:length(files)){
    ## Import data
    data = files[i] %>%
      map_df(~read_csv(.))
    ## Get probe length
    probe = str_extract(data$event, 'probe length: [0-9]*')
    probe = probe[!is.na(probe)]
    probe = str_remove(probe, 'probe length: ')
    probe = as.numeric(probe)
    ## Create reward port positions based on apparatus
    reward_ports = set_track_and_maze(track, maze)
    ## Select port angular position
    port1_angle = reward_ports[,port1]
    port2_angle = reward_ports[,port2]
    ## Get data from START to TERMINATE
    if (any(data$event == 'TERMINATE', na.rm = TRUE)){
      data = data[which(data$event == 'START'):which(data$event == 'TERMINATE'),]
    } else {
      data = data[which(data$event == 'START'):length(data$event),]
    }
    ## Normalize time stamps to start
    data$timestamp = as.numeric(data$timestamp)
    time = as.numeric(data$timestamp[data$event == 'START'])
    if (length(time) != 1){
      data$timestamp = as.numeric(data$timestamp) - time[1]
    } else if (length(time) == 1){
      data$timestamp = as.numeric(data$timestamp) - time
    }
    data$timestamp = abs(data$timestamp)
    if (probe > 0){
      data = data[data$timestamp > probe,]
    }
    ## Change REWARD to LICK
    data$event[data$event == "REWARD"] <- "LICK"
    ## Get angular position
    data = as.data.frame(data)
    data$a_pos = case_when(data$event == 'LOCATION' 
                           ~ str_extract(data$data, pattern = 'A[0-9]*'),
                          data$event == 'LICK' ~ as.character(NA),
                          TRUE ~ as.character(NA))
    data$a_pos = str_remove(data$a_pos, 'A')
    data$a_pos = as.numeric(data$a_pos)
    ## When a_pos is between 181 and 359, subtract 360 to get rid of jumps from
    ## 360 to 0, also allows to measure port distances
    vec = 181:359
    for (i in length(data$a_pos[i])){
      if (is.na(data$a_pos[i])){
        data$a_pos[i] = NA
      } else if (any(data$a_pos[i] == vec)){
        data$a_pos[i] = 360 - data$a_pos[i]
      }
    }
    ## Offset a_pos
    ## Can change the lags here to change frame integration
    data$a_offset = c(0, diff(data$a_pos, lag = 1))
    data$direction = case_when(data$a_offset < 0 ~ 'Correct',
                                   data$a_offset > 0 ~ 'Wrong',
                                   data$a_offset == 0 ~ 'NA')
    ## Get ROIs
    data$A = case_when((port1_angle - (window/2) < data$a_pos & 
                            data$a_pos < port1_angle + (window/2)) ~ TRUE,
                         TRUE ~ FALSE)
    data$B = case_when((port2_angle - (window/2) < data$a_pos &
                          data$a_pos < port2_angle + (window/2)) ~ TRUE,
                       TRUE ~ FALSE)
    data$C = case_when((port1_angle - (window/2) < data$a_pos & 
                          data$a_pos < port1_angle + (window/2)) ~ FALSE,
                       (port2_angle - (window/2) < data$a_pos &
                          data$a_pos < port2_angle + (window/2)) ~ FALSE,
                       TRUE ~ TRUE)
    data$lick = case_when(data$event == 'LICK' ~ TRUE,
                          TRUE ~ FALSE)
    ## Get start and end positions of lick bouts
    data$lick_diff = c(0, diff(data$lick, lag = 1))
    start = which(data$lick_diff == 1)
    end = which(data$lick_diff == -1)
    if (length(end) == length(start)){
      bouts = data.frame(
        'start' = start,
        'end' = end
      )
    } else if (length(end) != length(start)){
      if (length(start) > length(end)){
        start = start[-length(start)]
      }
      else if (length(start) < length(end)){
        end = end[-length(end)]
      }
      bouts = data.frame(
        'start' = start,
        'end' = end
      )
    }
    ## Index data based on start and end positions, fill in a_pos of lick
    ## with the location a_pos the frame before
    ## Index data based on lick events, fill in direction column with direction
    ## the frame before
    lick_events = which(data$event == 'LICK')
    if (length(start) != 0){
      for (i in 1:length(bouts$start)){
        data[bouts$start[i]:bouts$end[i], 'a_pos'] = 
          rep(data$a_pos[bouts$start[i]-1], 
              times = nrow(data[bouts$start[i]:bouts$end[i]-1,]))
      }
      for (i in 1:length(lick_events)){
        data[lick_events[i], 'direction'] = data$direction[lick_events[i]-1]
      }
      data$direction[is.na(data$direction)] = 'NA'
      continue = TRUE
    } else {
      continue = FALSE ## this accounts for no licks in a zone later on
    }
    ## Get ROIs again
    data$A = case_when((port1_angle - (window/2) < data$a_pos & 
                          data$a_pos < port1_angle + (window/2)) ~ TRUE,
                       TRUE ~ FALSE)
    data$B = case_when((port2_angle - (window/2) < data$a_pos &
                          data$a_pos < port2_angle + (window/2)) ~ TRUE,
                       TRUE ~ FALSE)
    data$C = case_when((port1_angle - (window/2) < data$a_pos & 
                          data$a_pos < port1_angle + (window/2)) ~ FALSE,
                       (port2_angle - (window/2) < data$a_pos &
                          data$a_pos < port2_angle + (window/2)) ~ FALSE,
                       TRUE ~ TRUE)
    data$lick = case_when(data$event == 'LICK' ~ TRUE,
                          TRUE ~ FALSE)
    ## Get A diff
    data$A_diff = c(0, diff(data$A, lag = 1))
    start_a = which(data$A_diff == 1)
    end_a = which(data$A_diff == -1)
    if (length(end_a) == length(start_a)){
      bouts_a = data.frame(
        'bouts' = 1:length(start_a),
        'start' = start_a,
        'end' = end_a
      )
    } else if (length(end_a) != length(start_a)){
      if (length(start_a) > length(end_a)){
        start_a = start_a[-length(start_a)]
      }
      else if (length(start_a) < length(end_a)){
        end_a = end_a[-length(end_a)]
      }
      bouts_a = data.frame(
        'bouts' = 1:length(start_a),
        'start' = start_a,
        'end' = end_a
      )
    }
    ## Get B diff
    data$B_diff = c(0, diff(data$B, lag = 1))
    start_b = which(data$B_diff == 1)
    end_b = which(data$B_diff == -1)
    if (length(end_b) == length(start_b)){
      bouts_b = data.frame(
        'bouts' = 1:length(start_b),
        'start' = start_b,
        'end' = end_b
      )
    } else if (length(end_b) != length(start_b)){
      if (length(start_b) > length(end_b)){
        start_b = start_b[-length(start_b)]
      }
      else if (length(start_b) < length(end_b)){
        end_b = end_b[-length(end_b)]
      }
      bouts_b = data.frame(
        'bouts' = 1:length(start_b),
        'start' = start_b,
        'end' = end_b
      )
    }
    ## Get C diff (c is when the animal is not in the rewarding zones)
    data$C_diff = c(1, diff(data$C, lag = 1))
    start_c = which(data$C_diff == 1)
    end_c = which(data$C_diff == -1)
    ## Remove last frame (TERMINATE) since it's starting a new chunk
    start_c = start_c[-length(start_c)]
    if (length(end_c) == length(start_c)){
      bouts_c = data.frame(
        'bouts' = 1:length(c),
        'start' = start_c,
        'end' = end_c
      )
    } else if (length(end_c) != length(start_c)){
      if (length(start_c) > length(end_c)){
        start_c = start_c[-length(start_c)]
      }
      else if (length(start_c) < length(end_c)){
        end_c = end_c[-length(end_c)]
      }
      bouts_c = data.frame(
        'bouts' = 1:length(start_c),
        'start' = start_c,
        'end' = end_c
      )
    }
    all_bouts_inzone = rbind(bouts_a, bouts_b)
    ## Calculate hit rate
    if (continue == TRUE){
      hits = 0
      for (i in 1:nrow(bouts_a)){
        if (sum(data$lick[bouts_a$start[i]:bouts_a$end[i]]) >= lick_thresh){ 
          if (any(is.na(data$direction[bouts_a$start[i]:bouts_a$end[i]])) != T){
            if (any(str_detect(data$direction[bouts_a$start[i]:bouts_a$end[i]], 
                                    'Correct')) == T){
              hits = hits + 1
            } 
          } else
            hits = hits 
        }
      }
      for (i in 1:nrow(bouts_b)){
        if (sum(data$lick[bouts_b$start[i]:bouts_b$end[i]]) >= lick_thresh){
          if (any(is.na(data$direction[bouts_b$start[i]:bouts_b$end[i]])) != T){
            if (any(str_detect(data$direction[bouts_b$start[i]:bouts_b$end[i]], 
                               'Correct')) == T){
            hits = hits + 1
            } 
          } else
            hits = hits
        }
      }
      hits = hits + 0.5
      signal_trials = 0
      for (i in 1:nrow(all_bouts_inzone)){
        if (any(is.na(data$direction[all_bouts_inzone$start[i]
                                     :all_bouts_inzone$end[i]])) != T){
          if (any(str_detect(data$direction[all_bouts_inzone$start[i]
                                            :all_bouts_inzone$end[i]], 
                             'Correct')) == T){
          signal_trials = signal_trials + 1
          }
        } else {
          signal_trials = signal_trials
        } 
      }
      signal_trials = signal_trials + 1
      hit_rate = hits / signal_trials
      ## Calculate false alarm
      ## Added 0.5 to hits and false alarms and 1 to signal and noise trials 
      ## to account for +/- infinity (loglinear approach to calculate d')
      false_rejects = 0
      for (i in 1:nrow(bouts_c)){
        if (sum(data$lick[bouts_c$start[i]:bouts_c$end[i]]) >= lick_thresh){
          if (any(is.na(data$direction[bouts_c$start[i]:bouts_c$end[i]])) != T){
            if (any(str_detect(data$direction[bouts_c$start[i]:bouts_c$end[i]], 
                                    'Correct')) == T){
            false_rejects = false_rejects + 1
          } 
        } else
          false_rejects = false_rejects
        }
      }
      false_rejects = false_rejects + 0.5
      noise_trials = 0
      for (i in 1:nrow(bouts_c)){
        if (any(is.na(data$direction[bouts_c$start[i]:bouts_c$end[i]])) != T){
          if (any(str_detect(data$direction[bouts_c$start[i]:bouts_c$end[i]], 
                                  'Correct')) == T){
          noise_trials = noise_trials + 1
        } 
        } else {
          noise_trials = noise_trials + 1
        }
      }
      noise_trials = noise_trials + 1
      false_alarm = false_rejects / noise_trials
      ## Calculate d' 
      dprime = qnorm(hit_rate) - qnorm(false_alarm)
    } else {
      dprime = NA
    }
    ## Get mouseID
    Mouse = mouse
    Day = DayID
    ##cbind outputs
    tmp = cbind(Mouse, Day, dprime, hits, signal_trials, false_rejects, noise_trials)
    dprime_data = rbind(dprime_data, tmp)
    DayID = DayID + 1
  }
  return(dprime_data)
}

## Plot heatmap
plot_heatmap = function(heatmap_data, mouse, color){
  ## Color scale
  vals <- unique(scales::rescale(c(volcano)))
  o <- order(vals, decreasing = FALSE)
  cols <- scales::col_numeric(color, domain = NULL)(vals)
  colz <- setNames(data.frame(vals[o], cols[o]), NULL)
  ## x-axis settings
  x_axis = list(
    title = 'Time within Session'
  )
  ## y-axis settings
  y_axis = list(
    title = 'Day'
  )
  fig = plot_ly(x = heatmap_data$time_int, y = heatmap_data$DayID,
                z = heatmap_data$per_corr, colorscale = colz, type = 'heatmap',
                colorbar = list(title = 'Lick Accuracy')) %>%
    layout(title = mouse,
           xaxis = x_axis,
           yaxis = y_axis)
}

## Create a function to plot a heatmap where rows = day (session), column is binned time across
## session, and one heatmap is one mouse
create_heatmap = function(path, mouse, binwidth = 30, color = 'magma'){
  ## Subset combined file_list and mouse_list
  files = subset_combined(path, mouse)
  ## Initialize empty data frame
  heatmap_data = data.frame()
  ## Loop through directory
  DayID = 1
  for (i in 1:length(files)){
    data = files[i] %>%
      map_df(~read_csv(.))
    ## Get rewards
    reward_ports = get_rewarding_ports(data)
    reward_first = reward_ports$data[1]
    reward_second = reward_ports$data[2]
    ## Get data from START to TERMINATE
    data = crop_data(data)
    ## Normalize time stamps
    time = as.numeric(data$timestamp[data$event == 'START'])
    if (length(time) != 1){
      data$timestamp = as.numeric(data$timestamp) - time[1]
    } else if (length(time) == 1){
      data$timestamp = as.numeric(data$timestamp) - time
    }
    ## First lick ratio calculations across chunks of time
    ## Binwidth is in seconds
    chunks = (tail(data$timestamp, n = 1) / binwidth)
    percent_correct = data.frame(matrix(ncol=3, nrow=0, 
                                        dimnames = list(NULL, c('time_int', 'percent_correct', 'Day'))))
    for (i in 1:chunks){
      start = ((i-1)*binwidth)
      end = ((i)*binwidth)
      subset_data = subset(data, timestamp >= start & timestamp < end)
      ## Get licks and rewards
      lick_tmp = subset_data[subset_data$event == "LICK" | subset_data$event == "REWARD",]
      per_corr = calculate_percent_correct_heatmap(lick_tmp, reward_first, reward_second)
      time_int = start
      tmp = cbind(time_int, per_corr, DayID)
      percent_correct = rbind(percent_correct, tmp)
    }
    DayID = DayID + 1
    heatmap_data = rbind(heatmap_data, percent_correct)
  }
  fig = plot_heatmap(heatmap_data, mouse, color)
  return(fig)
}


## Post-hoc analysis Functions
## The below functions can be used to analyze the outputs from the functions above
## and are used to answer specific questions
get_delta = function(data, day1, day2, type = 'percent_correct'){
  if (type == 'percent_correct'){
    day_prior = data[data$Day == day1,]
    day_after = data[data$Day == day2,]
    delta = data.frame(Mouse = day_prior$Mouse, 
                       delta = day_prior$Percent_Correct - day_after$Percent_Correct)
    return(delta)
  } else if (type == 'rewards'){
    day_prior = data[data$Day == day1,]
    day_after = data[data$Day == day2,]
    delta = data.frame(Mouse = day_prior$Mouse, 
                       delta = day_prior$total_rewards - day_after$total_rewards)
    return(delta)
  }
}

get_avg_correct = function(data){
  avg_correct = data %>%
    group_by(Day) %>%
    summarize(
      avg_correct = mean(percent_correct, na.rm = TRUE),
      SD = sd(percent_correct, na.rm = TRUE),
      SEM = sd(percent_correct, na.rm = TRUE) / sqrt(length(unique(Mouse)))
    )
  return(avg_correct)
}

get_avg_rewards = function(data){
  avg_rewards = data %>%
    group_by(Day) %>%
    summarize(
      avg_rewards = mean(total_rewards, na.rm = TRUE),
      SD = sd(total_rewards, na.rm = TRUE),
      SEM = sd(total_rewards, na.rm = TRUE) / sqrt(length(unique(Mouse)))
    )
  return(avg_rewards)
}