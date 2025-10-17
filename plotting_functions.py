import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from natsort import natsorted
import os

## Import custom functions
import circletrack_behavior as ctb


def custom_graph_template(x_title, y_title, template='simple_white', height=500, width=500, linewidth=1.5,
                          titles=[''], rows=1, columns=1, shared_y=False, shared_x=False, font_size=26, font_family='Arial', **kwargs):
    """
    Used to make a cohesive graph type. In most functions, these arguments are supplied through **kwargs.
    """
    fig = make_subplots(rows=rows, cols=columns, subplot_titles=titles, shared_yaxes=shared_y, **kwargs)
    fig.update_yaxes(title=y_title, linewidth=linewidth)
    fig.update_xaxes(title=x_title, linewidth=linewidth)
    fig.update_layout(title={
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.update_annotations(font_size=font_size+3)
    fig.update_layout(template=template, height=height, width=width, font=dict(size=font_size), font_family=font_family)
    if shared_x:
        fig.update_xaxes(matches='x')
    if shared_y:
        fig.update_yaxes(matches='y')
    return fig


def create_pairwise_heatmap(data, index, column, value, graph, colorscale='Viridis', boundaries=None, 
                            line_width=1.5, boundary_color='red', **kwargs):
    """
    Used to create pairwise comparison heatmaps for all days.
    Args: 
        data : pandas.DataFrame
        index : str
            name of first column you want in your pivot table
        column : str
            name of second column to pivot by
        value : str
            name of column you want as your values for your pivot table
        graph : str
            one of ['overlap', 'activity']
        boundaries : list
            list containing days that mark ending of each context, e.g. [5, 10, 15]
    Returns:
        fig : plotly object
    """
    ## Create heatmap matrix
    matrix = data.pivot_table(index=index, columns=column, values=value)
    matrix = matrix.sort_values(by=index)
    matrix = matrix.sort_values(by=column, axis=1)
    ## Create figure
    fig = custom_graph_template(**kwargs)
    fig.add_trace(go.Heatmap(z=matrix.values,
                             x=matrix.index,
                             y=matrix.columns,
                             colorscale=colorscale))
    ## Loop through context boundaries and add red line
    if boundaries is not None:
        for boundary in boundaries:
            fig.add_vline(x=boundary+0.5, line_width=line_width, line_color=boundary_color, opacity=1)
            fig.add_hline(y=boundary+0.5, line_width=line_width, line_color=boundary_color, opacity=1)  
    ## Based on what you chose to graph, set title and legend_title
    if graph == 'overlap':
        fig.update_layout(title = {'text': 'Cell Overlap Between Days',
                                   'xanchor': 'center',
                                   'y': 0.9,
                                   'x': 0.5})
        fig.data[0].colorbar.title = 'Percent Overlap'
    elif graph == 'activity':
        fig.update_layout(title = {'text': 'Correlated Activity Between Days',
                                   'xanchor': 'center',
                                   'y': 0.9,
                                   'x': 0.5})
        fig.data[0].colorbar.title = 'r value'
    else:
        raise Exception("Incorrect graph argument! Must be one of ['overlap', 'activity']")
    return fig


def plot_behavior_across_days(data, x_var, y_var, groupby_var=['day'], avg_color='turquoise', chance_color='darkgrey', transition_color=['darkgrey'],
                               marker_color='rgb(179,179,179)', symbols='circle', plot_datapoints=True, expert_line=True, chance=True, datapoint_type='lines',
                               plot_transitions=[5.5, 10.5, 15.5], **kwargs):
    """
    Creates a line plot of behavior variable of interest (rewards, percent_correct, etc.) over all days.
    Includes individual subjects plotted over average.
    Args:
        data : pandas.DataFrame
            output from circletrack_behavior functions
        behavior_output : str
            behavior variable of interest (e.g. 'total_rewards')
        marker_color : str
            color of individual subject points; by default grey
        template : str
            plot template; by default simple_white
    Returns:
        fig : plotly.graph_objs._figure.Figure
            figure     
    """
    ## Calculate mean for each day
    avg_data = data.groupby(groupby_var, as_index=False).agg({y_var: 'mean'})
    ## Calculate SEM for each day
    sem_data = data.groupby(groupby_var, as_index=False).agg({y_var: 'sem'})
    ## Create figure
    fig = custom_graph_template(**kwargs)
    ## Colors for each group
    if ('group' in groupby_var) | ('group_two' in groupby_var) | ('sex' in groupby_var) | ('maze' in groupby_var) | ('experimenter' in groupby_var):
        groups = np.unique(data[groupby_var[-1]])
        group_dict = {g:c for (g,c) in zip(groups, marker_color)}
        symbol_dict = {g:s for (g,s) in zip(groups, symbols)}
        
    if plot_datapoints:
        ## Plot individual subjects
        for subject in np.unique(data['mouse']):
            data_sub = data.loc[data['mouse'] == subject].reset_index()
            if ('group' in groupby_var) | ('group_two' in groupby_var) | ('sex' in groupby_var) | ('maze' in groupby_var) | ('experimenter' in groupby_var):
                subject_color = group_dict[data_sub.loc[0, groupby_var[-1]]]
                legendgroup = data_sub.loc[0, groupby_var[-1]]
            else:
                subject_color = marker_color
                legendgroup = marker_color
            fig.add_trace(go.Scatter(x=data_sub[x_var], y=data_sub[y_var], showlegend=False, legendgroup=legendgroup,
                                    mode=datapoint_type, opacity=0.5, name=subject, line_color=subject_color, line_width=1))
    ## Plot group average or multiple group averages
    if ('group' in groupby_var) | ('group_two' in groupby_var) | ('sex' in groupby_var) | ('maze' in groupby_var) | ('experimenter' in groupby_var):
        for group in np.unique(avg_data[groupby_var[-1]]):
            avg_sub = avg_data.loc[avg_data[groupby_var[-1]] == group]
            sem_sub = sem_data.loc[sem_data[groupby_var[-1]] == group]
            fig.add_trace(go.Scatter(x=avg_sub[x_var], y=avg_sub[y_var],
                                     mode='lines+markers', marker_symbol=symbol_dict[group], marker_size=6.5,
                                     error_y = dict(type='data', array=sem_sub[y_var]), legendgroup=group, 
                                     line=dict(color=group_dict[group]), name=group, showlegend=True))
    else:
        fig.add_trace(go.Scatter(x=avg_data[x_var], y=avg_data[y_var],
                                mode='lines+markers', 
                                error_y=dict(type='data', array=sem_data[y_var]), marker_size=6.5,
                                line=dict(color=avg_color), showlegend=False, marker_symbol=symbols))
    ## Add dashed lines   
    if expert_line:
        fig.add_hline(y=75, line_width=1, line_dash='dash', line_color=chance_color, opacity=1)
    if chance:
        fig.add_hline(y=25, line_width=1, line_dash='dash', line_color=chance_color, opacity=1)
    ## Plot transitions
    if plot_transitions is not None:
        for idx, value in enumerate(plot_transitions):
            fig.add_vline(x=value, line_width=1, line_dash='dash', line_color=transition_color[idx], opacity=1)
    return fig


def plot_across_groups(agg_data, groupby, separateby, plot_var, colors, title, datapoint_var='mouse', y_range=None, plot_datapoints=False, plot_datalines=False,
    y_title='', text_size=15, opacity=0.8, plot_width=600, plot_height=600, tick_angle=45, scale_y=True, h_spacing=0.05, save_path=None, plot_scale=5):
    """
    Plot multiple bar plots (one for each unique type in 'separateby') with multiple bars on each plot (one bar for each unique type in 'groupby')
    """
    # Note that the separating variable must be of type pd.Categorical(ordered=True), such that its unique values can be sorted
    # This can be done by: agg_data[separateby] = pd.Categorical(agg_data[separateby], categories=['list','of','unique','values'], ordered=True)
    subplot_titles = agg_data[separateby].unique()
    fig = make_subplots(
        cols=len(subplot_titles), subplot_titles=subplot_titles, horizontal_spacing=h_spacing, shared_yaxes=scale_y
    )
    for i, val in enumerate(agg_data[separateby].unique()):
        sub_data = agg_data[agg_data[separateby] == val]
        means = sub_data[[groupby, plot_var]].groupby(groupby).mean()[plot_var].sort_index()
        sems = sub_data[[groupby, plot_var]].groupby(groupby).sem()[plot_var].sort_index()
        xlabels = means.index.values
        fig.add_trace(
            go.Bar(
                x=xlabels,
                y=means[xlabels].values,
                error_y=dict(type="data", array=sems.values, visible=True),
                marker_color=colors,
                marker=dict(line=dict(width=1, color="black"), opacity=opacity),
            ),
            row=1,
            col=i + 1,
        )
        if plot_datapoints:
            for point in sub_data[datapoint_var].unique():
                point_data = sub_data[sub_data[datapoint_var] == point]
                fig.add_trace(
                    go.Scattergl(
                        x=point_data[groupby].values,
                        y=point_data[plot_var].values,
                        mode="markers",
                        marker=dict(color="black", opacity=0.4),
                        name=str(point),
                    ),
                    row=1,
                    col=i + 1,
                )
        if plot_datalines:
            for line in sub_data[datapoint_var].unique():
                line_data = sub_data[sub_data[datapoint_var] == line]
                line_data = line_data.iloc[line_data[groupby].argsort(),:]
                fig.add_trace(
                    go.Scatter(
                        x=line_data[groupby].values,
                        y=line_data[plot_var].values,
                        mode="lines+markers",
                        line=dict(width=1),
                        marker=dict(color="black", opacity=0.4),
                        name=str(line),
                    ),
                    row=1,
                    col=i + 1,
                )
    fig.add_hline(y=0, row=1, col='all', line_width=1, opacity=1, line_color='black')
    fig.update_layout(
        dragmode="pan",
        yaxis_title=y_title,
        font=dict(size=text_size),
        title_text=title,
        autosize=False,
        width=plot_width,
        height=plot_height,
        template="simple_white",
        showlegend=False,
    )
    if tick_angle is not None:
        fig.update_xaxes(tickangle=tick_angle)
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        fig.write_image(save_path, format='png', scale=plot_scale)
    config = {
        'scrollZoom':True,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'custom_image',
            'height': plot_height,
            'width': plot_width,
            'scale':plot_scale
            }
            }
    fig.show(config=config)


def plot_cell_contribution(assemblies, colorscale='Viridis', **kwargs):
    """
    Creates a heatmap of ensemble by cell, where the color indicates how strongly a cell weighs into an ensemble.
    Args:
        assemblies : dict
            output from find_assemblies function; dictionary has a key ['patterns']
    """
    fig = custom_graph_template(**kwargs)
    fig.add_trace(go.Heatmap(z=assemblies['patterns'],
                            coloraxis='coloraxis'))
    fig.update_layout(coloraxis={'colorscale': colorscale})
    fig.update_layout(coloraxis_colorbar={'title': 'Weight'})
    return fig
    # fig.show(config={'scrollZoom':True})


def stem_plot(pattern, baseline=0, plot_members=True, member_color='blue', nonmem_color='black', 
              hline_color='black', size=3, opacity=0.5, **kwargs):
    """
    Create a stem plot, where the marker indicates the weight of that neuron in this ensemble. Members of the ensemble are plotted in a separate color.
    Args:
        pattern : numpy.ndarray
            An ensemble determined by the find_assemblies function. Obtained by subsetting assemblies['patterns'] (e.g. assemblies['patterns'][0])
        baseline : int
            Determines where a solid line will be drawn to indicate the baseline of the ensemble; by default set at 0
        plot_members : boolean
            If True, will color members of the ensemble a different color; membership determined by their weight being 2 standard deviations above/below the mean
            By default True
    Returns:
        fig
    """
    ## Get number of neurons
    n_neurons = len(pattern)
    loc = []
    head = []
    ## Get x position (loc) and y value (head)
    for i in np.arange(0, n_neurons):
        loc.append(i)
        head.append(pattern[i])
    temp_dict = {'location': loc, 'head': head}
    ## Create a boolean list where True means that neuron is a member of this ensemble
    ## Includes both positive and negative values 2 SD above/below the mean
    if plot_members:
        ensemble_cutoff_pos = (pattern.mean() + (pattern.std() * 2))
        ensemble_cutoff_neg = (pattern.mean() - (pattern.std() * 2))
        participants = (pattern > ensemble_cutoff_pos) | (pattern < ensemble_cutoff_neg)
        temp_dict['participants'] = participants
    else:
        participants = np.repeat(False, n_neurons)
        temp_dict['participants'] = participants
    ## Create dataframe
    df = pd.DataFrame(temp_dict)
    ## Plot figure
    fig = custom_graph_template(**kwargs)
    for neuron in np.arange(0, n_neurons):
        if df.participants[neuron] == True:
            fig.add_trace(go.Scatter(x=[df['location'][neuron], df['location'][neuron]], y=[0, df['head'][neuron]], mode='lines+markers', 
                          line=dict(width=1, color=member_color), marker=dict(color=member_color), opacity=opacity, showlegend=False))
        else:
            fig.add_trace(go.Scatter(x=[df['location'][neuron], df['location'][neuron]], y=[0, df['head'][neuron]], mode='lines+markers', 
                          line=dict(width=1, color=nonmem_color), marker=dict(color=nonmem_color), opacity=opacity, showlegend=False))
    ## Change marker size, change template, add x/y titles, title, and horizontal line at zero
    fig.update_traces(marker=dict(size=size))
    fig.add_hline(y=baseline, line_dash='solid', opacity=1, line_width=1, line_color=hline_color)
    return fig


def plot_linearized_position(aligned_behavior, trials, forward_trials, reverse_trials, shift_factor=0, angle_type = 'radians', forward_trial_color = 'green', reverse_trial_color = 'orchid'):
    ## Get linearized position 
    linearized_position = ctb.linearize_trajectory(aligned_behavior, angle_type=angle_type, shift_factor=shift_factor)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = aligned_behavior.timestamp, y = linearized_position, line = dict(color = 'black'), showlegend = False))
    for forward in enumerate(forward_trials):
        if forward[0] == 0:
            fig.add_trace(go.Scatter(x = aligned_behavior.timestamp[trials == forward[1]], y = linearized_position[trials == forward[1]], line = dict(color = forward_trial_color), legendgroup = 'forward', name = 'Forward'))
        else:
            fig.add_trace(go.Scatter(x = aligned_behavior.timestamp[trials == forward[1]], y = linearized_position[trials == forward[1]], line = dict(color = forward_trial_color), legendgroup = 'forward', showlegend = False))
    for backward in enumerate(reverse_trials):
        if backward[0] == 0:
            fig.add_trace(go.Scatter(x = aligned_behavior.timestamp[trials == backward[1]], y = linearized_position[trials == backward[1]], line = dict(color = reverse_trial_color), legendgroup = 'backward', name = 'Reverse'))
        else:
            fig.add_trace(go.Scatter(x = aligned_behavior.timestamp[trials == backward[1]], y = linearized_position[trials == backward[1]], line = dict(color = reverse_trial_color), legendgroup = 'backward', showlegend = False))
    fig.update_layout(template = 'simple_white', showlegend = True)
    fig.update_layout(
        title={
            'text': 'Forward and Reverse Trials',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    fig.update_yaxes(title = 'Linearized Position')
    fig.update_xaxes(title = 'Time (s)')
    return fig


def plot_raster(data, bool_data, time, colorscale = 'gray_r', line_color = 'black', template = 'simple_white', 
                x_title = 'Time (ms)', y_title = 'Neuron Number', title = '', height = 500, width = 500):
    """
    Creates a raster plot.
    Args:
        data : boolean
            If you had binarized spikes/activation strength above a specifc value.
            Rows could be number of neurons, for example.
        time : list/np.array
            x axis will be whatever units this is in
    Returns:
        fig
    """
    mean = np.mean(data, axis = 0)
    fig = make_subplots(rows = 2, shared_xaxes = True, x_title = x_title, vertical_spacing = 0.05)
    fig.add_trace(go.Scatter(x = time, y = mean, mode = 'lines', line_color = line_color, showlegend = False), row = 1, col = 1)
    fig.add_trace(go.Heatmap(x = time, z = bool_data, colorscale = colorscale, showscale = False, showlegend = False), row = 2, col = 1)
    fig.update_layout(template = template, height = height, width = width)
    fig.update_layout(title={
        'text': title,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    return fig


def plot_activation_strength(activations, ensemble_number, sampling_rate=1/30, x_bin_size=None, marker_color='red', **kwargs):
    if x_bin_size is not None:
        act = ctb.bin_data(activations[ensemble_number], bin_size=x_bin_size)
        time_vector = np.arange(0, len(act), sampling_rate)
    else:
        act = activations[ensemble_number]
        time_vector = np.arange(0, activations.shape[1]*sampling_rate, sampling_rate)
    fig = custom_graph_template(**kwargs)
    fig.add_trace(go.Scatter(x=time_vector, y=act,
                            mode='markers', marker_color=marker_color, opacity=0.7))
    return fig


def plot_lick_raster(behav, symbol='square', plot_reward=True, lick_color='grey', reward_color='turquoise', **kwargs):
    fig = custom_graph_template(**kwargs)
    for i, trial in enumerate(np.unique(behav['trials'])):
        trial_df = pd.DataFrame()
        trial_df = behav.loc[behav['trials'] == trial]
        trial_df.insert(0, 'lick_bool', trial_df['lick_port'] != -1)
        fig.add_trace(go.Scatter(x=trial_df['lin_position'][trial_df['lick_bool']], y=(i + 1) * trial_df['lick_bool'][trial_df['lick_bool']], mode='markers', 
                                marker=dict(symbol = symbol, color = lick_color, opacity = 0.8, size = 5), showlegend=False))
        if plot_reward:
            fig.add_trace(go.Scatter(x=trial_df['lin_position'][trial_df['water']], y=(i + 1) * trial_df['water'][trial_df['water']], mode='markers',
                                     marker=dict(symbol = symbol, color = reward_color, opacity = 1, size = 3), showlegend=False))
    return fig


def plot_circle_position(behav, position_color='grey', position_size=4, lick_color='turquoise', lick_size=4, downsample_factor=15, plot_licks=True, **kwargs):
    """
    Plot the position of the mouse as x,y coordinates with the x,y coordinate of licks as a separate color.
    """
    try:
        behav.insert(0, 'lick_bool', behav['lick_port'] != -1)
    except:
        pass
    fig = custom_graph_template(**kwargs)
    fig.add_trace(go.Scatter(x=behav['x'][::downsample_factor], y=behav['y'][::downsample_factor], mode='markers', 
                            marker=dict(color = position_color, size = position_size), showlegend=False))
    if plot_licks:
        lick_x = behav['x'][behav['lick_bool']]
        lick_y = behav['y'][behav['lick_bool']]
        fig.add_trace(go.Scatter(x=lick_x, y=lick_y, mode='markers', marker=dict(color = lick_color, size = lick_size), name='Lick'))
    return fig


def plot_group_averages(df, y_col_name, x_col_name, group_color_dict, mouse_grouping_variables=['mouse', 'session', 'group'], 
                        avg_grouping_variables=['group', 'session'], flip_order=False, x_axis_order=None, **kwargs):
    """ 
    Plot averages for different groups of mice.
    Args:
        df : pandas.DataFrame
            dataframe of results
        y_col_name, x_col_name : str
            string of columns of interest
        mouse_grouping_variables, avg_grouping_variables : list
            list of column names to be grouped
        group_color_dict : dict
            dictionary of colors as {group: color} in alphabetical order
        x_axis_order : list
            specify the order of the x_axis
        kwargs are additional arguments for fig.update_layout()
    Returns:
        fig : plotly.graph_object
    """
    if mouse_grouping_variables is not None:
        mouse_grouped_data = df.groupby(mouse_grouping_variables, as_index=False).agg({y_col_name: 'mean'})
    else:
        mouse_grouped_data = df.copy()
    avg_df = df.groupby(avg_grouping_variables, as_index=False).agg({y_col_name: ['mean', 'sem']})

    fig = px.strip(mouse_grouped_data, x=x_col_name, y=y_col_name, color='group', hover_name='mouse',
                   color_discrete_sequence=list(group_color_dict.values())).update_traces(showlegend=False, opacity=0.8, marker_line_width=1)
    
    if flip_order:
        group_list = np.flip(np.unique(df['group']))
    else:
        group_list = np.unique(df['group'])

    for group in group_list:
        group_df = avg_df[avg_df['group'] == group]
        fig.add_trace(go.Bar(x=group_df[x_col_name], y=group_df[y_col_name]['mean'],
                             error_y=dict(type='data', array=group_df[y_col_name]['sem'], thickness=2.5, width=10),
                             marker_color=group_color_dict[group], marker_line_color='black', 
                             marker_line_width=2, name=group, opacity=0.8))
    
    fig.update_layout(**kwargs)
    if x_axis_order is not None:
        fig.update_xaxes(categoryorder='array', categoryarray=x_axis_order)
    return fig


def visualize_individual_cell(spike_data, calcium_data, cell_number, downsample_factor=20, show_rewards=True, spiking_threshold=None, **kwargs):
    """ 
    Used to visually inspect circle track location data as well as how well the S matrix recapitulates the C matrix.
    Args:
        spike_data, calcium_data : xarray.DataArray
            S and C matrix from Minian that has been preprocessed to align neural results with behavior.
        cell_number : int
            Cell you will visualize, based on indexing rather than choosing unit_id
        downsample_factor : int
            By how much you want to downsample the position data for visualization purposes.
        show_rewards : boolean
            By default True; adds markers to where the reward ports are on the circle.
    Returns:
        fig : plotly.graph_object
            Two panel figure of circle track position overlaid with cell firing positions in the first panel
            and cell firing (both calcium and approximated spiking) across the session on the right.
    """
    fig = custom_graph_template(**kwargs)
    cell_spike_data = spike_data[spike_data['unit_id'] == cell_number]
    cell_calc_data = calcium_data[calcium_data['unit_id'] == cell_number]
    ## Firing locations of cell
    fig.add_trace(go.Scatter(x=spike_data['x'].values[::downsample_factor], y=spike_data['y'].values[::downsample_factor], 
                        mode='markers', marker_color='darkgrey', showlegend=False), row=1, col=1)
    if show_rewards:
        fig.add_trace(go.Scatter(x=spike_data['x'].values[spike_data['water'].values], y=spike_data['y'].values[spike_data['water'].values], 
                                mode='markers', marker_color='black', showlegend=False, opacity=0.5), row=1, col=1)
    ## Calcium trace
    fig.add_trace(go.Scatter(x=calcium_data['behav_t'].values, y=cell_calc_data.values[0], mode='lines', line_color='darkgrey', showlegend=False), row=1, col=2)
    ## Approximated spiking of cell
    if spiking_threshold is not None:
        fig.add_trace(go.Scatter(x=spike_data['behav_t'].values, y=(cell_spike_data.values[0] > spiking_threshold).astype(int), 
                                 mode='lines', line_color='red', showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=spike_data['x'].values[cell_spike_data.values[0] > spiking_threshold], y=spike_data['y'].values[cell_spike_data.values[0] > spiking_threshold], 
                        mode='markers', marker_color='red', showlegend=False, opacity=0.4), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=spike_data['behav_t'].values, y=cell_spike_data.values[0], mode='lines', line_color='red', showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=spike_data['x'].values[cell_spike_data.values[0] > 0], y=spike_data['y'].values[cell_spike_data.values[0] > 0], 
                        mode='markers', marker_color='red', showlegend=False, opacity=0.4), row=1, col=1)
    fig.update_yaxes(title='Y', row=1, col=1)
    fig.update_yaxes(title='Amplitude (a.u.)', row=1, col=2)
    fig.update_xaxes(title='X', row=1, col=1)
    fig.update_xaxes(title='Time (s)', row=1, col=2)
    return fig


def plot_multiple_cells(calcium_data, xdata, start_cell, end_cell, shift_factor=4.6, **kwargs):
    """
    Plot multiple cell's calcium transients across time.
    Args:
        calcium_data : numpy.array
        start_cell, end_cell : int
            dictates what cells you will plot
        shift_factor : float
            how much to offset each calcium trace by in the plot
    Returns:
        fig : plotly.graph_object
    """
    fig = custom_graph_template(**kwargs)
    for cell in np.arange(start_cell, end_cell):
            fig.add_trace(go.Scattergl(x=xdata, y=calcium_data[cell]+(shift_factor*cell), showlegend=False))
    fig.update_layout(yaxis=dict(visible=True, showticklabels=False))
    return fig


def plot_spatial_footprints(A, unit_ids=None, threshold=0, base_color='gray', max_proj=None, showlegend=True, **kwargs):
    """
    Plots spatial footprints of cells, and can plot specific subpopulations on top of the max projection.
    Args:
        A : xarray.DataArray
            output from Minian, spatial footprints of cells
        unit_ids : list or np.array
            unit_ids of specific cells to plot on top of max projection, by default None
        threshold : int
            used for binarizing member spatial footprints
        base_color : str
            colorscale of max projection
        max_proj : xarray.DataArray
            output from Minian. If not included, max_proj will be calculated from the spatial footprints.
    Returns:
        plotly.graph_object
    """
    if max_proj is None:
        max_proj = A.max(axis=0)

    fig = custom_graph_template(**kwargs)
    fig.add_trace(go.Heatmap(z=max_proj, colorscale=base_color, showscale=False))

    if unit_ids is not None:
        sub_max = A.sel({'unit_id': unit_ids}).max(axis=0)
        sub_max_bin = (sub_max > threshold).astype(int)
        sub_max_bin = sub_max_bin.where(sub_max_bin.values == 1, np.nan)
        fig.add_trace(go.Heatmap(z=sub_max_bin, colorscale='Viridis', showscale=False, opacity=0.6, name='Spatial Footprints', showlegend=showlegend))
    return fig


def plot_ensemble_0(bin_edges, rasters, ensemble_id, normalized=True, reward_positions=None, **kwargs):
    """
    Plots the activation strength within a bin across trials. Used after the ica.make_ensemble_raster function.
    Args:
        bin_edges : numpy.ndarray
        rasters : numpy.ndarray
            output from ica.make_ensemble_raster function
        ensemble_id : int
            used to index rasters to plot your ensemble of interest
        normalized : boolean
            if True, will divide all values by the max activation value; by default True
        reward_positions : list
            list of reward position values
        **kwargs
            additional arguments for custom_graph_template
    Returns:
        fig : plotly.graph_object
    """
    fig = custom_graph_template(**kwargs)
    if normalized:
        plot_data = rasters[ensemble_id] / np.max(rasters[ensemble_id])
    else:
        plot_data = rasters[ensemble_id]
    fig.add_trace(go.Heatmap(x=bin_edges, z=plot_data))

    if reward_positions is not None:
        fig.add_vline(x=reward_positions[0], line_width=1, line_color='darkgrey', opacity=1)
        fig.add_vline(x=reward_positions[1], line_width=1, line_color='darkgrey', opacity=1)
    return fig


def create_polar_plots(fig_data, bar_color, data_col='r', error_col=None, **kwargs):
    fig = custom_graph_template(**kwargs)

    for idx, session in enumerate(natsorted(np.unique(fig_data['session']))):
            fig.add_trace(go.Barpolar(
                r=fig_data[data_col]['mean'][fig_data['session'] == session],
                theta=fig_data['rotation'][fig_data['session'] == session],
                marker_color=bar_color,
                marker_line_color='black',
                marker_line_width=1,
                opacity=0.8,
                showlegend=False
            ), row=1, col=idx + 1)
            if error_col is not None:
                fig.add_trace(go.Barpolar(
                        r=fig_data[data_col]['mean'][fig_data['session'] == session] + fig_data['r'][error_col][fig_data['session'] == session],
                        theta=fig_data['rotation'][fig_data['session'] == session],
                        marker_color=bar_color,
                        marker_line_color='black',
                        marker_line_width=0,
                        opacity=0.3,
                        showlegend=False,
                        name='upper'
                ), row=1, col=idx + 1)
                fig.add_trace(go.Barpolar(
                        r=fig_data[data_col]['mean'][fig_data['session'] == session] - fig_data['r'][error_col][fig_data['session'] == session],
                        theta=fig_data['rotation'][fig_data['session'] == session],
                        marker_color=bar_color,
                        marker_line_color='black',
                        marker_line_width=0,
                        opacity=0.3,
                        showlegend=False,
                        name='lower'
                ), row=1, col=idx + 1)
        
    fig.update_polars(
            radialaxis = dict(showticklabels=False, ticks='', angle=180, tickangle=-180, range=[-0.2, 1], showgrid=True, gridcolor='darkgrey', dtick=0.2),
            angularaxis = dict(showticklabels=True, ticks='', showgrid=True, gridcolor='darkgrey'),
            barmode='overlay'
        )
    return fig


def preprocessed_plots(data_out, angle_type='degrees', position_color='darkgrey', save_path=None):
    """
    Used to visualize accuracy of preprocessing raw circle track data.
    Args:
        data_out : pandas.DataFrame
            dataframe with (at least) columns: t, a_pos, lin_position, water, lick_port, trials, correct_dir
        angle_type : str
            degrees or radians
        save_path : str
            by default None; specify path to save html graphic
    Returns:
        fig : plotly.graph_object
    """
    fig = custom_graph_template(x_title='Time (s)', y_title=f'Position ({angle_type})', width=1000, height=1000, 
                                rows=3, shared_y=True, shared_x=True)

    if angle_type == 'degrees':
        lin_pos = data_out['a_pos']
    else:
        lin_pos = data_out['lin_position']

    ## Plot licks and rewards on top of linearized position in the first row
    fig.add_trace(go.Scattergl(x=data_out['t'], y=lin_pos, mode='markers', marker_color=position_color, marker_size=2, name='Position'), row=1, col=1)
    fig.add_trace(go.Scattergl(x=data_out['t'][data_out['water']], y=lin_pos[data_out['water']], 
                            mode='markers', marker_color='red', marker_size=5, name='Rewards'), row=1, col=1)
    fig.add_trace(go.Scattergl(x=data_out['t'][data_out['lick_port'] != -1], y=lin_pos[data_out['lick_port'] != -1],
                            mode='markers', marker_color='black', marker_size=4, opacity=0.3, name='Licks'), row=1, col=1)

    ## Plot trials
    fig.add_trace(go.Scatter(x=data_out['t'], y=lin_pos, mode='markers', marker_color=position_color, marker_size=2, name='Position'), row=2, col=1)
    for trial in data_out['trials'].unique():
        tbool = data_out['trials'] == trial
        fig.add_trace(go.Scatter(x=data_out['t'][tbool], y=lin_pos[tbool], mode='markers', marker_size=2, 
                                legendgroup='trials', name=f'Trial {trial}', showlegend=False), row=2, col=1)

    ## Plot data points in the correct direction
    fig.add_trace(go.Scattergl(x=data_out['t'], y=lin_pos, mode='markers', marker_color=position_color, marker_size=2, name='Position'), row=3, col=1)
    fig.add_trace(go.Scattergl(x=data_out['t'][data_out['correct_dir']], y=lin_pos[data_out['correct_dir']], 
                            mode='markers', marker_size=2, marker_color='midnightblue', name='Correct Direction'), row=3, col=1)

    fig['data'][3]['showlegend'] = True
    fig['data'][3]['name'] = 'Trials'

    if save_path is not None:
        fig.write_html(os.path.join(save_path, f"{data_out['animal'].unique()[0]}_{data_out['session_two'].unique()[0]}_preprocessed.html"))
    return fig