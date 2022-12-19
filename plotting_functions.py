import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_pairwise_heatmap(data, index, column, value, graph, colorscale = 'Viridis', boundaries = None, 
                            line_width = 1.5, boundary_color = 'red', template = 'simple_white', width = 800, height = 800):
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
    matrix = data.pivot_table(index = index, columns = column, values = value)
    matrix = matrix.sort_values(by = index)
    matrix = matrix.sort_values(by = column, axis = 1)
    ## Create figure
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z = matrix.values,
                             x = matrix.index,
                             y = matrix.columns,
                             colorscale = colorscale))
    ## Loop through context boundaries and add red line
    if boundaries is not None:
        for boundary in boundaries:
            fig.add_vline(x=boundary+0.5, line_width=line_width, line_color=boundary_color, opacity=1)
            fig.add_hline(y=boundary+0.5, line_width=line_width, line_color=boundary_color, opacity=1)  
    ## Layout options
    fig.update_layout(template = template, width = width, height = height,
                      xaxis_title = 'Day', yaxis_title = 'Day')
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


def plot_behavior_across_days(data, behavior_var, groupby_var = 'day', marker_color = 'rgb(179,179,179)', template = 'simple_white', plot_datapoints = True):
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
    avg_data = data.groupby([groupby_var]).mean().reset_index()
    ## Calculate SEM for each day
    sem_data = data.groupby([groupby_var]).sem().reset_index()
    ## Create figure
    fig = go.Figure()
    if plot_datapoints:
        ## Plot individual subjects
        fig.add_trace(go.Scatter(x = data[groupby_var], y = data[behavior_var],
                                mode = 'markers', opacity = 0.5,
                                marker = dict(color = marker_color, line = dict(width = 1))))
    ## Plot group average
    fig.add_trace(go.Scatter(x = avg_data[groupby_var], y = avg_data[behavior_var],
                             mode = 'lines+markers',
                             error_y = dict(type = 'data', array = sem_data[behavior_var]),
                             line = dict(color = 'rgb(172,78,163)')))
    fig.update_layout(template = template, xaxis_title = 'Day')
    fig.update_layout(showlegend = False)
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


def plot_cell_contribution(assemblies, colorscale = 'Viridis', template = 'simple_white', height = 800, width = 800):
    """
    Creates a heatmap of ensemble by cell, where the color indicates how strongly a cell weighs into an ensemble.
    Args:
        assemblies : dict
            output from find_assemblies function; dictionary has a key ['patterns']
    """
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z = assemblies['patterns'],
                            coloraxis = 'coloraxis'))
    fig.update_layout(template = template,
                      title = {'text': 'Cell Contribution to Each Ensemble',
                            'xanchor': 'center',
                            'y' : 0.9,
                            'x' : 0.5}, 
                      xaxis_title = 'Cell Number',
                      yaxis_title = 'Ensemble Number',
                      coloraxis = {'colorscale': colorscale},
                      width = width,
                      height = height)
    fig.update_layout(coloraxis_colorbar = {'title': 'Weight'})
    fig.show(config={'scrollZoom':True})


def stem_plot(pattern, baseline = 0, plot_members = True, member_color = 'blue', title = '', nonmem_color = 'black', hline_color = 'black', 
              size = 2, opacity = 0.5, height = 800, width = 800, template = 'simple_white'):
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
    fig = go.Figure()
    for neuron in np.arange(0, n_neurons):
        if df.participants[neuron] == True:
            fig.add_trace(go.Scatter(x = [df['location'][neuron], df['location'][neuron]], y = [0, df['head'][neuron]], mode = 'lines+markers', 
                          line=dict(width=1, color=member_color), marker=dict(color=member_color), opacity = opacity))
        else:
            fig.add_trace(go.Scatter(x = [df['location'][neuron], df['location'][neuron]], y = [0, df['head'][neuron]], mode = 'lines+markers', 
                          line=dict(width=1, color=nonmem_color), marker=dict(color=nonmem_color), opacity = opacity))
    ## Change marker size, change template, add x/y titles, title, and horizontal line at zero
    fig.update_traces(marker=dict(size = size))
    fig.update_layout(template = template, height = height, width = width, showlegend = False, xaxis_title = 'Neuron', yaxis_title = 'Weight')
    fig.update_layout(title = {'text': title,
                            'xanchor': 'center',
                            'y' : 0.9,
                            'x' : 0.5})
    fig.add_hline(y = baseline, line_dash = 'solid', opacity = 1, line_width = 1, line_color = hline_color)
    return fig