import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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


def plot_behavior_across_days(data, behavior_var, groupby_var = 'day', marker_color = 'rgb(179,179,179)', template = 'simple_white'):
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