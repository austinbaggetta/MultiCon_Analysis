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