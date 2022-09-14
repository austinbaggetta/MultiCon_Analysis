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


def create_pairwise_heatmap(data, index, column, value, graph, colorscale = 'Viridis', boundaries = [5, 10, 15, 20], 
                            line_width = 1.5, boundary_color = 'red', template = 'simple_white', width = 800, height = 800):
    """
    Used to create pairwise comparison heatmaps for all days.
    Args: 
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
    for boundary in boundaries:
        fig.add_vline(x=boundary+0.5, line_width=line_width, line_color=boundary_color, opacity=1)
        fig.add_hline(y=boundary+0.5, line_width=line_width, line_color=boundary_color, opacity=1)  
    ## Layout options
    fig.update_layout(template = template, width = width, height = height,
                      xaxis_title = 'Day', yaxis_title = 'Day')
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(dtick=1)
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