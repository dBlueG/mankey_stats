import numpy as np
import plotly.offline as py
import plotly.figure_factory as ff
import plotly.graph_objs as gobj
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_datetime64_dtype
def plot_categorical(df,col_name,unit = 'Y'):
    data = df
    values_count = pd.DataFrame(data[col_name].value_counts())
    if is_datetime64_dtype(df[col_name]):
        values_count = pd.DataFrame(pd.DataFrame(formate_date(data,col_name,unit)).value_counts())
    values_count.columns = ['count']
    values_count[col_name] = [ str(i) for i in values_count.index ]
    values_count['percent'] = values_count['count'].div(values_count['count'].sum()).multiply(100).round(2)
    values_count = values_count.reindex([col_name,'count','percent'],axis=1)
    values_count.reset_index(drop=True,inplace=True)
    font_size = 10
    trace = gobj.Bar( x = values_count[col_name], y = values_count['count'], marker = {'color':'#FFAA00'})
    data_ = gobj.Data( [trace] )
    annotations0 = [ dict(x = xi,
                            y = yi,
                            showarrow=False,
                            font={'size':font_size},
                            text = "{:,}".format(yi),
                            xanchor='center',
                            yanchor='bottom' )
                       for xi,yi,_ in values_count.values ]
    annotations1 = [ dict( x = xi,
                            y = yi/2,
                            showarrow = False,
                            text = "{}%".format(pi),
                            xanchor = 'center',
                            yanchor = 'middle',
                            font = {'color':'#AA00AA'})
                         for xi,yi,pi in values_count.values if pi > 10 ]
    annotations = annotations0 + annotations1
    layout = gobj.Layout( title = col_name.replace('_',' ').capitalize(),
                            titlefont = {'size': 50},
                            yaxis = {'title':'count'},
                            xaxis = {'type':'category'},
                            annotations = annotations,
                            plot_bgcolor = '#FFF8EC')
    figure = gobj.Figure( data = data_, layout = layout )
    py.iplot(figure)
def plot_numerical(df, col_name):
    data = df
    series = data[col_name]
    smin,smax = series.min(),series.max()
    percentiles = [ np.percentile(series,n) for n in (2.5,50,97.5) ]
    trace0 = gobj.Histogram( x = series,
                            histfunc = 'avg',
                            histnorm = 'probability density',
                            opacity=.75,
                           marker = {'color':'#FFAA00'})
    data_ = gobj.Data( [trace0] )
    shapes = [{ 'line': { 'color': '#AA00AA', 'dash':'dot', 'width':2 },
                'type':'line',
                'x0':percentiles[0], 'x1':percentiles[0], 'xref':'x',
                'y0':-0.1, 'y1':1, 'yref':'paper' },
              { 'line': { 'color': '#AA00AA', 'dash':'dot', 'width':1 },
                'type':'line',
                'x0':percentiles[1], 'x1':percentiles[1], 'xref':'x',
                'y0':-0.1, 'y1':1, 'yref':'paper' },
              { 'line': { 'color': '#AA00AA', 'dash':'dot', 'width':2 },
                'type':'line',
                'x0':percentiles[2], 'x1':percentiles[2], 'xref':'x',
                'y0':-0.1, 'y1':1, 'yref':'paper' }
             ]
    annotations = [ {'x': percentiles[0], 'xref':'x','xanchor':'right',
                     'y': .3, 'yref':'paper',
                     'text':'2.5%', 'font':{'size':10},
                     'showarrow':False},
                    {'x': percentiles[1], 'xref':'x','xanchor':'center',
                     'y': .2, 'yref':'paper',
                     'text':'95%<br>median = {0:,.2f}<br>mean = {1:,.2f}<br>min = {2:,}<br>max = {3:,}'
                         .format(percentiles[1],series.mean(),smin,smax),
                     'showarrow':False,
                     'font':{'size':10} },
                    {'x': percentiles[2], 'xref':'x','xanchor':'left',
                     'y': .3, 'yref':'paper',
                     'text':'2.5%','font':{'size':10},
                     'showarrow':False}
                  ]
    layout = gobj.Layout( title = col_name,
                        yaxis = {'title':'Probability/Density'},
                        xaxis = {'title':col_name, 'type':'linear'},
                        shapes = shapes,
                        annotations = annotations,
                        plot_bgcolor = '#FFF8EC'
                         )
    figure = gobj.Figure(data = data_, layout = layout)
    py.iplot(figure)
def formate_date(df,col,unit = 'Y'):
    import datetime
    formated_date = []
    for value in df[col]:
        date = datetime.datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
        if unit == 'Y':
            formated_date.append(date.year)
        elif unit == 'M':
            formated_date.append(date.month)
        elif unit == 'D':
            formated_date.append(date.day)
    return formated_date
def plot_univariate(df, cols = [], force_categorical = []):
    if len(cols) == 0:
        data = df
    else:
        data = df[cols]
    for col in data.columns:
        if col in force_categorical:
            plot_categorical(data,col)
        elif is_numeric_dtype(df[col]):
            plot_numerical(data, col)
        elif (is_object_dtype(df[col])) or (is_datetime64_dtype(df[col])) :
            plot_categorical(data,col)