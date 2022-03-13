# -*- coding: utf-8 -*-
"""
Created on in 2022

@author: Group A - Advanced Python - IE MBD
Hussain Alhajjaj  
Ali Alqawaeen
Mohammed Aljamed
Hadi Alsinan 
Khalid Nasser
Maryam Alblushi
Durrah Alzamil
Basil Alfakher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

from pandas.api.types import is_numeric_dtype, is_object_dtype
import numpy as np
import plotly.offline as py
import plotly.figure_factory as ff
import plotly.graph_objs as gobj

import custom_helpers

class MankeyDataframe(pd.DataFrame):
    """
    The class is used to extend the properties of Dataframes through providing 
    statistical based and robust functions. 
    It provides the end user with both general and specific cleaning functions
    
    It facilitates the End User to perform some Date Feature Engineering,
    Scaling, Encoding, etc. to avoid code repetition.
    """

    #Initializing the inherited pd.DataFrame
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
    
    @property
    def _constructor(self):
        def func_(*args,**kwargs):
            df = MankeyDataframe(*args,**kwargs)
            return df
        return func_
    
#-----------------------------------------------------------------------------
                        # DATA HANDLING
#-----------------------------------------------------------------------------

    def SetAttributes(self, kwargs):
        """
        The function will update the type of the variable submitted for change.
        It will veify first that the key is present in the desired dataframe.
        If present, it will try to change the type to the desired format.
        If not possible, it will continue to the next element.         
        Parameters
        ----------
        **kwargs : The key-argument pair of field-type relationship that
        wants to be updated.
        Returns
        -------
        None.
        """
        if self.shape[0] > 0:
            for key,vartype in kwargs.items():
                if key in self.columns:
                    try:
                        self[key] = self[key].astype(vartype)
                    except:
                        print("Undefined type {}".format(str(vartype)))
                else:
                    print("The dataframe does not contain variable {}.".format(str(key)))
        else:
            print("The dataframe has not yet been initialized")

#-----------------------------------------------------------------------------
                        # SUPERVISED - BINARY CLASSIFICATION - DATA CLEANING
#-----------------------------------------------------------------------------    

    def plot_categorical(self, df,col_name):
        data = df
        values_count = pd.DataFrame(data[col_name].value_counts())
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
                                font = {'color':'#FF27C1'})
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

    def plot_numerical(self, df, col_name):
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

    def plot_univariate(self, df, cols = []):
        if len(cols) == 0:
            data = df
        else:
            data = df[cols]
        for col in data.columns:
            if is_numeric_dtype(df[col]):
                self.plot_numerical(data, col)
            elif is_object_dtype(df[col]):
                self.plot_categorical(data,col)
    
    def cleaning_missing(self, input_vars=[] ):
        """
        TO BE IMPLEMENTED: data cleaning (provide methods for data scanning and cleaning, 
            for example: scan each column, indicating if droping or keeping the variable for 
            modelling and why, for the ones keeping indicates which cleaning / transformation 
            is recommended for the missing values and if scalling / dummy creation is recommended, 
            if not always inform that is not necessary);
        Returns
        -------
          A print with the analysis or new clean columns .

        """
        return "To be implemented."
    
 
    def recommended_transformation(self, X_train, y_train, X_test, y_test, input_vars=[],  target='', ordinal_var = [], woe_cat_threshold=5, print_only = True):
        """

        TO BE IMPLEMENTED: data preparation (for each column provide methods to perform
        transformations - for example: time calculation like age, days as customers, 
        days to due date, label encoding, imputation, standard scalling, dummy creation 
        or replacement of category value by its probability of default depending, justify 
        transformation depending of the variable type, or explain why transformation is 
        not necessary);

        e.g. of specifying ordinal categories
        dictionary as follows: {"feature_name": [list of all classes IN ORDER from low (1) to high(# of classes)]}
        
        

        Returns
        -------
          A print with the analysis or new transformed columns.                
        """
        result = {}
        remaining_vars = input_vars


        ### categorical

        
        #ordinal


        #WoE transformations
        woe_vars = []
        for feature in remaining_vars:
            if feature in ordinal_var:
                continue
            if self[feature].nunique() >= woe_cat_threshold:
                result[feature] = 'Weight of Evidence (WoE) transformation'
                woe_vars.append(feature)
                
        if(print_only):
            woe_categorical(self, X_train, y_train, X_test, y_test, input_vars=woe_vars, target_var = target)

        #one hot encoding

        return "To be implemented."
    
    def woe_categorical(self, X_train, y_train, X_test, y_test, input_vars=[], target_var = 'target', num_cat_threshold = 5):
        """
        TODO: This method gets a list of categorical columns and target variable, this calculates
        the WoE for each column and each class within it (fit), this can be used later to transform
        the test set

        """
        t_woe = custom_helpers.WoE_Transformer()
        t_woe.fit(X_train, y_train, target_var, input_vars)
        t_woe.transform(X_train,y_train)
        t_woe.tranform(X_test, y_test)
        

        return (X_train, X_test)
        

