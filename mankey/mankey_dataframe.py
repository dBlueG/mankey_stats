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

from os import stat
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
from sklearn.model_selection import train_test_split

import custom_helpers
import stats_helpers
import charting_helper

class MankeyDataFrame(pd.DataFrame):
    """
    The class is used to extend the properties of Dataframes through providing 
    statistical based and robust functions. 
    It provides the end user with both general and specific cleaning functions
    
    It facilitates the End User to perform some Date Feature Engineering,
    Scaling, Encoding, etc. to avoid code repetition.

    Example instantiation:

    import pandas as pd
    csv_list = pd.read_csv("list_stuff.csv")
    mdf = MankeyDataFrame(csv_list)

    """

    #Initializing the inherited pd.DataFrame
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
    
    @property
    def _constructor(self):
        def func_(*args,**kwargs):
            df = MankeyDataFrame(*args,**kwargs)
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



    def plot_charts(self, input_vars = [], force_categorical=['BINARIZED_TARGET']):
        charting_helper.plot_univariate(self, cols = input_vars, force_categorical=force_categorical)
    
    def explore_stat(self, input_vars=[], normal_test_alpha=0.05, grubbs_alpha=0.05, grubbs_max_outliers=20, iqr_factor=2.5):
        """
        This method describes the dataset per feature, for numeric fields several statistics are 
        calculated including skew, kurtosis, normality test, missing value, and quartiles.
        This also includes grubbs test for outliers

        Parameters
        ----------
        input_vars: list of columns (to limit the evaluation to certain features only)
        normal_test_alpha: alpha value to be used for the normality test
        grubbs_alpha: alpha value for the grubbs outlier test
        grubbs_max_outliers: how many outliers to look for
        iqr_factor: IQR factor to determine outlier

        Returns
        -------
        None.
        """
        df_analyze = self.copy()
        if(input_vars):
            df_analyze = df_analyze.loc[:,input_vars]
        
        return stats_helpers.eval_df(df_analyze)

    def create_train_test_sets(self, target_var, input_vars = [], test_size = 0.2, random_state = 42):
        """
        Divide the dataframe into two sets (training and test) to be used to ML models
        
        Parameters
        ----------
        input_vars: list of columns (to limit the sets to only certain fields)
        target_var: name of the feature containing the ML target (to be used for y)
        Returns
        -------
          Training and Test data sets including X and y (target) 
          (X_train, y_train, X_test, y_test)

        """

        X = self.loc[:, self.columns != target_var]
        y = self[target_var]

        if(input_vars):
            X = X.loc[:,input_vars]
        
        

        # split into input and output elements
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size, 
                                                            random_state=random_state) 
        
        return (X_train, X_test, y_train, y_test)

    
    def cleaning_missing(self, X_train, y_train, X_test, y_test, target_var, input_vars=[], print_only = True ):
        """
        Using the statistical tests from the explore_stat method, this method performs handling of 
        missing data imputation and/or removal of the feature (as per user preferences).
        The method can be run in simulation mode or active mode, in simulation, only the recommended transformations
        will be highlighted but the data is not modified.
        
        Parameters
        ----------
        input_vars: list of columns (to limit the evaluation to certain features only)
        print_only: only print an analysis of the findings

        Returns
        -------
          A print with the analysis or new clean columns .

        """
        df_train = X_train.copy()
        df_train[target_var] = y_train

        df_test = X_test.copy()
        df_test[target_var] = y_test

        if(print_only):
           
            (descriptive_statistics, logic_prep, _) = stats_helpers.eval_df(df_train)
            print("Training set")
            print(logic_prep)
            print("===============")
            print(descriptive_statistics)

            (descriptive_statistics, _, _) = stats_helpers.eval_df(df_test)
            print("===============")
            print("Test set stats")
            print(descriptive_statistics)

        else:
            return stats_helpers.clean_data(df_train, df_test)

        
    
 
    def recommended_transformation(self, X_train, y_train, X_test, y_test, target_var, input_vars=[],  ordinal_var = {}, woe_cat_threshold=5, print_only = True):
        """
        This method can be used to recommend and/or apply transformations including:
        impute missing values, date manipulations, categorical variable handling (dummy/WoE/ordinal).

        The method needs two sets (training and test) so that all transformation settings are based
        on the training set but applied on the both sets.

        Note: the method can be used to also print the transformations, without actually outputting new data
       
        e.g. of specifying ordinal categories
        dictionary as follows: {"feature_name": [list of all classes IN ORDER from low (1) to high(# of classes)]}
        
        Parameters
        ----------
        input_vars: list of columns (to limit the evaluation to certain features only)
        X_train: subset to be used for training
        X_test: subset to be used for test
        y_train: target output related to X_train
        y_test: target output related to X_test
        target_var: name of the feature containing the ML target (to be used for y)
        woe_cat_threshold: the number of classes per category to determine whether WoE should be used for transformation
        ordinal_var: dictionary for specifying ordinal categories as follows: {"feature_name": [list of all classes IN ORDER from low (1) to high(# of classes)]}
        print_only: only print an analysis of the findings

        Returns
        -------
          A print with the analysis or new transformed columns.                
        """
        result = {}
        remaining_vars = input_vars


        # categorical

        
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
        

