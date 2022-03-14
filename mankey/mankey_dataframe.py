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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from pandas.api.types import is_numeric_dtype, is_object_dtype
import numpy as np
import plotly.offline as py
import plotly.figure_factory as ff
import plotly.graph_objs as gobj
from sklearn.model_selection import train_test_split

import custom_helpers as custom
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
            (df_clean, df_clean_test) = stats_helpers.clean_data(df_train, df_test)
            #X_train, y_train, X_test, y_test
            return df_clean.loc[:, df_clean.columns != target_var], df_clean[target_var], \
                   df_clean_test.loc[:, df_clean_test.columns != target_var], df_clean_test[target_var]

        
    
 
    def recommended_transformation(self, X_train, y_train, X_test, y_test, input_vars = [], ordinal_var = {}, woe_cat_threshold=5, print_only = True, expand_dates = 'all'
    , due_date = "EXPECTED_CLOSE_DATE", start_date = "LOAN_OPEN_DATE", target_var = 'BINARIZED_TARGET'):
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
        #target_var: name of the feature containing the ML target (to be used for y)
        woe_cat_threshold: the number of classes per category to determine whether WoE should be used for transformation
        ordinal_var: dictionary for specifying ordinal categories as follows: {"feature_name": [list of all classes IN ORDER from low (1) to high(# of classes)]}
        print_only: only print an analysis of the findings
        expand_dates: expand date fields to "all" (year/month/day) or "year" or "month-year"
        due_date: a date to consider as due date - a new column will be added to represent differece between START_DATE and due date
        start_date: a date to consider as start date for due in calculation
        Returns
        -------
          A print with the analysis or new transformed columns.                
        """
        result = {}
        if(input_vars):
            X_train = X_train[input_vars]
            X_test = X_test[input_vars]
        #remaining_vars = X_train.columns 

        if(X_train.isna().sum().values.sum() > 0 or X_test.isna().sum().values.sum() > 0):
            print("X_train and X_test still have missing values, consider the clean missing method ")
            return

        if(not ((X_train.columns != X_test.columns).sum() == 0 and (X_train.dtypes != X_test.dtypes).sum() == 0)):
            print("X_train and X_test do not have matching feature list (columns)")
            return

        #numeric variables
        descriptive_statistics, _, _ = stats_helpers.eval_df(X_train)
        std_scaler = []
        minmax_scaler = []
        for col in X_train.select_dtypes(include=np.number):
            if((descriptive_statistics[descriptive_statistics["Name"] == col]["Is Normal"] == "Normal").bool()):
                print(f"Feature {col} will be scaled with a normalized scaler (since it is normally distributed)")
                std_scaler.append(col)
            else:
                print(f"Feature {col} will be scaled with a Min Max scaler (since it is NOT normally distributed)")
                minmax_scaler.append(col)
        print('debug - TODO')
        print(std_scaler)
        print('TODO')
        if(std_scaler):
            #scale numeric features (normal)
            std_scaler_transformer = StandardScaler()
            std_scaler_transformer.fit(X_train[std_scaler])
            X_train[std_scaler] = std_scaler_transformer.transform(X_train[std_scaler])
            X_test[std_scaler] = std_scaler_transformer.transform(X_test[std_scaler])
        if(minmax_scaler):
            #scale numeric features (non-normal)
            minmax_scaler_transformer = MinMaxScaler()
            minmax_scaler_transformer.fit(X_train[minmax_scaler])
            X_train[minmax_scaler] = minmax_scaler_transformer.transform(X_train[minmax_scaler])
            X_test[minmax_scaler] = minmax_scaler_transformer.transform(X_test[minmax_scaler])

        

        # calculate due date - start date
        # only if the the columns exist
        if(due_date in X_train.columns and start_date in X_train.columns):
            X_train['due_in_days'] = X_train[due_date] - X_train[start_date]
            X_test['due_in_days'] = X_test[due_date] - X_test[start_date]
            X_train['due_in_days'] = X_train['due_in_days'].dt.days
            X_test['due_in_days'] = X_test['due_in_days'].dt.days
        else:
            print("No due date calculation since columns specified are not in the dataset")
        
        print(y_train.shape)
        print(X_train.shape)
        # dates expansion
        for col in X_train.select_dtypes(include=['datetime64[ns]']):
            print(f"Exapnding feature {col} to {expand_dates}")
            if(expand_dates == 'year' or expand_dates == 'month-year' or expand_dates == 'all'):
                X_train[col+"_year"] = X_train[col].dt.year
                X_test[col+"_year"] = X_test[col].dt.year
            if(expand_dates == 'month-year' or expand_dates == 'all'):
                X_train[col+"_month"] = X_train[col].dt.month
                X_test[col+"_month"] = X_test[col].dt.month
            if(expand_dates == 'all'):
                X_train[col+"_day"] = X_train[col].dt.day
                X_test[col+"_day"] = X_test[col].dt.day
            
            print(f"Dropping the feature {col} after expansion")
            X_train.drop(col, axis = 1, inplace = True)
            X_test.drop(col, axis = 1, inplace = True)

        # categorical
        print(y_train.shape)
        print(X_train.shape)        
        #ordinal
        print("Ordinal variables specificed will be transformed to numeric according to the specified order")
        if(ordinal_var):
            ordinal_transformer = custom.Ordinal_Transformer()
            ordinal_transformer.fit( ordinal_var, X_train ,None)

            X_train = ordinal_transformer.transform(X_train, None)
            X_test = ordinal_transformer.transform(X_test, None)
            
        
        #WoE transformations
        woe_vars = []
        for feature in X_train.select_dtypes(exclude=[np.number]):
            if feature in ordinal_var: #we should not need this, but just in case
                continue
            if X_train[feature].nunique() >= woe_cat_threshold:
                print(f"Feature {feature} will be transformed to numeric using WoE - Weight of Evidence as it has more classes than the threshold")
                result[feature] = 'Weight of Evidence (WoE) transformation'
                woe_vars.append(feature)
        
        
        if(not print_only):
            for feature in woe_vars:
                X_train[feature] = X_train[feature].cat.remove_unused_categories()
                X_test[feature] = X_test[feature].cat.remove_unused_categories()
            #(X_train[woe_vars], X_test[woe_vars]) = self.woe_categorical(self, X_train[woe_vars], y_train, X_test[woe_vars], y_test, woe_vars, target_var)
            
            t_woe = custom.WoE_Transformer()
            
            t_woe.fit(X_train, y_train, target_var, woe_vars,  woe_cat_threshold)
            X_train = t_woe.transform(X_train,y_train)
            X_test = t_woe.transform(X_test, y_test)
            
        #remaining - one hot encoding
        
        ohe_vars = [col for col in X_train.columns if not is_numeric_dtype(X_train[col])]
        print("ohe - begin")
        print(ohe_vars)
    # define the data preparation for the columns
        if(ohe_vars):
            """
            t = [('cat', OneHotEncoder(), ohe_vars)]
            col_transform = ColumnTransformer(transformers=t)
            pipeline = Pipeline(steps=[ ('prep',col_transform) ])

            pipeline.fit(X_train, y_train)
            X_train  = pipeline.transform(X_train)
            X_test = pipeline.transform(X_test)
        """
            ohe_transformer = preprocessing.OneHotEncoder(sparse=False)
            ohe_transformer.fit(X_train[ohe_vars])
            
            X_train_ohe = X_train_ohe = pd.DataFrame(ohe_transformer.transform(X_train[ohe_vars]), columns=ohe_transformer.get_feature_names())
            X_test_ohe = pd.DataFrame(ohe_transformer.transform(X_test[ohe_vars]), columns=ohe_transformer.get_feature_names())

            X_train_ohe.set_index(X_train.index, inplace=True)
            X_test_ohe.set_index(X_test.index , inplace=True)
            X_train = pd.concat([X_train, X_train_ohe], axis=1)
            X_test = pd.concat([X_test, X_test_ohe], axis=1)

            print("remove remaining Categorical variables after OHE")
            X_train.drop(columns = ohe_vars, axis = 1, inplace = True)
            X_test.drop(columns = ohe_vars, axis = 1, inplace = True)
        """
        
        ohe_vars = []e
        for feature in X_train.select_dtypes(exclude=[np.number]):
            ohe_vars.append(ohe_vars)
        
        for feature in ohe_vars:
            (X_train_ohe, X_test_ohe) = self.fun_ohe(X_train, X_test, feature)
            
            X_train = pd.concat([X_train, X_train_ohe], axis=1).drop([feature], axis=1)
            X_test = pd.concat([X_test, X_test_ohe], axis=1).drop([feature], axis=1)

        """
        return X_train, X_test
    
    def woe_categorical(self, X_train, y_train, X_test, y_test, input_vars, target_var, num_cat_threshold = 5):
        """
        TODO: This method gets a list of categorical columns and target variable, this calculates
        the WoE for each column and each class within it (fit), this can be used later to transform
        the test set

        """
        
        

        #return (X_train, X_test)
        


    def fun_ohe(self, X_train_in, X_test_in, variable):
        print(X_train_in.shape)
        print(X_test_in.shape)
        ohe = OneHotEncoder(sparse=False)
        ohe.fit(X_train_in[[variable]])
        ohe_train = pd.DataFrame(ohe.transform(X_train_in[[variable]]), columns = ohe.get_feature_names([variable]))
        ohe_test = pd.DataFrame(ohe.transform(X_test_in[[variable]]), columns = ohe.get_feature_names([variable]))

        ohe_train.set_index(X_train_in.index, inplace=True)
        ohe_test.set_index(X_test_in.index , inplace=True)
        return(ohe_train, ohe_test)