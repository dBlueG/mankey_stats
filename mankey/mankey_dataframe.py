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
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from pandas.api.types import is_numeric_dtype, is_object_dtype
import numpy as np
import plotly.offline as py
import plotly.figure_factory as ff
import plotly.graph_objs as gobj
from sklearn.model_selection import train_test_split


from .stats_helpers import eval_df, clean_data
from .charting_helper import plot_univariate
from .custom_helpers import WoE_Transformer, Ordinal_Transformer

class MankeyDataFrame(pd.DataFrame):
    """
    The class is used to apply various data wrangling operations to Dataframes which includes Scaling and Encoding.
    In addition, the class provides some statistical based methods and robust functions such as Normality testing, Outlier detection, and Missing Data handling.
    It provides the end user with both general and specific cleaning functions


    Below is an example of instantiation of the class:

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
        plot_univariate(self, cols = input_vars, force_categorical=force_categorical)
    
    def explore_stat(self, input_vars=[], normal_test_alpha=0.05, grubbs_alpha=0.05, grubbs_max_outliers=20, iqr_factor=2.5):
       """
        This method describes the each feature of the dataset, for numeric fields, several statistics are 
        calculated including skewness, kurtosis, normality test (based on D'Agostino and Pearson's test), missing values, and quartiles.
        This also includes Grubbs test for outliers

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
        
        return eval_df(df_analyze)

    def create_train_test_sets(self, target_var, input_vars = [], test_size = 0.2, random_state = 42):
        """
        This method divides the dataframe into two datasets (training and test) to be used for ML models
        
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
        This method is used for missing data imputation and/or removal of the feature/observation (as per user preferences). 
        This is done using the statistical tests from the explore_stat method, 
        The method can be run in simulation mode or active mode. In simulation mode, only the recommended transformations
        will be highlighted but the data is not modified, while in active mode, the recommended transformations will be applied.
        
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
           
            (descriptive_statistics, logic_prep, _) = eval_df(df_train)
            print("Training set")
            print(logic_prep)
            print("===============")
            print(descriptive_statistics)

            (descriptive_statistics, _, _) = eval_df(df_test)
            print("===============")
            print("Test set stats")
            print(descriptive_statistics)

        else:
            (df_clean, df_clean_test) = clean_data(df_train, df_test)
            #X_train, y_train, X_test, y_test
            return df_clean.loc[:, df_clean.columns != target_var], df_clean[target_var], \
                   df_clean_test.loc[:, df_clean_test.columns != target_var], df_clean_test[target_var]

        
    
 
    def recommended_transformation(self, X_train, y_train, X_test, y_test, input_vars = [], ordinal_var = {}, woe_cat_threshold=5, print_only = True, expand_dates = 'all'
    , due_date = "EXPECTED_CLOSE_DATE", start_date = "LOAN_OPEN_DATE", target_var = 'BINARIZED_TARGET'):
        """
        This method can be used to recommend and/or apply transformations including:
        impute missing values, data manipulation, categorical variable handling.
        
        For WoE, this method gets a list of categorical columns and the target variable, it calculates
        the WoE for each column and each class within it (fit) and then replaces the original value of the category with the WoE, this can be used later to transform
        the test set. The transformed categorical variable will be a WOE value and it will be treated same as any continuous variable.
        
        This method can be used to transform categorical variables using one hot encoding, dummy variable encoding, or ordinal encoding.

        The method requires two datasets (training and test), all transformation rules are based
        on the training set which is then applied on both datasets.

        Note: the method can be used to also print the transformations, without actually generating new data.
       
        Here is an example of specifying ordinal categories for encoding:
        dictionary is as follows: {"feature_name": [list of all classes IN ORDER from low (1) to high(# of classes)]}
        
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
        if(input_vars):
            X_train = X_train[input_vars]
            X_test = X_test[input_vars]

        if(X_train.isna().sum().values.sum() > 0 or X_test.isna().sum().values.sum() > 0):
            print("X_train and X_test still have missing values, consider the clean missing method ")
            return

        if(not ((X_train.columns != X_test.columns).sum() == 0 and (X_train.dtypes != X_test.dtypes).sum() == 0)):
            print("X_train and X_test do not have matching feature list (columns)")
            return

        #numeric variables
        descriptive_statistics, _, _ = eval_df(X_train)
        std_scaler = []
        minmax_scaler = []
        for col in X_train.select_dtypes(include=np.number):
            if((descriptive_statistics[descriptive_statistics["Name"] == col]["Is Normal"] == "Normal").bool()):
                print(f"Feature {col} will be scaled with a normalized scaler (since it is normally distributed)")
                std_scaler.append(col)
            else:
                print(f"Feature {col} will be scaled with a Min Max scaler (since it is NOT normally distributed)")
                minmax_scaler.append(col)

        #only change data if we are not running in print_only mode
        if(not print_only):
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
            print(f"Calulate days difference beteen {due_date} and {start_date}")
            if(not print_only):
                X_train['due_in_days'] = X_train[due_date] - X_train[start_date]
                X_test['due_in_days'] = X_test[due_date] - X_test[start_date]
                X_train['due_in_days'] = X_train['due_in_days'].dt.days
                X_test['due_in_days'] = X_test['due_in_days'].dt.days
        else:
            print("No due date calculation since columns specified are not in the dataset")
        
        
        # dates expansion
        for col in X_train.select_dtypes(include=['datetime64[ns]']):
            print(f"Exapnding feature {col} to {expand_dates}")
            if(not print_only):
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
            if(not print_only):
                X_train.drop(col, axis = 1, inplace = True)
                X_test.drop(col, axis = 1, inplace = True)

        # categorical
                
        #ordinal
        
        if(ordinal_var):
            print("Ordinal variables specificed will be transformed to numeric according to the specified order")
            if(not print_only):
                ordinal_transformer = Ordinal_Transformer()
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
            
            t_woe = WoE_Transformer()
            
            t_woe.fit(X_train, y_train, target_var, woe_vars,  woe_cat_threshold)
            X_train = t_woe.transform(X_train,y_train)
            X_test = t_woe.transform(X_test, y_test)
            
        #remaining - one hot encoding
        
        ohe_vars = [col for col in X_train.columns if not is_numeric_dtype(X_train[col])]
        print("The following features will be transformed using one-hot-encoding:")
        print(ohe_vars)
    # define the data preparation for the columns
        if(not print_only and ohe_vars):
            
            
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
        
        return X_train, X_test
    
