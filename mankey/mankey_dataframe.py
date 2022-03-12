# -*- coding: utf-8 -*-
"""
Created on in 2022

@author: Group A - Advanced Python - IE MBD
Hussain Alhajjaj  
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
        

