# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 22:49:01 2021

@author: Nicolas Ponte
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

class BrushingDataframe(pd.DataFrame):
    """
    The class is used to extend the properties of Dataframes to a prticular
    type of Dataframes in the Risk Indistry. 
    It provides the end user with both general and specific cleaning functions, 
    though they never reference a specific VARIABLE NAME.
    
    It facilitates the End User to perform some Date Feature Engineering,
    Scaling, Encoding, etc. to avoid code repetition.
    """

    #Initializing the inherited pd.DataFrame
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
    
    @property
    def _constructor(self):
        def func_(*args,**kwargs):
            df = BrushingDataframe(*args,**kwargs)
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
    
 
    def recommended_transformation(self, input_vars=[], target='' ):
        """

        TO BE IMPLEMENTED: data preparation (for each column provide methods to perform
        transformations - for example: time calculation like age, days as customers, 
        days to due date, label encoding, imputation, standard scalling, dummy creation 
        or replacement of category value by its probability of default depending, justify 
        transformation depending of the variable type, or explain why transformation is 
        not necessary);

        Returns
        -------
          A print with the analysis or new transformed columns.                
        """
        return "To be implemented."
    