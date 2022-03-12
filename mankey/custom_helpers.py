from msilib.schema import Error
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


# Ordinal transformer

class Ordinal_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.dict_ = {}

    def _woe_cal(self, input, target) -> pd.DataFrame:
        df = self.df
        # Define a binary target: duration >= 24
        # df['binary_target'] = (df.Duration >= 24).astype(int)
        woe_df = pd.DataFrame(df.groupby(input, as_index = False)[target].mean())
        woe_df = woe_df.rename(columns={target: 'positive'})
        woe_df['negative'] = 1 - woe_df['positive']

        # Cap probabilities to avoid zero division
        woe_df['positive'] = np.where(
            woe_df['positive'] < 1e-6, 1e-6, woe_df['positive'])
        woe_df['negative'] = np.where(
            woe_df['negative'] < 1e-6, 1e-6, woe_df['negative'])

        woe_df['WoE'] = np.log(woe_df['positive'] / woe_df['negative'])
        woe_dict = woe_df['WoE'].to_dict()

        return woe_df

    def fit(self, X: pd.DataFrame, y: pd.Series, target_name: str, input_vars=[],   num_cat_threshold=5):
        
        self.df = X
        self.input_vars = input_vars
        self.df['target'] = y

        # check target is binary, otherwise raise error
       
        
        if(y.nunique() != 2):
            return "WoE implementation only supports binary targets"
                
        

        # calculate woe for each categorical variable and store it in the instance attributes
        # ln(#bad / #good) per class
        for var_name in input_vars:
            if(X[var_name].nunique() >= num_cat_threshold):
                woe_df = self._woe_cal(var_name, target_name)
                self.dict_[var_name] = woe_df
            else:
                self.input_vars.remove(var_name)

        return self

    def transform(self, X, y=None):

        input_vars = self.input_vars
        for var_name in input_vars:
            X[var_name+"_woe"] = 0
            
            for line in range(len(self.dict_[var_name])):
                
                X[var_name+"_woe"] = np.where(X[var_name] == self.dict_[var_name].loc[line,var_name], self.dict_[var_name].loc[line,"WoE"], X[var_name+"_woe"])

            X.drop(var_name, axis = 1, inplace = True)
        return X




# Custom categorical WoE handler
class WoE_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.dict_ = {}

    def _woe_cal(self, input, target) -> pd.DataFrame:
        df = self.df
        # Define a binary target: duration >= 24
        # df['binary_target'] = (df.Duration >= 24).astype(int)
        woe_df = pd.DataFrame(df.groupby(input, as_index = False)[target].mean())
        woe_df = woe_df.rename(columns={target: 'positive'})
        woe_df['negative'] = 1 - woe_df['positive']

        # Cap probabilities to avoid zero division
        woe_df['positive'] = np.where(
            woe_df['positive'] < 1e-6, 1e-6, woe_df['positive'])
        woe_df['negative'] = np.where(
            woe_df['negative'] < 1e-6, 1e-6, woe_df['negative'])

        woe_df['WoE'] = np.log(woe_df['positive'] / woe_df['negative'])
        woe_dict = woe_df['WoE'].to_dict()

        return woe_df

    def fit(self, X: pd.DataFrame, y: pd.Series, target_name: str, input_vars=[],   num_cat_threshold=5):
        
        self.df = X
        self.input_vars = input_vars
        self.df['target'] = y

        # check target is binary, otherwise raise error
       
        
        if(y.nunique() != 2):
            return "WoE implementation only supports binary targets"
                
        

        # calculate woe for each categorical variable and store it in the instance attributes
        # ln(#bad / #good) per class
        for var_name in input_vars:
            if(X[var_name].nunique() >= num_cat_threshold):
                woe_df = self._woe_cal(var_name, target_name)
                self.dict_[var_name] = woe_df
            else:
                self.input_vars.remove(var_name)

        return self

    def transform(self, X, y=None):

        input_vars = self.input_vars
        for var_name in input_vars:
            X[var_name+"_woe"] = 0
            
            for line in range(len(self.dict_[var_name])):
                
                X[var_name+"_woe"] = np.where(X[var_name] == self.dict_[var_name].loc[line,var_name], self.dict_[var_name].loc[line,"WoE"], X[var_name+"_woe"])

            X.drop(var_name, axis = 1, inplace = True)
        return X

