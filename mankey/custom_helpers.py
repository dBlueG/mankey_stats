from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np


# Ordinal transformer
class Ordinal_Transformer(BaseEstimator, TransformerMixin):
    """
    Ordinal transformer takes a dictionary of feature names and the levels of the categories and returns a dataframe with the levels encoded, 
    this is useful for categorical variables where we need to encode the order based on the user input. In case, the test set does not have the
    category then the category is not encoded and the test set is not changed.

    Parameters
    ----------
    class_order_dict: dictionary of feature names and the levels of the categories
    X: dataframe
    y: target variable
    input_vars: list of feature names containing categorical variables

    Returns
    -------
    A new dataframe with the levels encoded
    """

    ord_enc = []

    def __init__(self):
        super().__init__()
        self.dict_ = {}

    def fit(self, class_order_dict, X: pd.DataFrame, y=None):

        self.df = X

        self.features_transform = []
        self.feature_levels = []
        for feature in class_order_dict:
            self.features_transform.append(feature)
            self.feature_levels.append(class_order_dict[feature])

        X_limited = X[self.features_transform]
        # if the test set does not have the category... do not fail!!
        self.ord_enc = OrdinalEncoder(
            categories=self.feature_levels, handle_unknown="use_encoded_value", unknown_value=-1)
        self.ord_enc.fit(X_limited)

        return self

    def transform(self, X, y=None):

        if(not self.ord_enc):
            return("you must fit first")
        X_limited = X[self.features_transform]
        X_limited = self.ord_enc.transform(X_limited)
        X_limited = X_limited.astype('int64')
        X[self.features_transform] = X_limited

        return X


# Custom categorical WoE handler
class WoE_Transformer(BaseEstimator, TransformerMixin):
      """
      WoE values for the various categories of a categorical variable can be used to impute a categorical feature 
      and convert it into a numerical feature as a logistic regression model requires all its features to be numerical.
    
      This class will calculate WoE for a given variable 

      Parameters
      ----------

     input_var: name of the variable to calculate WoE for
 
     class_order_dict: dictionary containing the class order for each variable

     target_var: name of the target variable

     Returns
     ----------
     A new dataframe with the WoE values for each category
     """
    def __init__(self):
        super().__init__()
        self.dict_ = {}

    def _woe_cal(self, input, target):
        df = self.df
        # Define a binary target: duration >= 24
        # df['binary_target'] = (df.Duration >= 24).astype(int)
        woe_df = pd.DataFrame(df.groupby(input, as_index=False)[target].mean())
        
        
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

    def fit(self, X: pd.DataFrame, y: pd.Series, target_var, input_vars, num_cat_threshold):
        self.df = X
        self.input_vars = input_vars
        print(y.shape)
        self.df['target'] = y
        
        print(input_vars)
        # check target is binary, otherwise raise error

        if(y.nunique() != 2):
            return "WoE implementation only supports binary targets"

        # calculate woe for each categorical variable and store it in the instance attributes
        # ln(#bad / #good) per class
        for var_name in input_vars:
            
            if(X[var_name].nunique() >= num_cat_threshold):
                woe_df = self._woe_cal(var_name, 'target')
                self.dict_[var_name] = woe_df
            else:
                self.input_vars.remove(var_name)

        self.df.drop('target', axis = 1, inplace=True)
        return self

    def transform(self, X, y=None):

        input_vars = self.input_vars
        for var_name in input_vars:
            X[var_name+"_woe"] = 0

            for line in range(len(self.dict_[var_name])):

                X[var_name+"_woe"] = np.where(X[var_name] == self.dict_[var_name].loc[line, var_name], self.dict_[
                                              var_name].loc[line, "WoE"], X[var_name+"_woe"])

            X.drop(var_name, axis=1, inplace=True)
        return X
