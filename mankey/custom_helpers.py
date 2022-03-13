from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
import datetime

#Date transformer
def date_expander(X, input_var=[], date_format = 'YYYY-MM-DD'):
    """
    Expands a date series/or dataframe with date columns into a new data frame with 
    five columns for each date: day of the week, month of the year, day of month, days till due date (need reference date),
    and the year.
    

    Parameters
    ----------
        X: dataframe or series
        input_var: list of feature names containing dates
        date_format: how the date is formated in the data (YYYY-MM-DD or MM-DD-YYYY or DD-MM-YYYY are implemented)
    Returns
    -------
        A new dataframe with expanded date information (see description)

    """


def date_expand_func(year, month, day, ref_date: datetime.date):
    dt = datetime.date(year,month, day)
    
    day_of_month = dt.strftime("%d")
    month_of_year = dt.strftime("%m")
    year = dt.strftime("%Y")
    day_of_week = dt.strftime("%w")

    delta = dt - ref_date
    days_remaining = abs(delta.days)
    return(year, month_of_year, day_of_month, day_of_week, days_remaining)

# Ordinal transformer

class Ordinal_Transformer(BaseEstimator, TransformerMixin):

    ord_enc = []
    def __init__(self):
        super().__init__()
        self.dict_ = {}
        
        


    def fit(self, class_order_dict, X: pd.DataFrame, y = None, input_vars=[]):
        
        self.df = X
        
        
        #make sure we have the same columns as input_vars
        #if list(X.columns)!=input_vars:
        #    return("Columns do not match")
            
        self.features_transform = []
        self.feature_levels = []
        for feature in class_order_dict:
            self.features_transform.append(feature)
            self.feature_levels.append(class_order_dict[feature])

        X_limited = X[self.features_transform]
        # if the test set does not have the category... do not fail!!
        self.ord_enc = OrdinalEncoder(categories=self.feature_levels, handle_unknown="use_encoded_value", unknown_value= -1)
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

