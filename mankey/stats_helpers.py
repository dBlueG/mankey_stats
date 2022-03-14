import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from pandas.api.types import is_object_dtype, is_numeric_dtype
from scipy import stats

import seaborn as sns


def grubbs_remove_outlier(X, alpha):
    ''' 
    Run Grubbs’ Test and remove the outlier based on significance level
    and grubbs value and  grubbs critical value

    Parameters
    ----------
    X : ndarray
        A python list to be tested for outliers. 

    alpha : float
        The significance level.

    Returns
    -------
    X : ndarray
        The original array with one outlier removed. 
    '''

    # Calculte Grubbs Statistics Value
    std_dev = np.std(X)
    avg_X = np.mean(X)
    abs_val_minus_avg = abs(X - avg_X)
    max_of_deviations = max(abs_val_minus_avg)
    grubbs_value = max_of_deviations / std_dev

    # Find Max Index
    max_index = np.argmax(abs_val_minus_avg)
    outlier = X[max_index]

    # Calculte Grubbs Critical Value
    size = len(X)
    t_dist = stats.t.ppf(1 - alpha / (2 * size), size - 2)
    numerator = (size - 1) * np.sqrt(np.square(t_dist))
    denominator = np.sqrt(size) * np.sqrt(size - 2 + np.square(t_dist))
    grubbs_critical_value = numerator / denominator

    if grubbs_value > grubbs_critical_value:
        print('{} is an outlier. Grubbs value  > Grubbs critical: {:.4f} > {:.4f} \n'.format(
            outlier, grubbs_value, grubbs_critical_value))
        #X = np.delete(X, np.where(X == outlier) )
        X.remove(outlier)

    return X


def grubbs(X, max_outliers,  alpha=0.05, name=''):
    '''
    Grubbs’ Test is also known as the maximum normalized residual test or extreme studentized deviate test is a test used to detect outliers in a univariate data set assumed to come from a normally distributed population. This test is defined for the hypothesis:

    Ho: There are no outliers in the data set
    Ha: There is exactly one outlier in the database

    Parameters
    ----------
    X : ndarray
        A python list to be tested for outliers.  

    alpha : float
        The significance level.

    max_outliers: int
    The maximum  number of iteration that will be deleted by this method  

    Returns
    -------
    X : ndarray
        The original array with outliers removed.

    outliers: the list of all outliers
    '''

    init_x_len = len(X)
    reach_end = False

    clean_X = X.copy().tolist()

    for i in range(max_outliers):
        clean_X = grubbs_remove_outlier(clean_X, alpha)
        if len(X) == len(clean_X):
            reach_end = True
            break

    if name != "":
        name = " in " + name + " dataset"
    else:
        name = " in the dataset "

    outlisers = []

    if (len(X)-len(clean_X) == 0):
        print("we can not reject the null hypothesis and there are no outliers " + name)

    else:
        print("we reject the null hypothesis and there are outliers "+name)
        outlisers = [value for value in X if value not in clean_X]

        if reach_end:
            print("Found all Grubbs outliers, " +
                  str(len(X)-len(clean_X)) + " outliers")
        else:
            print("Found some Grubbs outliers, " + str(len(X)-len(clean_X)
                                                       ) + " outliers. increase the max itration to find more")

    return clean_X, outlisers


def detect_outliers(X, factor=1.5):

    # list to store outlier indices
    outlier_indices = []

    # Get the 1st quartile (25%)
    Q1 = np.percentile(X, 25)

    # Get the 3rd quartile (75%)
    Q3 = np.percentile(X, 75)

    # Get the Interquartile range (IQR)
    IQR = Q3 - Q1

    # Define our outlier step
    outlier_step = 1.5 * IQR

    return X[(X < Q1 - outlier_step) | (X > Q3 + outlier_step)]


def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != -1].sort_values(
        '% of Total Values', ascending=False).round(1)
    # print("Your selected dataframe has " + str(df.shape[1]) + " column/s and " + str(df.shape[0]) + " row/s.\n"
    #       "There are " + str(mis_val_table_ren_columns.shape[0]) +
        #   " columns that have missing values.")
    return mis_val_table_ren_columns


#logic preparation
def logic_preparation(x_df, no_treatment_columns = [] ):
    missing = missing_values_table(x_df).iloc[:,1]
    df_missing = pd.DataFrame(columns=['name', 'missing','treatment' ])

    df_missing["missing"] = missing.tolist()
    df_missing["name"] = missing.index.values.tolist()    
    df_missing.treatment = "Treat" 
    df_missing.treatment.loc[df_missing["missing"] >= 70] = "Drop" 
    df_missing.treatment.loc[df_missing["missing"] < 5] = "Omit"
    df_missing.treatment.loc[df_missing["missing"] == 0] = "Do Nothing" 
  
    columns_to_drop = []
    columns_to_drop = df_missing.name.loc[df_missing["treatment"] == "Drop"].tolist() 
    columns_to_drop = list(dict.fromkeys(columns_to_drop)) 
    
    
    for i in range(len(df_missing.index)):
        if (df_missing.iloc[i]["name"] in no_treatment_columns and
                df_missing.iloc[i]["treatment"] == "Treat"):
            df_missing.treatment.iloc[i] = "Omit"
            
    if (len(columns_to_drop) > 0 ):
        print("\nColumns to drop are :" )
        print(*columns_to_drop, sep='\n')

    columns_to_omit_missing_values = [] 
    columns_to_omit_missing_values = df_missing.name.loc[df_missing["treatment"] == "Omit"].tolist() 
    columns_to_omit_missing_values = list(dict.fromkeys(columns_to_omit_missing_values)) 
    
    if (len(columns_to_omit_missing_values) > 0 ): 
        print("\ncannot be treated columns is :" )
        print(*columns_to_omit_missing_values, sep='\n')
    
    
    return df_missing


def eval_df_bk(df_o, normal_test_alpha=0.05,
            grubbs_alpha=0.05,
            grubbs_max_outliers=20,
            iqr_factor=2.5
            ):
    df = df_o.copy()

    names = []
    types = []
    count = []
    unique = []
    top = []
    freq = []
    mins = []
    mean = []
    variance = []
    skewness = []
    kurtosis = []
    maxs = []
    q1 = []
    q2 = []
    q3 = []
    iqrs = []
    missing = []
    missing_prc = []
    is_normals = []
    has_outliers = []

    outliers = {}

    df_describe = df.describe(include='all')

    df_missing = missing_values_table(df)

    for item in df.columns:
        names.append(item)
        count.append(df_describe[item]['count'])
        missing.append(df_missing.loc[item]['Missing Values'])
        missing_prc.append(df_missing.loc[item]['% of Total Values'])

        if is_numeric_dtype(df[item]):
            types.append('Numerical')
            num_item_describe = stats.describe(df[item])
            _q1 = df_describe[item]['25%']
            _q3 = df_describe[item]['75%']
            iqr = _q3 - _q1

            _, p = stats.normaltest(df[item])

            is_normal = True
            if (p < normal_test_alpha):
                is_normal = False
                is_normals.append("Non-normal")
            else:
                is_normal = True
                is_normals.append("Normal")

            unique.append(np.nan)
            top.append(np.nan)
            freq.append(np.nan)
            mins.append(df_describe[item]['min'])
            maxs.append(df_describe[item]['max'])
            mean.append(df_describe[item]['mean'])
            q1.append(_q1)
            q2.append(df_describe[item]['50%'])
            q3.append(_q3)
            variance.append(num_item_describe.variance)
            skewness.append(num_item_describe.skewness)
            kurtosis.append(num_item_describe.kurtosis)
            iqrs.append(iqr)

            if (is_normal):
                _, grubbs_outliers = grubbs(
                    df[item].dropna(), grubbs_max_outliers, grubbs_alpha, item)

                if len(grubbs_outliers) > 0:
                    outliers[item] = grubbs_outliers
            else:
                iqr_outliers = detect_outliers(df[item], iqr_factor)

                if len(iqr_outliers) > 0:
                    outliers[item] = iqr_outliers

            if item in outliers:
                has_outliers.append('True')
            else:
                has_outliers.append('False')

        else:
            types.append('Categorical')
            unique.append(df_describe[item]['unique'])
            top.append(df_describe[item]['top'])
            freq.append(df_describe[item]['freq'])
            mins.append(np.nan)
            maxs.append(np.nan)
            mean.append(np.nan)
            variance.append(np.nan)
            skewness.append(np.nan)
            kurtosis.append(np.nan)
            q1.append(np.nan)
            q2.append(np.nan)
            q3.append(np.nan)
            iqrs.append(np.nan)
            is_normals.append(np.nan)
            has_outliers.append(np.nan)
    descriptive_statistics = pd.DataFrame({'Name': names,
                                           'Type': types,
                                           'Count': count,
                                           'Unique': unique,
                                           'Top': top,
                                           'Freq': freq,
                                           'Min': mins,
                                           'Max': maxs,
                                           'Mean': mean,
                                           'Variance': variance,
                                           'Skewness': skewness,
                                           'Kurtosis': kurtosis,
                                           '25%': q1,
                                           '50%': q2,
                                           '75%': q3,
                                           'IQR': iqrs,
                                           '# Missing Values': missing,
                                           '% Missing Values': missing_prc,
                                           'Is Normal': is_normals,
                                           'Has Outliers': has_outliers
                                           })

    return descriptive_statistics, logic_preparation(df), outliers



def eval_df(df_o, 
            normal_test_alpha = 0.05,
            grubbs_alpha = 0.05,
            grubbs_max_outliers = 20,
            iqr_factor = 2.5 ):
    df = df_o.copy()
    
    output = {} 
    
    output['name']          = []
    output['count']         = []
    output['missing']       = []
    output['missing_prc']   = []
    output['types']         = []
    output['q1']            = []
    output['q2']            = []
    output['q3']            = []
    output['is_normal']     = []
    output['unique']        = []
    output['top']           = []
    output['freq']          = []
    output['min']           = []
    output['max']           = []
    output['mean']          = []
    output['sd']            = [] 
    output['variance']      = []
    output['skewness']      = []
    output['kurtosis']      = []
    output['iqrs']          = []  
    output['has_outliers']  = []  
    outliers   = {}
     

    def get_stat(name): 
        if(name in output['name']):
            return
        
        mis_val         = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)  
        
        df_describe = df.describe(include='all', datetime_is_numeric=True)  
        
        output['name'].append(name)
        output['count'].append(df_describe[name]['count'])
        output['missing'].append( mis_val[name])
        output['missing_prc'].append(mis_val_percent[name]) 
         
        if is_numeric_dtype(df[name]):
            output['types'].append('Numerical')
            
            num_item_describe  =stats.describe(df[name]) 
            output['q1'].append(df_describe[name]['25%'])
            output['q3'].append(df_describe[name]['75%'])
            
            iqr = df_describe[name]['75%'] - df_describe[name]['25%']
            
            _, p = stats.normaltest(df[name])
             
            is_normal = True
            if (p < normal_test_alpha):
                is_normal = False 
                output['is_normal'].append("Non-normal")  
            else:
                is_normal = True
                output['is_normal'].append("Normal")
                 
            output['unique'].append(np.nan)
            output['top'].append(np.nan)
            output['freq'].append(np.nan) 
            output['min'].append(df_describe[name]['min'])
            output['max'].append(df_describe[name]['max'])
            output['mean'].append(df_describe[name]['mean']) 
            output['sd'].append(df_describe[name]['std']) 
            output['q2'].append(df_describe[name]['50%']) 
            output['variance'].append(num_item_describe.variance)
            output['skewness'].append(num_item_describe.skewness)
            output['kurtosis'].append(num_item_describe.kurtosis)
            output['iqrs'].append(iqr)
            #to skip outlier detection in binary vars or very small range features   
            if(df_describe[name]['max'] -  df_describe[name]['min'] <= 1):
                pass

            elif (is_normal):  
                _ , grubbs_outliers = grubbs( df[name].dropna() , grubbs_max_outliers, grubbs_alpha,name) 
                
                if len(grubbs_outliers) > 0:
                    outliers[name]  = grubbs_outliers
            else:  
                iqr_outliers =  detect_outliers(df[name],iqr_factor) 
                
                if len(iqr_outliers) > 0:
                    outliers[name]  = iqr_outliers  
             
            if name in outliers:
                output['has_outliers'].append('True')
            else:
                output['has_outliers'].append('False')
                    
        else: 
            output['types'].append('Categorical')
            output['unique'].append(df_describe[name]['unique'])
            output['top'].append(df_describe[name]['top'])
            output['freq'].append(df_describe[name]['freq']) 
            output['min'].append(np.nan)
            output['max'].append(np.nan)
            output['mean'].append(np.nan)
            output['sd'].append(np.nan)
            output['variance'].append(np.nan)
            output['skewness'].append(np.nan)
            output['kurtosis'].append(np.nan)
            output['q1'].append(np.nan)
            output['q2'].append(np.nan)
            output['q3'].append(np.nan) 
            output['iqrs'].append(np.nan) 
            output['is_normal'].append(np.nan) 
            output['has_outliers'].append(np.nan) 
    
    
    vfunc = np.vectorize(get_stat) 
    vfunc(df.columns)
 
    descriptive_statistics = pd.DataFrame({'Name'     : output['name'],
                                           'Type'     : output['types'] ,
                                           'Count'    : output['count'], 
                                           'Unique'   : output['unique'],
                                           'Top'      : output['top'],
                                           'Freq'     : output['freq'] ,
                                           'Min'      : output['min'],
                                           'Max'      : output['max'],
                                           'Mean'     : output['mean'],
                                           'SD'       : output['sd'],
                                           'Variance' : output['variance'] ,
                                           'Skewness' : output['skewness'],
                                           'Kurtosis' : output['kurtosis'], 
                                           '25%'      : output['q1'],
                                           '50%'      : output['q2'],
                                           '75%'      : output['q3'] ,
                                           'IQR'      : output['iqrs'] ,
                                           '# Missing Values' : output['missing'],
                                           '% Missing Values' : output['missing_prc'],
                                           'Is Normal': output['is_normal'],
                                           'Has Outliers':output['has_outliers']
                                           })
    
    
    
    return descriptive_statistics , logic_preparation(df) , outliers


def clean_data(df_train, df_test):

    df_clean = df_train.copy()
    descriptive_statistics, df_treatment_logic, outliers = eval_df(df_clean)

    df_test_clean = df_test.copy()
    descriptive_statistics_test, df_treatment_logic_test, outliers_test = eval_df(
        df_test_clean.copy())
    
    print("Number of rows before handling the outliers values in training " +
          str(df_clean.shape[0])+"\n")
    print("Number of rows before handling the outliers values in testing " +
          str(df_test_clean.shape[0])+"\n")

    # clean all outliers in training data set
    for item in df_clean.columns:
        if item in outliers:
            df_clean = df_clean.loc[~df_clean[item].isin(outliers[item])]

    # clean all outliers in test data set
    for item in df_test_clean.columns:
        if item in outliers_test:
            df_test_clean = df_test_clean.loc[~df_test_clean[item].isin(
                outliers_test[item])]

    print("Number of rows after handling outliers (before handling missing data) in training " +
          str(df_clean.shape[0])+"\n")
    print("Number of rows after handling outliers (before handling missing data) in testing " +
          str(df_test_clean.shape[0])+"\n")

    for column in df_clean.columns:
        if(df_treatment_logic.loc[df_treatment_logic.name == column, :].treatment.tolist()[0] == "Drop"):
            df_clean = df_clean.drop(column, axis=1)
            df_test_clean = df_test_clean.drop(column, axis=1)
        elif(df_treatment_logic.loc[df_treatment_logic.name == column, :].treatment.tolist()[0] == "Omit"):
            df_clean = df_clean.dropna(subset=[column])
        elif(df_treatment_logic_test.loc[df_treatment_logic_test.name == column, :].treatment.tolist()[0] == "Omit"):
            df_test_clean = df_test_clean.dropna(subset=[column])
        

   

    # Outliers Tratment

    # impute based if the data is normal or not
    for item in df_clean.columns:
        row = descriptive_statistics[descriptive_statistics['Name'] == item]
        is_norm = str(row['Is Normal'].to_string(index=False))

        if (is_norm == 'Normal'):
            df_clean[item].fillna((df_clean[item].mean()), inplace=True)
            df_test_clean[item].fillna((df_clean[item].mean()), inplace=True)

        elif (is_norm == 'Non-normal'):
            df_clean[item].fillna((df_clean[item].median()), inplace=True)
            df_test_clean[item].fillna((df_clean[item].median()), inplace=True)
        else:
            df_clean[item].fillna((df_clean[item].mode()[0]), inplace=True)
            df_test_clean[item].fillna((df_clean[item].mode()[0]), inplace=True)

    print("Number of rows after handling the missing values in training " +
          str(df_clean.shape[0])+"\n")
    print("Number of rows after handling the missing values in testing " +
          str(df_test_clean.shape[0])+"\n")

    return df_clean, df_test_clean
