# Mankey Stats

![PythonVersion](https://img.shields.io/badge/python-3.6%20|3.7%20|%203.8%20|%203.9-success)
[![License https://github.com/dBlueG/mankey_stats/blob/main/LICENSE.md](https://img.shields.io/badge/license-MIT-success.svg)](https://github.com/dBlueG/mankey_stats/blob/main/LICENSE.md)
[![PyPI version fury.io](https://badge.fury.io/py/mankey-stats.svg)](https://pypi.python.org/pypi/mankey-stats/)
[![Documentation Status https://mankey-stats.readthedocs.io/en/main/genindex.html](https://readthedocs.org/projects/feature-engine/badge/?version=latest)](https://mankey-stats.readthedocs.io/en/main/genindex.html)



![alt text](https://github.com/dBlueG/mankey_stats/blob/main/mankey.png)


Mankey_stats is a Python library that allows the user to quickly and efficiently perform data preparation techniques to transform the datasets for ML modeling, this is done through the utilization of Scikit-learn's fit() and transform() methods.


## Documentation

* [Documentation](https://mankey-stats.readthedocs.io/en/main/#)


## Current mankey-stats's transformers include functionality for:

* Datetime Feature Extraction
* Ordinal Encoding
* Weight of Evidence Calculation for Categorical Variables
* Outlier Detection 
* Outlier Removal
* Missing Value handling recommendation
* Feature Normality test
* Preprocessing
* Scikit-learn Wrappers

### Imputation Methods
* Fill Null Values with Mean/Median

### Encoding Methods
* OneHotEncoder
* OrdinalEncoder
* WoEEncoder

### Outlier Handling methods
* Grubb's Test

### Variable Transformation methods
* MinMaxScaler
* StandardScaler

### Variable Creation:
 * date_expander

### Datetime
 * date_expander
 
### Preprocessing
 * MatchVariables
 
### Wrappers:
 * Based on Sklearn Transformers

## Installation

From PyPI using pip:

```
pip install mankey_stats
```

Or simply clone it from github:

```
git clone https://github.com/mankey_stats/mankey_stats.git
```

## Example Usage

```python
>>> import pandas as pd
>>> from mankey_stats.ordinal_encoder as ordinal_encoder

>>> data = {'type':  ['bad', 'average', 'good', 'very good', 'excellent'],
            'level': [1, 2, 3, 4, 5]
            
>>> levels_dict = {'type':  ['bad', 'average', 'good', 'very good', 'excellent'],
                   }
                   
>>> data = pd.DataFrame(data)
>>> print(data)
```

```
Out[1]:
type    level
bad       1
average   2
good      3
very good 4
excellent 5  
Name: var_A, dtype: int64
```
    
```python 
    t_ord = transformers.Ordinal_Transformer()
    t_ord.fit( levels_dict, df,None)

    df = t_ord.transform(df, None)
```

```
Out[2]:
0       1
1       2
2       3
3       4
4       5
Name: var_A, dtype: int64
```

Find more in the [documentation](https://mankey-stats.readthedocs.io/en/main/#).

### Documentation

mankey-stats documentation is built using [Sphinx](https://www.sphinx-doc.org) and is hosted on [Read the Docs](https://readthedocs.org/).

To rebuild the documentation, you need to use the `make html` command to re-build the sphinx docs


## License

MIT

