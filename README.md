# Mankey Stats

![PythonVersion](https://img.shields.io/badge/python-3.6%20|3.7%20|%203.8%20|%203.9-success)
[![License https://github.com/dBlueG/mankey_stats/blob/main/LICENSE.md](https://img.shields.io/badge/license-MIT-success.svg)](https://github.com/dBlueG/mankey_stats/blob/main/LICENSE.md)
[![PyPI version fury.io](https://badge.fury.io/py/mankey-stats.svg)](https://pypi.python.org/pypi/mankey-stats/)
[![Documentation Status https://mankey-stats.readthedocs.io/en/main/genindex.html](https://readthedocs.org/projects/feature-engine/badge/?version=latest)](https://mankey-stats.readthedocs.io/en/main/genindex.html)



![alt text](https://github.com/dBlueG/mankey_stats/blob/main/mankey.png)


Mankey_stats is a Python library that allows the user to quickly and efficiently perform data preparation techniques to transform the datasets for ML modeling, this is done through the utilization of several transformation and statistical analysis methods.


## Documentation

* [Documentation](https://mankey-stats.readthedocs.io/en/main/#)

## Primary functionality include:

* Detailed analysis of features, including numerical distibution tests
* Analysis and handling of outliers and missing data
* Interactive plotting and data visualization functionality
* Tranformation options including One hot encoding, ordinal transformations, and weight of evidence
* Functionality to prepare date fields for ML models
* Ability to examine and recommend without modifying the underlying data
* Optimized logic to ensure fast execution times, using numpy, scipy, and vectorization techniques

## Analysis of features:

* Feature Normality test
* Grubb's test and Tucky's fences for handling outliers (based on stat. distribution)
* Missing value analysis (% and best method to handle - mode/median/or mean)
* Best scaling methods are selected for each numeric feature (min-max scaler or standard scaler)

## Multiple methods to handle categorical features:

* One Hot encoder
* Ordinal encoder
* Weight of Evidence transformations

## Date manipulation

* Ability to expand date fields to YEAR, MONTH, and/or DAY fields
* Subtract date features to create a "due in days" field



# Installation

The library is published in the PyPi repository, it can be installed with pip:
```
pip install mankey_stats
```

Feel free to help us improve, simply clone it from this github and submit your features :)
```
git clone https://github.com/mankey_stats/mankey_stats.git
```

# Dependencies:
We rely on the proven ML libraries: pandas, Seabor, plotly, numpy, scipy, and Scikit Learn

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

You can re-build the docs using `build html`


## License

MIT

