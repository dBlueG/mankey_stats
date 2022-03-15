# Mankey Stats

![PythonVersion](https://img.shields.io/badge/python-3.6%20|3.7%20|%203.8%20|%203.9-success)
[![License https://github.com/dBlueG/mankey_stats/blob/main/LICENSE.md](https://img.shields.io/badge/license-MIT-success.svg)](https://github.com/dBlueG/mankey_stats/blob/main/LICENSE.md)
[![PyPI version](https://badge.fury.io/py/feature-engine.svg)](https://badge.fury.io/py/feature-engine)
[![Documentation Status https://mankey-stats.readthedocs.io/en/main/genindex.html](https://readthedocs.org/projects/feature-engine/badge/?version=latest)](https://mankey-stats.readthedocs.io/en/main/genindex.html)
[![Downloads](https://pepy.tech/badge/feature-engine)](https://pepy.tech/project/feature-engine)
[![Downloads](https://pepy.tech/badge/feature-engine/month)](https://pepy.tech/project/feature-engine)

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
* Variable Selection
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
 * SklearnTransformerWrapper

## Installation

From PyPI using pip:

```
pip install mankey_stats
```

From Anaconda:

```
conda install -c conda-forge mankey_stats
```

Or simply clone it:

```
git clone https://github.com/mankey_stats/mankey_stats.git
```

## Example Usage

```python
>>> import pandas as pd
>>> from mankey_stats.ordinal_encoder as ordinal_encoder

>>> data = {'type':  ['bad', 'average', 'good', 'very good', 'excellent'],
            'level': [1, 2, 3, 4, 5]
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
>>> ordinal_encoding = Oridnal_encoder()
>>> data_encoded = ordinal_encoding(data)
>>> print(data_encoded)
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

To build the documentation make sure you have the dependencies installed: from the root directory: ``pip install -r docs/requirements.txt``.

Now you can build the docs using: ``sphinx-build -b html docs build``


## License

MIT

