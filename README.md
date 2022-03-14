# Mankey Stats

![PythonVersion](https://img.shields.io/badge/python-3.6%20|3.7%20|%203.8%20|%203.9-success)
[![License https://github.com/dBlueG/mankey_stats/blob/main/LICENSE.md](https://img.shields.io/badge/license-MIT-success.svg)](https://github.com/dBlueG/mankey_stats/blob/main/LICENSE.md)
[![PyPI version](https://badge.fury.io/py/feature-engine.svg)](https://badge.fury.io/py/feature-engine)
[![Documentation Status https://mankey-stats.readthedocs.io/en/main/genindex.html](https://readthedocs.org/projects/feature-engine/badge/?version=latest)](https://mankey-stats.readthedocs.io/en/main/genindex.html)
[![Downloads](https://pepy.tech/badge/feature-engine)](https://pepy.tech/project/feature-engine)
[![Downloads](https://pepy.tech/badge/feature-engine/month)](https://pepy.tech/project/feature-engine)

![alt text](https://www.pngfind.com/pngs/m/607-6079786_056-mankey-mankey-pokemon-hd-png-download.png)


Mankey_stats is a Python library that allows the user to quickly and efficiently perform data preparation techniques to transform the datasets for ML modeling, this is done through the utilization of Scikit-learn's fit() and transform() methods.


## Documentation

* [Documentation](https://mankey-stats.readthedocs.io/en/main/#)


## Current Feature-engine's transformers include functionality for:

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

### Feature Selection:
 * 

### Datetime
 * DatetimeFeatures
 
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
>>> from mankey_stats.plot_charts as Plot_mankey

>>> data = {'var_A': ['A'] * 10 + ['B'] * 10 + ['C'] * 2 + ['D'] * 1}
>>> data = pd.DataFrame(data)
>>> data['var_A'].value_counts()
```

```
Out[1]:
A    10
B    10
C     2
D     1
Name: var_A, dtype: int64
```
    
```python 
>>> rare_encoder = RareLabelEncoder(tol=0.10, n_categories=3)
>>> data_encoded = rare_encoder.fit_transform(data)
>>> data_encoded['var_A'].value_counts()
```

```
Out[2]:
A       10
B       10
Rare     3
Name: var_A, dtype: int64
```

Find more examples in our [Jupyter Notebook Gallery](https://nbviewer.org/github/feature-engine/feature-engine-examples/tree/main/) 
or in the [documentation](http://feature-engine.readthedocs.io).

## Contribute

Details about how to contribute can be found in the [Contribute Page](https://feature-engine.readthedocs.io/en/latest/contribute/index.html)

Briefly:

- Fork the repo
- Clone your fork into your local computer: ``git clone https://github.com/<YOURUSERNAME>/feature_engine.git``
- navigate into the repo folder ``cd feature_engine``
- Install Feature-engine as a developer: ``pip install -e .``
- Optional: Create and activate a virtual environment with any tool of choice
- Install Feature-engine dependencies: ``pip install -r requirements.txt`` and ``pip install -r test_requirements.txt``
- Create a feature branch with a meaningful name for your feature: ``git checkout -b myfeaturebranch``
- Develop your feature, tests and documentation
- Make sure the tests pass
- Make a PR

Thank you!!


### Documentation

Feature-engine documentation is built using [Sphinx](https://www.sphinx-doc.org) and is hosted on [Read the Docs](https://readthedocs.org/).

To build the documentation make sure you have the dependencies installed: from the root directory: ``pip install -r docs/requirements.txt``.

Now you can build the docs using: ``sphinx-build -b html docs build``


## License

BSD 3-Clause

## Donate

[Sponsor us](https://github.com/sponsors/solegalli) to support her continue expanding 
Feature-engine.
