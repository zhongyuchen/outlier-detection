# Outlier Detection

![build status](https://img.shields.io/badge/build-passing-green.svg)
[![python version](https://img.shields.io/badge/python-3.6.8-purple.svg)](https://python.org)
[![Apache License](https://img.shields.io/badge/license-Apache2.0-blue.svg)](http://www.apache.org/licenses/)

Detect outliers with 3 methods: LOF, DBSCAN and SVDD

## Prerequisites

* Required packages can be installed with the following command:

```
pip install -r requirements.txt
```

## Data

* `consumption_data.xls` is provided. There are 4 columns with 940 entries. The first column denotes entry ID, which is ignored in detecting outliers. Therefore, the data entries are 3-dimensional.
* Get numpy array data with size `[940, 3]` with the following code (check out `dataset.py` for implementation):

```python
from dataset import get_dataset

data = get_dataset()
```

* Data visualization:

![data](pic/data.png)

## Methods

For detailed descriptions please see `report.pdf`.

### Density based method: LOF

* LOF: local outlier factor
* Check out `lof.py` for implementation.
* Result:

![lof](pic/lof.png)

### Cluster based method: DBSCAN

* Check out `dbscan.py` for implementation.
* Result:

![dbscan](pic/dbscan.png)

### Classification based method: SVDD

* SVDD: one class SVM clssification
* Check out `svdd.py` for implementation.
* Result with Gaussian kernel:

![rbf](pic/rbf.png)

* Result with linear kernel:

![linear](pic/linear.png)

## Author

Zhongyu Chen
