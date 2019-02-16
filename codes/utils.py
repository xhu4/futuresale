# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains helper functions/constants/imported modules for
Coursera class __How to Win a Data Science Competition: Learn From Top Kagglers__
final project __Predict Future Sales__.

This file is intended to be used by %run in Jupyter Notebook (this allows editing of
this file and fast refresh of changes.) This in my opinion works better than defining
macros as recommended by the instructors.
"""

import os
from time import sleep
from math import sqrt
from gc import collect
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns

sns.set()

TARGET = "item_cnt_month"
DS = pd.Series(pd.date_range(start = '2013-01-01',end='2015-11-01', freq = 'MS'))


def load(filename):
    """Load function for different file types. Support csv, csv.gz, pickle, json."""
    ext = filename.split(".")[-1]
    if ext == "csv" or filename.endswith(".csv.gz"):
        return pd.read_csv(filename)
    elif ext == "pickle":
        with open(filename, 'rb') as f:
            return pickle.load(f)
        return pd.read_pickle(filename)
    elif ext == 'json':
        import json
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        raise ValueError("Cannot decide file type or filetype is not supported for %s."%filename)


def dump(obj, filename):
    """Dump function for different file types. Support csv, csv.gz, pickle, json."""
    # Make sure we don't overwrite original data
    assert not filename.startswith("../data/"), (
        "Trying to overwrite original data, file %s." % filename
    )
    if filename.endswith(".csv"):
        return obj.to_csv(filename, index=False)
    elif filename.endswith(".pickle"):
        with open(filename, "wb") as f:
            return pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif filename.endswith(".csv.gz"):
        return obj.to_csv(filename, compression="gzip", index=False)
    elif filename.endswith(".json"):
        import json
        with open(filename, 'w') as f:
            json.dump(obj, f)
    else:
        raise ValueError("Cannot decide file type from filename: %s." % filename)


def prepare_submission(pred, filename=None):
    """
    Prepare submission file from a prediction array pred. Will clip `pred`.
    `filename` can be a path to a file, a filename with extension, or a filename
    without extension. Default folder is '../submissions', default extension (filetype)
    is '.cvs.gz'.
    """
    res = pred.clip(0, 20)
    if isinstance(res, pd.Series):
        res = res.values
    ID = np.arange(res.size)
    res = pd.DataFrame({"ID": ID, TARGET: res})

    if filename is not None:
        if not filename.startswith("../submissions"):
            filename = os.path.join("../submissions", filename)
        if not filename.endswith(".csv.gz"):
            filename = filename + ".csv.gz"
        dump(res, filename)
    return res


def score(true, pred):
    """Compute rmse of true and pred"""
    if isinstance(true, pd.DataFrame):
        _true = true[TARGET]
    else:
        _true = true
    if isinstance(pred, pd.DataFrame):
        _pred = pred[TARGET]
    else:
        _pred = pred
    return sqrt(mean_squared_error(np.clip(_true, 0, 20), np.clip(_pred, 0, 20)))


def metric4xgb(preds, dtrain):
    labels = dtrain.get_label()
    scr = score(preds, labels)
    return 'clip_rmse', scr