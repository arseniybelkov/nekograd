import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score as _f1_score
from sklearn.metrics import (log_loss, precision_score, recall_score,
                             roc_auc_score)

from .utils import ravel, threshold, to_numpy


@to_numpy
@ravel
@threshold(0.5)
def accuracy(t: np.ndarray, p: np.ndarray, *args, **kwargs):
    return accuracy_score(t, p, *args, **kwargs)


@to_numpy
@ravel
@threshold(0.5)
def recall(t: np.ndarray, p: np.ndarray, *args, **kwargs):
    return recall_score(t, p, *args, **kwargs)


@to_numpy
@ravel
@threshold(0.5)
def precision(t: np.ndarray, p: np.ndarray, *args, **kwargs):
    return precision_score(t, p, *args, **kwargs)


@to_numpy
@ravel
@threshold(0.5)
def f1_score(t: np.ndarray, p: np.ndarray, *args, **kwargs):
    return _f1_score(t, p, *args, **kwargs)


@to_numpy
@ravel
def roc_auc(t: np.ndarray, p: np.ndarray, *args, **kwargs):
    return roc_auc_score(t, p, *args, **kwargs)


@to_numpy
@ravel
def binary_cross_entropy(t: np.ndarray, p: np.ndarray, *args, **kwargs):
    return log_loss(t, p, *args, **kwargs)