import numpy as np
from sklearn.metrics import (accuracy_score, log_loss, precision_score,
                             recall_score, roc_auc_score)

from .utils import ravel, threshold


@ravel
@threshold(0.5)
def accuracy(t: np.ndarray, p: np.ndarray, **kwargs):
    return accuracy_score(t, p, **kwargs)


@ravel
@threshold(0.5)
def recall(t: np.ndarray, p: np.ndarray, **kwargs):
    return recall_score(t, p, **kwargs)


@ravel
@threshold(0.5)
def precision(t: np.ndarray, p: np.ndarray, **kwargs):
    return precision_score(t, p, **kwargs)


@ravel
def roc_auc(t: np.ndarray, p: np.ndarray, **kwargs):
    return roc_auc_score(t, p, **kwargs)


@ravel
def bce_metric(t: np.ndarray, p: np.ndarray, **kwargs):
    return log_loss(t, p, **kwargs)