# Copied from https://github.com/neuro-ml/deep_pipe


import numpy as np

from .checks import check_bool


def fraction(numerator, denominator, empty_val: float = 1):
    assert numerator <= denominator, f"{numerator}, {denominator}"
    return numerator / denominator if denominator != 0 else empty_val


def accuracy(x: np.ndarray, y: np.ndarray) -> float:
    return fraction(np.sum(x == y), x.size)


@check_bool
def dice_score(x: np.ndarray, y: np.ndarray) -> float:
    return fraction(2 * np.sum(x & y), np.sum(x) + np.sum(y))


@check_bool
def sensitivity(y_true, y_pred) -> float:
    return fraction(np.sum(y_pred & y_true), np.sum(y_true))


@check_bool
def specificity(y_true, y_pred) -> float:
    tn = np.sum((~y_true) & (~y_pred))
    fp = np.sum(y_pred & (~y_true))
    return fraction(tn, tn + fp, empty_val=0)


@check_bool
def recall(y_true, y_pred) -> float:
    tp = np.count_nonzero(np.logical_and(y_pred, y_true))
    fn = np.count_nonzero(np.logical_and(~y_pred, y_true))

    return fraction(tp, tp + fn, 0)


@check_bool
def precision(y_true, y_pred) -> float:
    tp = np.count_nonzero(y_pred & y_true)
    fp = np.count_nonzero(y_pred & ~y_true)

    return fraction(tp, tp + fp, 0)


@check_bool
def f1_score(y_true, y_pred) -> float:
    pr = precision(y_true, y_pred)
    rc = recall(y_true, y_pred)
    return fraction(2 * pr * rc, pr + rc, 0)
