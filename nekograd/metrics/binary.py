# Copied from https://github.com/neuro-ml/deep_pipe


import numpy as np


def fraction(numerator, denominator, empty_val: float = 1):
    assert numerator <= denominator, f"{numerator}, {denominator}"
    return numerator / denominator if denominator != 0 else empty_val


def dice_score(x: np.ndarray, y: np.ndarray) -> float:
    return fraction(2 * np.sum(x & y), np.sum(x) + np.sum(y))


def sensitivity(y_true, y_pred):
    return fraction(np.sum(y_pred & y_true), np.sum(y_true))


def specificity(y_true, y_pred):
    tn = np.sum((~y_true) & (~y_pred))
    fp = np.sum(y_pred & (~y_true))
    return fraction(tn, tn + fp, empty_val=0)


def recall(y_true, y_pred):
    tp = np.count_nonzero(np.logical_and(y_pred, y_true))
    fn = np.count_nonzero(np.logical_and(~y_pred, y_true))

    return fraction(tp, tp + fn, 0)


def precision(y_true, y_pred):
    tp = np.count_nonzero(y_pred & y_true)
    fp = np.count_nonzero(y_pred & ~y_true)

    return fraction(tp, tp + fp, 0)


def f1_score(y_true, y_pred):
    pr = precision(y_true, y_pred)
    rc = recall(y_true, y_pred)
    return fraction(2 * pr * rc, pr + rc, 0)
