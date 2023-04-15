import numpy as np
from more_itertools import zip_equal
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score as f1_score_skl

from nekograd.metrics.binary import accuracy, f1_score, dice_score, recall, precision
from nekograd.metrics.utils import ravel
from nekograd.model.commands import convert_to_aggregated


def test_classification():
    ones = np.asarray([1, 1, 1], dtype=bool)
    zeros = np.asarray([0, 0, 0], dtype=bool)
    x = np.asarray([0, 1, 1], dtype=bool)
    y = np.asarray([1, 0, 1], dtype=bool)

    assert np.allclose(accuracy(y, x), accuracy_score(y, x))
    assert np.allclose(recall(y, x), recall_score(y, x))
    assert np.allclose(precision(y, x), precision_score(y, x))
    assert np.allclose(f1_score(y, x), f1_score_skl(y, x))


def test_segmentation():
    x = np.random.randn(5, 5) > 0
    y = np.random.randn(5, 5) > 0

    assert np.allclose(accuracy(y, x), ravel(accuracy_score)(y, x))
    assert np.allclose(recall(y, x), ravel(recall_score)(y, x))
    assert np.allclose(precision(y, x), ravel(precision_score)(y, x))
    assert np.allclose(f1_score(y, x), ravel(f1_score_skl)(y, x))
    assert np.allclose(dice_score(y, x), ravel(f1_score_skl)(y, x))


def test_aggregated_metrics():
    x = [np.random.randn(5, 5) > 0, np.random.randn(5, 5) > 0]
    y = [np.random.randn(5, 5) > 0, np.random.randn(5, 5) > 0]

    agg_accuracy = convert_to_aggregated(accuracy)
    agg_recall = convert_to_aggregated(recall)
    agg_precision = convert_to_aggregated(precision)
    agg_f1_score = convert_to_aggregated(f1_score)
    agg_dice_score = convert_to_aggregated(dice_score)

    aggregate = lambda ys, xs, m: np.mean([m(y, x) for y, x in zip_equal(ys, xs)])

    assert np.allclose(agg_accuracy(y, x), aggregate(y, x, ravel(accuracy_score)))
    assert np.allclose(agg_recall(y, x), aggregate(y, x, ravel(recall_score)))
    assert np.allclose(agg_precision(y, x), aggregate(y, x, ravel(precision_score)))
    assert np.allclose(agg_f1_score(y, x), aggregate(y, x, ravel(f1_score_skl)))
    assert np.allclose(agg_dice_score(y, x), aggregate(y, x, ravel(f1_score_skl)))
