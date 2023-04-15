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

    assert np.allclose(accuracy(x, y), accuracy_score(x, y))
    assert np.allclose(recall(x, y), recall_score(x, y))
    assert np.allclose(precision(x, y), precision_score(x, y))
    assert np.allclose(f1_score(x, y), f1_score_skl(x, y))


def test_segmentation():
    x = np.random.randn(5, 5) > 0
    y = np.random.randn(5, 5) > 0

    assert np.allclose(accuracy(x, y), ravel(accuracy_score)(x, y))
    assert np.allclose(recall(x, y), ravel(recall_score)(x, y))
    assert np.allclose(precision(x, y), ravel(precision_score)(x, y))
    assert np.allclose(f1_score(x, y), ravel(f1_score_skl)(x, y))
    assert np.allclose(dice_score(x, y), ravel(f1_score_skl)(x, y))


def test_aggregated_metrics():
    x = [np.random.randn(5, 5) > 0, np.random.randn(5, 5) > 0]
    y = [np.random.randn(5, 5) > 0, np.random.randn(5, 5) > 0]

    agg_accuracy = convert_to_aggregated(accuracy)
    agg_recall = convert_to_aggregated(recall)
    agg_precision = convert_to_aggregated(precision)
    agg_f1_score = convert_to_aggregated(f1_score)
    agg_dice_score = convert_to_aggregated(dice_score)

    aggregate = lambda ys, xs, m: np.mean([m(x, y) for y, x in zip_equal(y, x)])

    assert np.allclose(agg_accuracy(x, y), aggregate(x, y, ravel(accuracy_score)))
    assert np.allclose(agg_recall(x, y), aggregate(x, y, ravel(recall_score)))
    assert np.allclose(agg_precision(x, y), aggregate(x, y, ravel(precision_score)))
    assert np.allclose(agg_f1_score(x, y), aggregate(x, y, ravel(f1_score_skl)))
    assert np.allclose(agg_dice_score(x, y), aggregate(x, y, ravel(f1_score_skl)))
