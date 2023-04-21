from functools import reduce
from itertools import combinations
from string import ascii_lowercase

import numpy as np
from more_itertools import collapse

from nekograd import k_fold_single_test, k_fold, train_val_test_split

ids = tuple(ascii_lowercase)
x = np.random.randn(len(ids))
y = np.random.binomial(len(ids), p=0.5)


def intersect(f1, f2):
    return bool(set(f1).intersection(f2))


def test_train_val_test_split():
    split = train_val_test_split(ids, 0.2, 0.2)
    assert all(map(lambda i: type(i) == type(ids[0]), collapse(split)))
    assert not reduce(lambda s1, s2: set(s1).intersection(s2), split)


def test_k_fold():
    folds = k_fold(ids, 0.2, 3)

    assert len(folds) == 3
    assert all(map(lambda f: len(f) == 3, folds))
    assert isinstance(folds, tuple)
    assert all(map(lambda f: isinstance(f, tuple), folds))
    assert all([all(map(lambda f: isinstance(f, tuple), fold)) for fold in folds])
    assert all(map(lambda i: type(i) == type(ids[0]), collapse(folds)))

    # Check uniqueness
    assert all([len(set(fold[-1])) == len(fold[-1]) for fold in folds])

    # Check empty intersection
    assert not any(
        map(lambda fs: intersect(fs[0][-1], fs[1][-1]), combinations(folds, 2))
    )
    # Check completeness
    test_ids_union = folds[0][-1] + folds[1][-1] + folds[2][-1]
    assert sorted(test_ids_union) == sorted(ids)


def test_k_fold_single_test():
    folds = k_fold_single_test(ids, 0.2, 3)

    assert len(folds) == 3
    assert all(map(lambda f: len(f) == 3, folds))
    assert isinstance(folds, tuple)
    assert all(map(lambda f: isinstance(f, tuple), folds))
    assert all([all(map(lambda f: isinstance(f, tuple), fold)) for fold in folds])
    assert sorted(folds[0][-1]) == sorted(folds[1][-1]) == sorted(folds[2][-1])
    assert all(map(lambda i: type(i) == type(ids[0]), collapse(folds)))

    # Check uniqueness
    assert all([len(set(fold[1])) == len(fold[1]) for fold in folds])
    assert not set(collapse([f[0] + f[1] for f in folds])).intersection(folds[0][-1])

    # Check empty intersection
    assert not any(
        map(lambda fs: intersect(fs[0][1], fs[1][1]), combinations(folds, 2))
    )
    # Check completeness
    val_ids_union = folds[0][1] + folds[1][1] + folds[2][1]
    assert sorted(val_ids_union) == sorted(set(ids) - set(folds[0][-1]))
