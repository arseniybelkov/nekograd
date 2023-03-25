import math
from typing import List, Sequence

from sklearn.model_selection import StratifiedKFold, train_test_split


def train_val_test_split(
    ids: Sequence, qval: float = 0.05, qtest: float = 0.1, random_state=42
) -> List[List]:
    if qval + qtest >= 1:
        raise ValueError(f"No instances for train fold, {qval=}, {qtest=}")
    tr, test_val = train_test_split(
        ids, test_size=qval + qtest, random_state=random_state
    )
    return [
        tr,
        *train_test_split(
            test_val, test_size=qtest / (qval + qtest), random_state=random_state
        ),
    ]


def stratified_k_fold(
    ids: Sequence,
    target: Sequence,
    n_splits: int = 3,
    qval: float = 0.2,
    random_state=42,
) -> List[List[List]]:
    """
    Stratified K-Fold Cross-Validation
    Parameters
    ----------
    ids : Sequence
        Sequence of dataset instances identifiers.
    target : Sequence
        Sequence of target values.
    n_splits : int
        Number of cross-validation folds
    qval : float
        Part of data to be moved to validation sets (must be 0 < qtest < 1)
    random_state

    Returns
    -------
    split : List[List[List]]
        n_splits of train, validation, test splits
        [..., [train_i, val_i, test_i], ...].
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    split = []
    for train_val, test in skf.split(ids, target):
        train_val_skf = StratifiedKFold(round(1 / qval))
        _target = [target[i] for i in train_val]
        _train, _val = next(train_val_skf.split(train_val, _target))
        train = [train_val[i] for i in _train]
        val = [train_val[i] for i in _val]
        split.append(
            [[ids[i] for i in train], [ids[i] for i in val], [ids[i] for i in test]]
        )

    return split


def stratified_k_fold_single_test(
    ids: Sequence,
    target: Sequence,
    n_splits: int = 3,
    qtest: float = 0.2,
    random_state=42,
) -> List[List[List]]:
    """
    Stratified K-Fold Cross-Validation with single test
    Parameters
    ----------
    ids : Sequence
        Sequence of dataset instances identifiers.
    target : Sequence
        Sequence of target values.
    qtest : float
        Part of data to be moved to test set (must be 0 < qtest < 1)
    n_splits : int
        Number of cross-validation folds
    random_state

    Returns
    -------
    split : List[List[List]]
        n_splits of cross-validation folds with single test for them
        [..., [train_i, val_i, test], ...].
    """
    if not 0 < qtest < 1:
        raise ValueError(f"Expected qtest to be in range (0, 1), got {qtest}")
    qtest = round(1 / qtest)
    _ids, _test = next(
        StratifiedKFold(qtest, shuffle=True, random_state=random_state).split(
            ids, target
        )
    )

    ids = [ids[i] for i in _ids]
    target = [target[i] for i in _ids]
    test = [ids[i] for i in _test]

    split = []
    for train, val in StratifiedKFold(
        n_splits, shuffle=True, random_state=random_state
    ).split(ids, target):
        split.append([[ids[i] for i in train], [ids[i] for i in val], test])
    return split
