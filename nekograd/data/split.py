from typing import Any, Hashable, Sequence, Tuple, Union

from cytoolz import get
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

__all__ = ["train_val_test_split", "k_fold", "k_fold_single_test"]


def train_val_test_split(
    ids: Sequence,
    val_size: Union[int, float],
    test_size: Union[int, float],
    stratify: Union[Sequence, None] = None,
    shuffle: bool = True,
    random_state: Any = 42,
) -> Tuple[Tuple[Hashable, ...], ...]:
    kwargs = dict(
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
        stratify=stratify,
    )

    if stratify is None:
        train_val, test = train_test_split(ids, **kwargs)
        train_val_stratify = None
    else:
        train_val, train_val_stratify, test, _ = train_test_split(
            ids, stratify, **kwargs
        )

    train, val = train_test_split(
        train_val,
        test_size=val_size,
        shuffle=shuffle,
        stratify=train_val_stratify,
        random_state=random_state,
    )

    return tuple(map(tuple, (train, val, test)))


def k_fold(
    ids: Sequence,
    val_size: Union[int, float],
    n_splits: int = 3,
    stratify: Union[Sequence, None] = None,
    shuffle: bool = True,
    random_state: Any = 42,
) -> Tuple[Tuple[Tuple[Hashable, ...], ...], ...]:
    """
    Stratified K-Fold Cross-Validation
    Parameters
    ----------
    ids : Sequence
        Sequence of dataset instances identifiers.
    val_size : float or int
        Part (or number if int) of ids to be used as validation
    stratify : Sequence or None
        If not None, should be sequence of values for stratifiction.
    shuffle : bool
        Whether to shuffle ids, default = True.
    n_splits : int
        Number of cross-validation folds
    random_state

    Returns
    -------
    split : Tuple
        n_splits of train, validation, test splits
        (..., (train_i, val_i, test_i), ...).
        Union of all test_i == ids
    """
    k_fold_class = KFold if stratify is None else StratifiedKFold

    kf = k_fold_class(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    split = []
    for train_val, test in kf.split(ids, stratify):
        _stratify = extract(train_val, stratify) if stratify is not None else None
        train, val = train_test_split(
            extract(train_val, ids),
            test_size=val_size,
            stratify=_stratify,
            shuffle=shuffle,
            random_state=random_state,
        )
        split.append((tuple(train), tuple(val), extract(test, ids)))

    return tuple(split)


def k_fold_single_test(
    ids: Sequence,
    test_size: Union[int, float],
    n_splits: int = 3,
    stratify: Union[Sequence, None] = None,
    shuffle: bool = True,
    random_state: Any = 42,
) -> Tuple[Tuple[Tuple[Hashable, ...], ...], ...]:
    """
    Stratified K-Fold Cross-Validation with single test
    Parameters
    ----------
    ids : Sequence
        Sequence of dataset instances identifiers.
    test_size : float or int
        Part (or number if int) of ids to be used as test.
    stratify : Sequence or None
        If not None, should be sequence of values for stratifiction.
    shuffle : bool
        Whether to shuffle ids, default = True
    n_splits : int
        Number of cross-validation folds
    random_state

    Returns
    -------
    split : Tuple
        n_splits of train, validation, test splits
        (..., (train_i, val_i, test), ...).
        Test is the same for all folds
    """

    k_fold_class = KFold if stratify is None else StratifiedKFold
    kwargs = dict(
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
        stratify=stratify,
    )

    if stratify is None:
        train_val, test = train_test_split(ids, **kwargs)
        train_val_stratify = None
    else:
        train_val, train_val_stratify, test, _ = train_test_split(
            ids, stratify, **kwargs
        )

    kf = k_fold_class(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    split = []
    for train, val in kf.split(train_val, train_val_stratify):
        split.append((extract(train, train_val), extract(val, train_val), tuple(test)))

    return tuple(split)


def extract(ids, sequence) -> Tuple:
    return tuple(get(list(ids), sequence))
