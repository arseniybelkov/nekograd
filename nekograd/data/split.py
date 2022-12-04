from typing import Sequence, List

from sklearn.model_selection import train_test_split, StratifiedKFold


def train_val_test_split(ids: Sequence, qval: float=0.05, qtest: float=0.1, random_state=42) -> List[List]:
    tr, test_val = train_test_split(ids, test_size=qval + qtest, random_state=random_state)
    return [tr, *train_test_split(test_val, test_size=qtest / (qval + qtest), random_state=random_state)]


def stratified_k_fold(ids: Sequence, target: Sequence, n_splits: int=3, random_state=42) -> List[List[List]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    split = []
    for train, val_test in skf.split(ids, target):
        val_test_skf = StratifiedKFold(2)
        _target = [target[i] for i in val_test]
        split.append([train, *next(val_test_skf.split(val_test, _target))])

    return split
