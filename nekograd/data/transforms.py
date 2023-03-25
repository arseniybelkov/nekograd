from typing import Callable, Tuple, Union

import numpy as np
from torch.utils.data import Dataset


class Transform(Dataset):
    def __init__(self, dataset: Dataset, *funcs: Callable):
        self.dataset = dataset
        self.funcs = funcs

    def __len__(self) -> int:
        return len(self.dataset)

    def apply(self, *xs: np.ndarray) -> Tuple[np.ndarray]:
        for f in self.funcs:
            xs = f(*xs)
        return xs

    def __getitem__(self, idx: Union[str, int]):
        return self.apply(*self.dataset[idx])

    def __repr__(self):
        return f"{self.__class__.__name__}([{self.funcs}])"

    def __str__(self):
        return self.__repr__()


class Augment(Transform):
    pass


class Preprocess(Transform):
    # TODO
    # @cache (lru seems not legit)
    def apply(self, *xs: np.ndarray):
        return super().apply(*xs)
