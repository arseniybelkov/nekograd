from functools import wraps
from typing import Callable, Sequence, Union

import numpy as np
from torch import Tensor

from ..torch.utils import to_np


ArrayLike = Union[np.ndarray, Tensor]


def ravel(metric):
    @wraps(metric)
    def wrapper(t: np.ndarray, p: np.ndarray, *args, **kwargs):
        if p.ndim == t.ndim:
            p = p.ravel().squeeze()
        else:
            p = p.reshape(p.shape[0], np.prod(p.shape[1:])).T.squeeze()
        return metric(t.ravel().squeeze(), p, *args, **kwargs)

    return wrapper


def threshold(th: float = 0.5):
    def decorator(metric):
        @wraps(metric)
        def wrapper(t: np.ndarray, p: np.ndarray, *args, **kwargs):
            return metric(t, p > th, *args, **kwargs)

        return wrapper

    return decorator


def to_numpy(metric):
    @wraps(metric)
    def wrapper(t: ArrayLike, p: ArrayLike, *args, **kwargs):
        return metric(to_np(t), to_np(p), *args, **kwargs)

    return wrapper


def squeeze(metric):
    @wraps(metric)
    def wrapper(t: np.ndarray, p: np.ndarray, *args, **kwargs):
        return metric(t.squeeze(), p.squeeze(), *args, **kwargs)

    return wrapper


def channels_first(metric):
    @wraps(metric)
    def wrapper(t: np.ndarray, p: np.ndarray, *args, **kwargs):
        transpose = lambda x: np.transpose(x, (-1, -3, -2))
        return metric(transpose(t), transpose(p), *args, **kwargs)

    return wrapper


def channels_last(metric):
    @wraps(metric)
    def wrapper(t: np.ndarray, p: np.ndarray, *args, **kwargs):
        transpose = lambda x: np.transpose(x, (-2, -3, -1))
        return metric(transpose(t), transpose(p), *args, **kwargs)

    return wrapper


def swap_args(metric):
    @wraps(metric)
    def wrapper(t: np.ndarray, p: np.ndarray, *args, **kwargs):
        return metric(p, t, *args, **kwargs)

    return wrapper


def batched(aggregate: Callable = np.mean):
    def decorator(metric):
        @wraps(metric)
        def wrapper(
            ts: Sequence[np.ndarray], ps: Sequence[np.ndarray], *args, **kwargs
        ):
            return aggregate([metric(t, p, *args, **kwargs) for t, p in zip(ts, ps)])

        return wrapper

    return decorator


def argmax(axis: int = 0):
    def decorator(metric):
        @wraps(metric)
        def wrapper(t: np.ndarray, p: np.ndarray, *args, **kwargs):
            return metric(t, p.argmax(axis), *args, **kwargs)

        return wrapper

    return decorator
