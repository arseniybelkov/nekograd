from functools import wraps

import numpy as np


def ravel(metric):
    @wraps(metric)
    def wrapper(t, p, *args, **kwargs):
        t = t.ravel()
        p = p.reshape(p.shape[-1], np.prod(p.shape[:-1]))
        return metric(t, p, *args, **kwargs)
    return wrapper


def threshold(th: float=0.5):
    def decorator(metric):
        @wraps(metric)
        def wrapper(t: np.ndarray, p: np.ndarray, *args, **kwargs):
            return metric(t, p > th,  *args, **kwargs)
        return wrapper
    return decorator