from functools import wraps
from typing import Callable

import numpy as np


def check_type(_type: type):
    def decorator(metric: Callable):
        @wraps(metric)
        def wrapper(*args: np.ndarray):
            for i, arr in enumerate(args):
                if arr.dtype != _type:
                    raise ValueError(
                        f"Argument #{i + 1} is of type {arr.dtype}, expected {_type}"
                    )
            return metric(*args)

        return wrapper

    return decorator


check_bool = check_type(bool)
