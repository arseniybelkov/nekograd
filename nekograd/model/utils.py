from functools import wraps
from typing import Callable, Dict

import torch


def criterion_wrapper(loss_key: str = "loss"):
    def decorator(criterion: Callable):
        @wraps(criterion)
        def wrapper(*args, **kwargs) -> Dict[str, torch.Tensor]:
            loss = criterion(*args, **kwargs)
            if isinstance(loss, torch.Tensor):
                return {loss_key: loss}
            elif isinstance(loss, dict):
                return (
                    loss if loss_key in loss else {loss_key: sum(loss.values()), **loss}
                )
            else:
                raise ValueError(
                    f"Expected `loss` to be dict or tensor, got {type(loss)}"
                )

        return wrapper

    return decorator
