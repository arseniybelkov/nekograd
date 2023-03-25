from itertools import chain
from typing import Dict, Sequence, Any, List

import numpy as np
import torch
import torch.nn as nn


def switch_grad(*models: nn.Module, mode: bool = False) -> None:
    if not isinstance(mode, bool):
        raise ValueError(f"Expected `mode` to be True or False, got {mode}")

    for p in chain.from_iterable(m.parameters() for m in models):
        p.requires_grad = mode


def tensor2np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def tensor_sequence2np(x: Sequence[torch.Tensor]) -> List[np.ndarray]:
    return [tensor2np(t) for t in x]


def tensor_dict2np(x: Dict[Any, torch.Tensor]) -> Dict[Any, np.ndarray]:
    return {k: tensor2np(v) for k, v in x.items()}


def to_np(x):
    if isinstance(x, torch.Tensor):
        return tensor2np(x)
    elif isinstance(x, Dict):
        return tensor_dict2np(x)
    elif isinstance(x, Sequence):
        return tensor_sequence2np(x)
    else:
        return np.asarray(x)
