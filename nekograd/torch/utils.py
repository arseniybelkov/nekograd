from itertools import chain
from typing import Dict, Sequence, Union

import numpy as np
import torch
import torch.nn as nn


def tensor2np(x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> np.ndarray:
    return x.detach().cpu().numpy()


def tensor_dict2np(x: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    return {k: tensor2np(v) for k, v in x.items()}


def sequence_tensor2np(x: Sequence[torch.Tensor]) -> Sequence[np.ndarray]:
    return np.asarray([tensor2np(e) for e in x])


def to_np(x: Sequence) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return tensor2np(x)
    return np.array(x)


def switch_grad(*models: nn.Module, mode: bool=False) -> None:
    if not isinstance(mode, bool):
        raise ValueError(f"Expected `mode` to be True or False, got {mode}")

    for p in chain.from_iterable(m.parameters() for m in models):
        p.requires_grad = mode
