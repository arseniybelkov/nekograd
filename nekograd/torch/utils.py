from typing import Dict, Union, Sequence

import numpy as np
import torch


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