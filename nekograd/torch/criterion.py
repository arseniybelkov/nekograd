from inspect import isfunction
from typing import Callable, Dict, List, Union

import torch
from cytoolz.functoolz import juxt


class CriterionDict(torch.nn.Module):
    """
    Parameters
    ----------
    criterions: Callable
        critertions to compute losses.
    alphas: float, List[float]
        weights of criterions for total loss.
        Default: for any i, j alphas[i] == alphas[j] and sum(alphas) == 1 (uniform).
    loss_keys: str, List[str]
        names of criterions for logging.
        Default: uses the names of criterions.
    scale: bool
        if True then every loss is logged multiplied by its weight.
        Default: False
    """

    def __init__(
        self,
        *criterions: Callable,
        alphas: Union[float, List[float], None] = None,
        loss_keys: Union[str, List[str], None] = None,
        scale: bool = False,
    ) -> None:
        super().__init__()
        # If alphas is None -> all losses coeffs sum up to 1
        alphas = [1 / len(criterions)] * len(criterions) if alphas is None else alphas
        alphas = alphas if isinstance(alphas, (list, tuple)) else [alphas]

        if len(alphas) != len(criterions):
            if len(alphas) == 1:
                alphas = alphas * len(criterions)
            else:
                alphas.extend([1] * (len(criterions) - len(alphas)))

        assert len(alphas) == len(
            criterions
        ), f"Got {len(alphas)} alphas\
                                            and {len(criterions)} criterions,\
                                            but the numbers must be the same"
        if loss_keys is not None:
            loss_keys = (
                loss_keys if isinstance(loss_keys, (list, tuple)) else [loss_keys]
            )
        else:
            loss_keys = [
                c.__name__ if isfunction(c) else c.__class__.__name__
                for c in criterions
            ]

        assert len(set(loss_keys)) == len(
            loss_keys
        ), f"There are criterions with simmilar names: {loss_keys}"
        assert len(loss_keys) == len(
            criterions
        ), f"Got {len(loss_keys)} loss_keys\
                                            and {len(criterions)} criterions,\
                                            but the numbers must be the same"

        self.scale = scale
        self.alphas = alphas
        self.criterions = criterions
        self.loss_keys = loss_keys

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        loss = {"loss": 0}
        for _loss, alpha, loss_key in zip(
            juxt(self.criterions)(*args, **kwargs), self.alphas, self.loss_keys
        ):
            if isinstance(_loss, (int, float)):
                _loss = torch.tensor(_loss, dtype=torch.float32).cuda()
            loss["loss"] += _loss * alpha
            loss.update({loss_key: _loss * alpha if self.scale else _loss})
        return loss
