from typing import Callable, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from nekograd.torch.criterion import CriterionDict
from nekograd.torch.utils import tensor_dict2np


LOG_KWARGS = {"reduce_fx": np.mean, "on_epoch": True, "on_step": False}


class CoreModel(pl.LightningModule):
    def __init__(
        self,
        architecture: nn.Module,
        criterions: List[Callable],
        metrics: List[Callable],
        n_targets: int=1,
    ):
        super().__init__()
        self.architecture = architecture
        self.metrics = metrics
        self.criterion = CriterionDict(criterions)
        if n_targets < 1:
            raise ValueError(f"Expected n_targets to be at least 1, got {n_targets}")
        self._nt = n_targets

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        return self.architecture(*xs)

    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch[:-self._nt], batch[-self._nt:]
        p = self(*x)
        loss = self.criterion(p, *y)
        self.log_dict(tensor_dict2np(loss), **LOG_KWARGS)
        return loss["total"]

    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, metrics = self._evaluate(batch, batch_idx, key="val")
        self.log_dict({**{"val " + n: m for n, m in tensor_dict2np(loss).items()},
                       **metrics}, **LOG_KWARGS)
        return loss["total"]

    def test_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, metrics = self._evaluate(batch, batch_idx, key="test")
        self.log_dict({**{"test " + n: m for n, m in tensor_dict2np(loss).items()},
                       **metrics}, **LOG_KWARGS)
        return loss["total"]

    def predict_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self(*batch[:-self._nt])

    def _evaluate(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        x, y = batch[:-self._nt], batch[-self._nt:]
        p = self(*x)
        loss = self.criterion(p, *y)
        metrics = {n: m(tensor_dict2np(*y), tensor_dict2np(p)) for n, m in self.metrics}
        return loss, metrics
