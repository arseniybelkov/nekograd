from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .base import BaseModel, STEP_OUTPUT, EPOCH_OUTPUT
from .commands import compute_metrics
from ..torch.utils import to_np


class CoreModel(BaseModel):
    def __init__(
        self,
        architecture: nn.Module,
        criterion: Callable,
        metrics: Dict[str, Callable],
        activation: Callable = nn.Identity(),
        n_targets: int = 1,
    ):
        super().__init__()
        self.architecture = architecture
        self.metrics = metrics
        self.criterion = criterion
        self.activation = activation
        if n_targets < 1:
            raise ValueError(f"Expected n_targets to be at least 1, got {n_targets}")
        self._nt = n_targets

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        optimizer = torch.optim.Adam(self.parameters(), 1e-4)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1),
            "name": "lr_scheduler",
            "interval": "epoch",
        }

        return [optimizer], [lr_scheduler]

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        return self.architecture(*xs)

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        x, y = batch[: -self._nt], batch[-self._nt :]
        loss = self.criterion(self(*x), *y)
        if isinstance(loss, torch.Tensor):
            return {"loss": loss}
        elif isinstance(loss, dict):
            return loss if "loss" in loss else {"loss": sum(loss.values()), **loss}
        else:
            raise ValueError(f"Expected `loss` to be dict or tensor, got {type(loss)}")

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> STEP_OUTPUT:
        preds, targets = self.inference_step(batch[0]), to_np(batch[1])
        return compute_metrics(targets, preds, self.metrics)

    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> STEP_OUTPUT:
        preds, targets = self.inference_step(batch[0]), to_np(batch[1])
        return compute_metrics(targets, preds, self.metrics)

    # TODO
    def predict_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int, **kwargs
    ) -> STEP_OUTPUT:
        return self.inference_step(batch[0])

    def inference_step(self, x: torch.Tensor) -> np.ndarray:
        return to_np(self.activation(self(x)))

    def on_train_epoch_end(self) -> None:
        self.log_metrics_on_epoch_end(self.training_step_outputs, "train")

    def on_validation_epoch_end(self) -> None:
        self.log_metrics_on_epoch_end(self.validation_step_outputs, "val")

    def on_test_epoch_end(self) -> None:
        self.log_metrics_on_epoch_end(self.test_step_outputs, "test")

    def log_metrics_on_epoch_end(self, outputs: EPOCH_OUTPUT, prefix: str = "") -> Dict[str, float]:
        if prefix:
            prefix += "/"
        logs = {}
        for k in outputs[0].keys():
            logs.update({prefix + k: np.stack([to_np(o[k]) for o in outputs])})
        logs = {k: np.mean(log) for k, log in logs.items()}
        self.log_dict(logs, prog_bar=True, on_epoch=True)
        return logs
