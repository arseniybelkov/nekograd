from functools import partial
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from cytoolz import compose, keymap
from more_itertools import collapse, unzip
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ..torch.utils import to_np
from .base import EPOCH_OUTPUT, STEP_OUTPUT, BaseModel
from .commands import compute_metrics
from .utils import criterion_wrapper


class CoreModel(BaseModel):
    def __init__(
        self,
        architecture: nn.Module,
        criterion: Callable,
        metrics: Dict[str, Callable],
        activation: Callable = nn.Identity(),
        optimizer: Union[Optimizer, None] = None,
        lr_scheduler: Union[_LRScheduler, None] = None,
        n_targets: int = 1,
    ):
        super().__init__()
        self.architecture = architecture
        self.metrics = metrics
        self.criterion = criterion
        self.activation = activation
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        if n_targets < 1:
            raise ValueError(f"Expected n_targets to be at least 1, got {n_targets}")
        self._nt = n_targets
        self._wrap_criterion()

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        if self.optimizer is None or self.lr_scheduler is None:
            raise NotImplementedError(
                "If you do not pass optimizer and lr_scheduler to __init__ method, "
                "you must specify "
                "configure_optimizers method"
            )
        self.lr_scheduler = {
            "scheduler": self.lr_scheduler,
            "name": "lr_scheduler",
            "interval": "epoch",
        }
        return [self.optimizer], [self.lr_scheduler]

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        return self.architecture(*xs)

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        x, y = batch[: -self._nt], batch[-self._nt :]
        return self.wrapped_criterion(self(*x), *y)

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> STEP_OUTPUT:
        return self.inference_step(batch[0]), to_np(batch[1])

    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> STEP_OUTPUT:
        return self.inference_step(batch[0]), to_np(batch[1])

    # TODO
    def predict_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int, **kwargs
    ) -> STEP_OUTPUT:
        raise NotImplementedError("Currently in development")

    def inference_step(self, x: torch.Tensor) -> np.ndarray:
        return to_np(self.activation(self(x)))

    def on_train_epoch_end(self) -> None:
        self.log_metrics_on_epoch_end(self.training_step_outputs, "train")

    def on_validation_epoch_end(self) -> None:
        self.log_metrics_on_epoch_end(self.validation_step_outputs, "val")

    def on_test_epoch_end(self) -> None:
        self.log_metrics_on_epoch_end(self.test_step_outputs, "test")

    def log_metrics_on_epoch_end(
        self, outputs: EPOCH_OUTPUT, prefix: str = ""
    ) -> Dict[str, Union[np.ndarray, float]]:
        if prefix:
            prefix += "/"

        if isinstance(outputs[0], dict):
            logs = {}
            for k in filter(lambda s: not s.startswith("_"), outputs[0].keys()):
                logs[k] = np.mean(list(collapse(o[k] for o in outputs)))
        elif isinstance(outputs[0], (list, tuple)):
            x, y = map(compose(list, partial(collapse, levels=1)), unzip(outputs))
            logs = compute_metrics(y, x, self.metrics)
        else:
            raise TypeError(
                f"Unknown type of outputs: {type(outputs[0])}, "
                "expected Union[dict, list, tuple]"
            )

        logs = keymap(lambda k: prefix + k, logs)
        self.log_dict(logs, prog_bar=True, on_epoch=True)
        return logs

    def _wrap_criterion(self) -> None:
        self.wrapped_criterion = criterion_wrapper("loss")(self.criterion)
