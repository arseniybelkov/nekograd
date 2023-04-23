from abc import ABCMeta
from typing import Dict, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch

from ..torch.utils import np2tensor, to_np

TRAIN_STEP_OUTPUT = Dict[str, Union[torch.Tensor, np.ndarray]]
VAL_STEP_OUTPUT = TEST_STEP_OUTPUT = Tuple[np.ndarray, ...]
STEP_OUTPUT = Union[TRAIN_STEP_OUTPUT, VAL_STEP_OUTPUT]
EPOCH_OUTPUT = List[STEP_OUTPUT]


class BaseModel(pl.LightningModule, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def on_before_batch_transfer(
        self, batch: Tuple[np.ndarray, ...], dataloader_idx: int
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(map(np2tensor, batch))

    def on_train_epoch_start(self) -> None:
        self.training_step_outputs.clear()

    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs.clear()

    def on_test_epoch_start(self) -> None:
        self.test_step_outputs.clear()

    def on_train_batch_end(
        self,
        outputs: TRAIN_STEP_OUTPUT,
        batch: Tuple[torch.Tensor, ...],
        batch_idx: int,
    ) -> None:
        self.training_step_outputs.append(to_np(outputs))

    def on_validation_batch_end(
        self,
        outputs: VAL_STEP_OUTPUT,
        batch: Tuple[torch.Tensor, ...],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.validation_step_outputs.append(outputs)

    def on_test_batch_end(
        self,
        outputs: TEST_STEP_OUTPUT,
        batch: Tuple[torch.Tensor, ...],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.test_step_outputs.append(outputs)
