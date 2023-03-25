from abc import ABCMeta
from typing import List, Dict, Union, Tuple

import numpy as np
import pytorch_lightning as pl
import torch

STEP_OUTPUT = Dict[str, Union[np.ndarray, float]]
EPOCH_OUTPUT = List[STEP_OUTPUT]


class BaseModel(pl.LightningModule, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def on_train_epoch_start(self) -> None:
        self.training_step_outputs.clear()

    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs.clear()

    def on_test_epoch_start(self) -> None:
        self.test_step_outputs.clear()

    def on_train_batch_end(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Tuple[torch.Tensor, ...],
        batch_idx: int,
    ) -> None:
        self.training_step_outputs.append(outputs)

    def on_validation_batch_end(
        self,
        outputs: STEP_OUTPUT,
        batch: Tuple[torch.Tensor, ...],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.validation_step_outputs.append(outputs)

    def on_test_batch_end(
        self,
        outputs: STEP_OUTPUT,
        batch: Tuple[torch.Tensor, ...],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.test_step_outputs.append(outputs)
