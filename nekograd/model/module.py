from typing import Callable, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn


class Module(pl.LightningModule):
    def __init__(self, architecture: nn.Module, criterions: List[Callable], metrics: List[Callable], n_targets: int=1):
        super().__init__()
        self.architecture = architecture
        self.metrics = metrics
        self.criterions = criterions
        if n_targets < 1:
            raise ValueError(f"Expected n_targets to be at least 1, got {n_targets}")
        self.nt = n_targets
        
    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        return self.architecture(*xs)
    
    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        pass
    
    def validation_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        pass