from pathlib import Path
from typing import Callable, Dict, Union

import numpy as np
import pytorch_lightning as pl

PathLike = Union[Path, str]


def compute_metrics(
    y: np.ndarray, x: np.ndarray, metrics: Dict[str, Callable]
) -> Dict[str, float]:
    return {name: metric(y, x) for name, metric in metrics.items()}


def compute_individual_metrics(
    ys: np.ndarray, xs: np.ndarray, metrics: Dict[str, Callable]
) -> Dict[str, np.ndarray]:
    return {
        name: np.asarray([metric(y, x) for y, x in zip(ys, xs)])
        for name, metric in metrics.items()
    }


def predict_to_dir(
    trainer: pl.Trainer,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    dir_path: PathLike,
) -> None:
    # TODO
    pass
