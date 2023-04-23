from functools import wraps
from pathlib import Path
from typing import Callable, Dict, Sequence, Union

import numpy as np
import pytorch_lightning as pl
from more_itertools import zip_equal

PathLike = Union[Path, str]


def convert_to_aggregated(metric: Callable, aggregate_fn: Callable = np.mean):
    @wraps(metric)
    def wrapper(ys: np.ndarray, xs: np.ndarray):
        return compute_aggregated_metrics(ys, xs, {"m": metric}, aggregate_fn)["m"]

    return wrapper


def compute_metrics(
    y: np.ndarray, x: np.ndarray, metrics: Dict[str, Callable]
) -> Dict[str, np.ndarray]:
    return {name: metric(y, x) for name, metric in metrics.items()}


def compute_individual_metrics(
    ys: Sequence[np.ndarray], xs: Sequence[np.ndarray], metrics: Dict[str, Callable]
) -> Dict[str, np.ndarray]:
    return {
        name: np.asarray([metric(y, x) for y, x in zip_equal(ys, xs)])
        for name, metric in metrics.items()
    }


def compute_aggregated_metrics(
    ys: np.ndarray,
    xs: np.ndarray,
    metrics: Dict[str, Callable],
    aggregate_fn: Callable = np.mean,
) -> Dict[str, float]:
    return {
        name: aggregate_fn(metric)
        for name, metric in compute_individual_metrics(ys, xs, metrics).items()
    }


def predict_to_dir(
    trainer: pl.Trainer,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    dir_path: PathLike,
) -> None:
    # TODO
    pass
