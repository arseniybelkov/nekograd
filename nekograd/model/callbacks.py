from datetime import datetime
from typing import Dict, List

from more_itertools import windowed
from pytorch_lightning.callbacks import Callback


class TimeProfiler(Callback):
    def __init__(self, *keys: str):
        self._default_keys = (
            "Train Epoch"
            "Validation Epoch",
            "Total Batch Iter",
            "Avg Batch Iter",
            "Avg Train Step",
            "Avg Validation Step"
        )
        self._optional_keys = ("Avg Backward", "Avg Optimizer Step")

        allowed_keys = self._default_keys + self._optional_keys
        if sorted(set(keys).intersection(allowed_keys)) != sorted(keys):
            raise ValueError(f"TimeProfiler got unknown keys: {set(keys) - set(allowed_keys)}")

        self.keys = sorted(set(keys).union(self._default_keys))
        self.time_stamps: Dict[str, List[datetime]] = {}

    def log_time(self, key: str) -> None:
        if key in self.keys:
            key = f"{self.__class__.__name__}/" + key
            if key not in self.time_stamps:
                self.time_stamps[key] = [datetime.now()]
            else:
                self.time_stamps[key].append(datetime.now())

    def compute_time_delta(self) -> Dict[str, float]:
        def delta(t1, t2=None):
            if isinstance(t1, (list, tuple)):
                return (t1[1] - t1[0]).total_seconds()
            return (t2 - t1).total_seconds()

        deltas = {}
        for key, time_stamps in self.time_stamps.items():
            deltas[key] = list(map(delta, windowed(time_stamps, 2)))
            deltas[key] = sum(deltas[key]) / len(delta[key])

        return deltas

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.log_time("Avg Train Step")

    def on_train_batch_end(self, trainer, pl_nodule, outputs, batch, batch_idx):
        self.log_time("Avg Train Step")
        self.log_time("Avg Optimizer Step")

    def on_train_epoch_start(self, trainer, pl_module):
        self.log_time("Train Epoch")

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_time("Train Epoch")

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.log_time("Avg Validation Step")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.log_time("Avg Validation Step")

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_time("Validation Epoch")

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_time("Validation Epoch")

    def on_before_backward(self, trainer, pl_module, loss):
        self.log_time("Avg Backward")

    def on_after_backward(self, trainer, pl_module):
        self.log_time("Avg Backward")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        self.log_time("Avg Optimizer Step")

    def teardown(self, trainer, pl_module, stage: str):
        if stage == "validate":
            pl_module.log_dict(self.compute_time_delta(), on_step=True, on_epoch=True, prog_bar=False)
            self.time_stamps.clear()
