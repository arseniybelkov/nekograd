import os

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score

from nekograd.metrics.binary import dice_score, precision, recall
from nekograd.metrics.utils import argmax
from nekograd.model import CoreModel
from nekograd.model.commands import convert_to_aggregated


@pytest.fixture
def architecture():
    conv_block = lambda c_in, c_out: nn.Sequential(
        nn.Conv2d(c_in, c_out, 3), nn.BatchNorm2d(c_out), nn.ReLU()
    )
    return nn.Sequential(
        conv_block(1, 16),
        conv_block(16, 32),
        nn.MaxPool2d(2, 2),
        conv_block(32, 32),
        conv_block(32, 16),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(16, 10),
    )


def test_core_model(mnist_datamodule, architecture):
    criterion = torch.nn.CrossEntropyLoss()
    metrics = {"accuracy": argmax(1)(accuracy_score)}

    class Model(CoreModel):
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), 1e-3)
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lambda epoch: 1
                ),
                "name": "lr_scheduler",
                "interval": "epoch",
            }

            return [optimizer], [lr_scheduler]

    model = Model(
        architecture=architecture,
        criterion=criterion,
        activation=nn.Softmax(1),
        metrics=metrics,
    )

    device = "gpu" if torch.cuda.is_available() else "cpu"
    print(device)

    trainer = pl.Trainer(
        accelerator=device,
        max_epochs=5,
        logger=TensorBoardLogger(
            save_dir=os.getcwd(), version="test_model", name="lightning_logs"
        ),
    )

    trainer.fit(model, datamodule=mnist_datamodule)
    test_metrics = trainer.test(model, datamodule=mnist_datamodule)[0]

    assert all(map(lambda v: v > 0.9, test_metrics.values()))


def test_optimizer_init(mnist_datamodule, architecture):
    criterion = torch.nn.CrossEntropyLoss()
    metrics = {"accuracy": argmax(1)(accuracy_score)}

    optimizer = torch.optim.Adam(architecture.parameters(), lr=1e-3)

    model = CoreModel(
        architecture=architecture,
        criterion=criterion,
        activation=nn.Softmax(1),
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1),
    )

    device = "gpu" if torch.cuda.is_available() else "cpu"
    print(device)

    trainer = pl.Trainer(
        accelerator=device,
        max_epochs=5,
        logger=TensorBoardLogger(
            save_dir=os.getcwd(), version="test_model", name="lightning_logs"
        ),
    )

    trainer.fit(model, datamodule=mnist_datamodule)
    test_metrics = trainer.test(model, datamodule=mnist_datamodule)[0]
    assert all(map(lambda v: v > 0.9, test_metrics.values()))
    assert all(map(lambda v: v <= 1, test_metrics.values()))


def test_segmentation(mnist_datamodule):
    criterion = torch.nn.BCEWithLogitsLoss()
    metrics = {
        "precision": convert_to_aggregated(lambda y, x: precision(y > 0.5, x > 0.5)),
        "recall": convert_to_aggregated(lambda y, x: recall(y > 0.5, x > 0.5)),
        "dice_score": convert_to_aggregated(lambda y, x: dice_score(y > 0.5, x > 0.5)),
    }

    def conv_block(i, o, padding: int = 0, T: bool = False):
        conv = nn.Conv2d(i, o, kernel_size=3, padding=padding, bias=False)
        if T:
            conv = nn.ConvTranspose2d(i, o, kernel_size=3, padding=padding, bias=False)
        return nn.Sequential(conv, nn.BatchNorm2d(o), nn.ReLU())

    architecture = nn.Sequential(
        conv_block(1, 8, padding=1),
        conv_block(8, 16, padding=1),
        nn.MaxPool2d(2, 2),
        conv_block(16, 32, padding=1),
        nn.MaxPool2d(2, 2),
        conv_block(32, 32, padding=1),
        nn.Upsample(scale_factor=2),
        conv_block(32, 16, padding=1, T=True),
        nn.Upsample(scale_factor=2),
        conv_block(16, 8, padding=1, T=True),
        conv_block(8, 1, padding=1, T=True),
    )

    optimizer = torch.optim.Adam(architecture.parameters(), lr=1e-3)

    class SegmentationModel(CoreModel):
        def on_after_batch_transfer(self, batch, dataloader_idx: int = 0):
            dtype = batch[0].dtype
            return (batch[0] > 0).type(dtype), (batch[0] > 0).type(dtype)

    model = SegmentationModel(
        architecture=architecture,
        criterion=criterion,
        activation=nn.Sigmoid(),
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1),
    )
    device = "cpu"
    print(device)

    trainer = pl.Trainer(
        accelerator=device,
        max_epochs=5,
        logger=TensorBoardLogger(
            save_dir=os.getcwd(), version="test_model", name="lightning_logs"
        ),
    )

    trainer.fit(model, datamodule=mnist_datamodule)
    test_metrics = trainer.test(model, datamodule=mnist_datamodule)[0]
    assert all(map(lambda v: v > 0.85, test_metrics.values()))
    assert all(map(lambda v: v <= 1, test_metrics.values()))
