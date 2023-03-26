import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score

from nekograd.metrics.utils import argmax
from nekograd.model import CoreModel


def test_core_model(mnist_datamodule):
    criterion = torch.nn.CrossEntropyLoss()
    conv_block = lambda c_in, c_out: nn.Sequential(
        nn.Conv2d(c_in, c_out, 3), nn.BatchNorm2d(c_out), nn.ReLU()
    )
    architecture = nn.Sequential(
        conv_block(1, 16),
        conv_block(16, 32),
        nn.MaxPool2d(2, 2),
        conv_block(32, 32),
        conv_block(32, 16),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(16, 10),
    )

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
