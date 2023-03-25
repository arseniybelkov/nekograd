from pathlib import Path

import pytest
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST


@pytest.fixture
def test_root():
    return Path(__file__).resolve().parent


@pytest.fixture
def mnist(test_root):
    return torchvision.datasets.MNIST(test_root, train=True, download=True)


@pytest.fixture
def mnist_datamodule(test_root):
    class MNISTDataModule(pl.LightningDataModule):
        def __init__(self, data_dir, val: float = 0.2, batch_size: int = 1024):
            super().__init__()
            self.data_dir = data_dir
            self.transform = transforms.Compose([transforms.ToTensor()])
            self.val = val
            self.batch_size = batch_size

        def prepare_data(self) -> None:
            MNIST(test_root, train=True, download=True)
            MNIST(test_root, train=False, download=True)

        def setup(self, stage: str = None):
            if stage == "fit":
                mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
                val_part = int(len(mnist_full) * self.val)
                self.mnist_train, self.mnist_val = random_split(
                    mnist_full, [len(mnist_full) - val_part, val_part]
                )
            elif stage == "test":
                self.mnist_test = MNIST(
                    self.data_dir, train=False, transform=self.transform
                )
            else:
                raise ValueError(f"Unknown stage: {stage}")

        def train_dataloader(self):
            return DataLoader(self.mnist_train, batch_size=self.batch_size)

        def val_dataloader(self):
            return DataLoader(self.mnist_val, batch_size=self.batch_size)

        def test_dataloader(self):
            return DataLoader(self.mnist_test, batch_size=self.batch_size)

    return MNISTDataModule(test_root)
