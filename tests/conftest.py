from pathlib import Path

import numpy as np
import pytest
import torchvision


@pytest.fixture
def test_root():
    return Path(__file__).resolve().parent


@pytest.fixture
def mnist(test_root):
    return torchvision.datasets.MNIST(test_root, train=False, download=True)
