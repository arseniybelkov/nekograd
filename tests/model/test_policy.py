import numpy as np
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from nekograd.model.policy import Multiply, Switch


def test_multiply():
    multiply_policy = Multiply({1: 0.1, 3: 0.1})
    lr = 2
    lrs = [multiply_policy(epoch) * lr for epoch in range(5)]
    assert np.allclose(lrs, [2, 0.2, 0.2, 0.02, 0.02])

    f = nn.Linear(2, 2)
    optimizer = SGD(f.parameters(), lr=lr)
    scheduler = LambdaLR(optimizer, lr_lambda=Multiply({1: 0.1, 3: 0.1}))
    lrs = []
    for _ in range(5):
        lrs.extend(scheduler.get_last_lr())
        optimizer.step()
        scheduler.step()

    assert np.allclose(lrs, [2, 0.2, 0.2, 0.02, 0.02])


def test_switch():
    lr = 2
    switch_policy = Switch(lambda i: np.cos(i / np.pi), lr_init=lr)
    lrs = [switch_policy(epoch) * lr for epoch in range(5)]
    assert np.allclose(lrs, [np.cos(i / np.pi) for i in range(5)])

    f = nn.Linear(2, 2)
    optimizer = SGD(f.parameters(), lr=lr)
    scheduler = LambdaLR(
        optimizer, lr_lambda=Switch(lambda i: np.cos(i / np.pi), lr_init=lr)
    )
    lrs = []
    for _ in range(5):
        lrs.extend(scheduler.get_last_lr())
        optimizer.step()
        scheduler.step()

    assert np.allclose(lrs, [np.cos(i / np.pi) for i in range(5)])
