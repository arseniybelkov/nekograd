import numpy as np

from nekograd.model.policy import Switch


def test_switch():
    switch_policy = Switch({1: 0.1, 3: 0.1})
    lr = 2
    lrs = [switch_policy(epoch) * lr for epoch in range(5)]
    assert np.allclose(lrs, [2, 0.2, 0.2, 0.02, 0.02])

    switch_policy = Switch({3: 0.1, 1: 0.1})
    lrs = [switch_policy(epoch) * lr for epoch in range(5)]
    assert np.allclose(lrs, [2, 0.2, 0.2, 0.02, 0.02])
