from abc import ABC, abstractmethod
from typing import Callable, Dict


class Policy(ABC):
    def __call__(self, epoch: int):
        return self.epoch2value(epoch)

    @abstractmethod
    def epoch2value(self, epoch: int):
        pass


class Multiply(Policy):
    """
    Useful with torch.optim.lr_scheduler.LambdaLR
    lr_policy = Multiply({1: 0.1, 3: 0.1})
    optimizer = Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_policy)
    Suppose we have 6 epochs
    lr = {0: 1, 1: 0.1, 2: 0.1, 3: 0.01, 4: 0.01, 5: 0.01} # lr2epoch mapping
    """

    def __init__(self, mapping: Dict[int, float]):
        super().__init__()

        sorted_mapping = {k: v for k, v in sorted(mapping.items(), key=lambda i: i[0])}
        self.mapping = sorted_mapping
        self.factor = 1

    def epoch2value(self, epoch: int) -> float:
        self.factor *= self.mapping.get(epoch, 1)
        return self.factor


class Switch(Policy):
    """
    Useful with torch.optim.lr_scheduler.LambdaLR
    lr_policy = Switch(lr_init, lambda i: np.cos(i / np.pi))
    optimizer = Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_policy)
    On i-th epoch lr_i = cos(i / np.pi)
    """

    def __init__(self, epoch2lr: Callable[[int], float], lr_init: float):
        super().__init__()
        if lr_init == 0:
            raise ValueError(f"lr_init must not be equal to 0")
        self.lr_init = lr_init
        self.epoch2lr = epoch2lr

    def epoch2value(self, epoch: int) -> float:
        return self.epoch2lr(epoch) / self.lr_init
