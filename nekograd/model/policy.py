from typing import Dict
from abc import ABC, abstractmethod
from functools import reduce


class Policy(ABC):
    def __call__(self, epoch: int):
        return self.epoch2value(epoch)

    @abstractmethod
    def epoch2value(self, epoch: int):
        pass


class Switch(Policy):
    """
    Useful with torch.optim.LambdaLR
    lr_policy = Switch({1: 0.1, 3: 0.1})
    scheduler = torch.optim.LambdaLR(optimizer, lambda_lr=lr_policy, lr_init=1)
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
