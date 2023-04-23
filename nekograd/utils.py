import random
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch


def fix_seed(seed: Any = 42, backends: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = backends
