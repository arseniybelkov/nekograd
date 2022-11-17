from pl_bolts.datamodules import AsynchronousLoader as AsyncLoader
from torch.utils.data import DataLoader as _DataLoader


class DataLoader(AsyncLoader):
    def __init__(self, *args, q_size: int=10, **kwargs):
        dataloader = _DataLoader(*args, **kwargs)
        super().__init__(dataloader, q_size=q_size)