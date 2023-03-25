from typing import Callable, List, Union

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from .split import train_val_test_split
from .transforms import Augment, Preprocess

ID_T = Union[str, int]


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        preprocessing: List[Callable] = [],
        augmentation: List[Callable] = [],
        batch_size: int = 32,
        split: list = None,
        loader_params: dict = {},
        random_seed=42,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.loader_params = loader_params
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.split = split

    def setup(self, stage: str):
        if self.split is None:
            self.split = train_val_test_split(
                self.dataset.ids, random_state=self.random_seed
            )

        def get_split(split: list):
            return Preprocess(DataSplit(self.dataset, split), *self.preprocessing)

        self.train_data = get_split(self.split[0])
        self.val_data = get_split(self.split[1])
        self.test_data = get_split(self.split[2])

    def _create_loader(
        self, data: Dataset, bs: int, shuffle: bool = False, augm: bool = False
    ) -> DataLoader:
        _data = Augment(data, *self.augmentation) if augm else data
        return DataLoader(_data, batch_size=bs, **self.loader_params, shuffle=shuffle)

    def train_dataloader(self):
        return self._create_loader(
            self.train_data, self.batch_size, shuffle=True, augm=True
        )

    def val_dataloader(self):
        return self._create_loader(self.val_data, 1)

    def test_dataloader(self):
        return self._create_loader(self.test_data, 1)

    def predict_dataloader(self):
        return self._create_loader(self.test_data, 1)


class DataSplit(Dataset):
    def __init__(self, dataset: Dataset, split_ids: List[ID_T]):
        super().__init__()
        self.dataset = dataset
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: ID_T):
        return self.dataset[self.ids[idx]]

    def __repr__(self):
        return f"{self.__class__.__name__}(n_samples={self.__len__()})"

    def __str__(self):
        return self.__repr__()
