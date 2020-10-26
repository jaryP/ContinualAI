from abc import ABC, abstractmethod
from operator import itemgetter
from os import makedirs
from os.path import join, dirname, exists
from typing import Callable, Union, Tuple

import numpy as np
from datasets.utils import split_dataset


class UnsupervisedDataset(object):
    def __init__(self, x, train, test=None, dev=None, transformer: Callable = None, **kwargs):

        super().__init__(**kwargs)

        if dev is None:
            dev = []
        if test is None:
            test = []

        assert len(x) == sum(map(len, [train, test, dev]))

        self._x = x
        self.train_split, self.test_split, self.dev_split = train, test, dev

        self._split = 'train'
        self._current_split_idx = self.train_split

        self.transformer = transformer if transformer is not None else lambda z: z

        self.__initialized = True

    # def __setattr__(self, key, value):
    #     if key in ['train_split', 'test_split', 'dev_split', '_split', '_x', '__initialized'] \
    #             and hasattr(self, '__initialized'):
    #         raise ValueError()
    #
    #     super().__setattr__(key, value)

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            idx = itemgetter(*range(start, stop, step))(self._current_split_idx)
            return idx, [self.transformer(self._x[i]) for i in idx]
        elif isinstance(item, (tuple, list)):
            return item, [self.transformer(self._x[i]) for i in item]
        elif isinstance(item, int):
            return item, self.transformer(self._x[self._current_split_idx[item]])
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self._current_split_idx)

    @property
    def split(self):
        return self._split

    @property
    def x(self):
        return [self._x[i] for i in self._current_split_idx]

    @property
    def data(self):
        return self.x

    def train(self):
        self._split = 'train'
        self._current_split_idx = self.train_split

    def test(self):
        self._split = 'test'
        self._current_split_idx = self.test_split

    def dev(self):
        self._split = 'dev'
        self._current_split_idx = self.dev_split

    def all(self):
        self._split = 'all'
        self._current_split_idx = self.train_split + self.test_split + self.dev_split

    def preprocess(self, f: Callable):
        for i, x in enumerate(self._x):
            self._x[i] = f(x)

    def split_dataset(self, test_split: float = 0.2, dev_split: float = 0.0,
                      random_state: Union[np.random.RandomState, int] = None):

        assert test_split >= 0
        assert dev_split >= 0
        assert test_split + dev_split < 1

        self.train_split, self.test_split, self.dev_split = \
            split_dataset(self._x, test_split=test_split, dev_split=dev_split, random_state=random_state)


class SupervisedDataset(UnsupervisedDataset):
    def __init__(self, x, y, train, test=None, dev=None, transformer: Callable = None,
                 target_transformer: Callable = None, **kwargs):

        super().__init__(x, train, test, dev, transformer, **kwargs)

        self._y = y
        assert len(self._x) == len(self._y)

        self.target_transformer = target_transformer if target_transformer is not None else lambda z: z

        self.labels = sorted(list(set(y)))

    def __getitem__(self, item):
        item, x = super().__getitem__(item)

        if isinstance(item, (tuple, list)):
            y = [self.transformer(self._y[i]) for i in item]
        elif isinstance(item, int):
            y = self.transformer(self._y[item])
        else:
            raise NotImplementedError()

        return item, x, y

    # def __setattr__(self, key, value):
    #     if key in ['_y'] \
    #             and hasattr(self, '__initialized'):
    #         raise ValueError()
    #
    #     super().__setattr__(key, value)

    @property
    def y(self):
        return [self._y[i] for i in self._current_split_idx]

    @property
    def data(self):
        return self.x, self.y

    def preprocess_targets(self, f: Callable):
        for i, y in enumerate(self._y):
            self._y[i] = f(y)

    def split_dataset(self, test_split: float = 0.2, dev_split: float = 0.0, balanced_split: bool = True,
                      random_state: Union[np.random.RandomState, int] = None):

        assert test_split >= 0
        assert dev_split >= 0
        assert test_split + dev_split < 1

        self.train_split, self.test_split, self.dev_split = \
            split_dataset(self._y, balance_labels=True,
                          test_split=test_split, dev_split=dev_split, random_state=random_state)


class DownloadableDataset(ABC):

    def __init__(self, name, transformer: Callable = None,
                 download_if_missing: bool = True, data_folder: str = None, **kwargs):

        if data_folder is None:
            data_folder = join(dirname(__file__), 'downloaded_datasets', name)

        self.data_folder = data_folder
        self.name = name

        self.transformer = transformer if transformer is not None else lambda x: x

        if not exists(self.data_folder):
            if not download_if_missing:
                raise IOError("Data not found and `download_if_missing` is False")
            else:
                if not exists(self.data_folder):
                    makedirs(self.data_folder)

                print('Downloading dataset')
                self.download_dataset()

    @abstractmethod
    def download_dataset(self):
        raise NotImplementedError


class UnsupervisedDownloadableDataset(DownloadableDataset, UnsupervisedDataset, ABC):
    def __init__(self, name, download_if_missing: bool = True, data_folder: str = None,
                 transformer: Callable = None, target_transformer: Callable = None, **kwargs):

        super().__init__(name=name, transformer=transformer,
                         download_if_missing=download_if_missing, data_folder=data_folder)

        x, (train, test, dev) = self.load_dataset()

        super(DownloadableDataset, self).__init__(x=x, train=train, test=test, dev=dev,
                                                  transformer=transformer, target_transformer=target_transformer,
                                                  **kwargs)

    @abstractmethod
    def load_dataset(self) -> Tuple[list, Tuple[list, list, list]]:
        raise NotImplementedError


class SupervisedDownloadableDataset(DownloadableDataset, SupervisedDataset, ABC):
    def __init__(self, name, download_if_missing: bool = True, data_folder: str = None,
                 transformer: Callable = None, target_transformer: Callable = None, **kwargs):

        super().__init__(name=name, transformer=transformer,
                         download_if_missing=download_if_missing, data_folder=data_folder)

        (x, y), (train, test, dev) = self.load_dataset()

        super(DownloadableDataset, self).__init__(x=x, y=y, train=train, test=test, dev=dev,
                                                  transformer=transformer, target_transformer=target_transformer,
                                                  **kwargs)

    @abstractmethod
    def load_dataset(self) -> Tuple[Tuple[list, list], Tuple[list, list, list]]:
        raise NotImplementedError