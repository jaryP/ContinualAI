from abc import ABC, abstractmethod
from os import makedirs
from os.path import join, dirname, exists
from typing import Callable, Union, Tuple
import numpy as np


class AbstractBaseDataset(ABC):

    def __init__(self, name, transformer: Callable = None,
                 download_if_missing: bool = True, data_folder: str = None):

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


class UnlabeledDataset(AbstractBaseDataset, ABC):
    def __init__(self, name, transformer: Callable = None, download_if_missing: bool = True, data_folder: str = None):

        super(UnlabeledDataset, self).__init__(name=name, transformer=transformer,
                                               download_if_missing=download_if_missing, data_folder=data_folder)

        (self.__x), (self._train_split, self._test_split, self._dev_split) = self.load_dataset()

        self._split = 'train'
        self._current_split_idx = self._train_split

        self.labels = None

    def preprocess(self, f: Callable):
        for i, x in enumerate(self.__x):
            self.__x[i] = f(x)

    @abstractmethod
    def load_dataset(self) -> Tuple[list, Tuple[list, list, list]]:
        raise NotImplementedError

    def train(self):
        self._split = 'train'
        self._current_split_idx = self._train_split

    def test(self):
        self._split = 'test'
        self._current_split_idx = self._test_split

    def dev(self):
        self._split = 'dev'
        self._current_split_idx = self._dev_split

    def all(self):
        self._split = 'all'
        self._current_split_idx = self._train_split + self._train_split + self._dev_split

    @property
    def x(self):
        return [self.__x[i] for i in self._current_split_idx]

    def __getitem__(self, item):
        return item, self.transformer(self.__x[self._current_split_idx[item]])

    def __len__(self):
        return len(self._current_split_idx)

    @property
    def data(self):
        return self.x

    # TODO: da adattare
    def split_dataset(self, test_split: float = 0.2, dev_split: float = 0.0, balance_labels: bool = False,
                      random_state: Union[np.random.RandomState, int] = None):

        x = []
        y = []

        for i in ['train', 'test', 'dev']:
            if not getattr(self, '_x_{}'.format(i)) is None:
                x.append(getattr(self, '_x_{}'.format(i)))
                y.append(getattr(self, '_y_{}'.format(i)))

        x = np.concatenate(x, 0)
        y = np.concatenate(y, 0)

        (self._x_train, self._y_train, _), (self._x_dev, self._y_dev, _), (self._x_test, self._y_test, _) = \
            split_dataset(x, y, test_split=test_split, dev_split=dev_split, random_state=random_state,
                          balance_labels=balance_labels)

    # def pre_process(self, f=None):
    #     for i in 'train', ''
    #     if f is not None and hasattr(f, '__call__'):
    #         self._x, self._y = f(self._x, self._y)


class LabeledDataset(UnlabeledDataset, ABC):
    def __init__(self, name, transformer: Callable = None, download_if_missing: bool = True, data_folder: str = None,
                 target_transformer: Callable = None):

        super(LabeledDataset, self).__init__(name=name, transformer=transformer,
                                             download_if_missing=download_if_missing, data_folder=data_folder)

        self.target_transformer = target_transformer if target_transformer is not None else lambda x: x

        (self.__x, self.__y), (self._train_split, self._test_split, self._dev_split) = self.load_dataset()

        self._split = 'train'
        self._current_split_idx = self._train_split

        labels = sorted(list(set(self.__y)))
        self.labels = labels

    def preprocess_targets(self, f: Callable):
        for i, y in enumerate(self.__y):
            self.__y[i] = f(y)

    @abstractmethod
    def load_dataset(self) -> Tuple[Tuple[list, list], Tuple[list, list, list]]:
        raise NotImplementedError

    # def __setattr__(self, attr, val):
    #     if attr == '_split':
    #         raise ValueError('{} can be set using the functions train(), test(), dev() and all() '.
    #                          format(attr, self.__class__.__name__))
    #     elif attr in ['__x', '__y', 'train_split', 'test_split', 'dev_split']:
    #         raise ValueError('{} attribute cannot be set manually '.
    #                          format(attr, self.__class__.__name__))
    #
    #     super(UnlabeledDataset, self).__setattr__(attr, val)

    # def train(self):
    #     self._split = 'train'
    #     self._current_split_idx = self._train_split
    #
    # def test(self):
    #     self._split = 'test'
    #     self._current_split_idx = self._test_split
    #
    # def dev(self):
    #     self._split = 'dev'
    #     self._current_split_idx = self._dev_split
    #
    # def all(self):
    #     self._split = 'all'
    #     self._current_split_idx = self._train_split + self._train_split + self._dev_split

    # def _x(self):
    #     return getattr(self, F'_x_{self._split}')
    #
    # def _y(self):
    #     return getattr(self, F'_y_{self._split}')

    # @property
    # def current_split(self):
    #     return self._split

    # @property
    # def x(self):
    #     return [self.__x[i] for i in self._current_split_idx]

    @property
    def y(self):
        return [self.__y[i] for i in self._current_split_idx]

    @property
    def x(self):
        return [self.__x[i] for i in self._current_split_idx]

    def __getitem__(self, item):
        item, x = super().__getitem__([item])
        return item, x, self.target_transformer(self.__y[self._current_split_idx[item]])

    # def __len__(self):
    #     return len(self._current_split_idx)

    @property
    def data(self):
        return super().x, self.y

    # TODO: da adattare
    def split_dataset(self, test_split: float = 0.2, dev_split: float = 0.0, balance_labels: bool = False,
                      random_state: Union[np.random.RandomState, int] = None):

        x = []
        y = []

        for i in ['train', 'test', 'dev']:
            if not getattr(self, '_x_{}'.format(i)) is None:
                x.append(getattr(self, '_x_{}'.format(i)))
                y.append(getattr(self, '_y_{}'.format(i)))

        x = np.concatenate(x, 0)
        y = np.concatenate(y, 0)

        (self._x_train, self._y_train, _), (self._x_dev, self._y_dev, _), (self._x_test, self._y_test, _) = \
            split_dataset(x, y, test_split=test_split, dev_split=dev_split, random_state=random_state,
                          balance_labels=balance_labels)

    # def pre_process(self, f=None):
    #     for i in 'train', ''
    #     if f is not None and hasattr(f, '__call__'):
    #         self._x, self._y = f(self._x, self._y)
