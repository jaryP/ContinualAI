import bisect
import os
from abc import ABC, abstractmethod
from os import makedirs
from os.path import join, dirname, exists
from typing import Callable, Tuple, Union, Sequence

import numpy as np
from torch.utils.data import DataLoader

from continual_learning.benchmarks import UnsupervisedDataset, SupervisedDataset
from continual_learning.benchmarks.base import DatasetSplits


class DownloadableDataset(ABC):

    def __init__(self,
                 name: str,
                 transformer: Callable = None,
                 download_if_missing: bool = True,
                 data_folder: str = None,
                 **kwargs):
        """
        An abstract class used to download the benchmarks.
        :param name: The name of the dataset.
        :param transformer: The transformer function used when a sample is retrieved.
        :param download_if_missing: If the dataset needs to be downloaded if missing.
        :param data_folder: Where the dataset is stored.
        """

        if data_folder is None:
            data_folder = join(dirname(__file__), 'downloaded_datasets', name)

        self.data_folder = data_folder
        self._name = name

        self.transformer = transformer \
            if transformer is not None else lambda x: x

        missing = not self._check_exists()

        if missing:
            if not download_if_missing:
                raise IOError("Data not found and "
                              "`download_if_missing` is False")
            else:
                if not exists(self.data_folder):
                    makedirs(self.data_folder)

                print('Downloading dataset {}'.format(self.name))
                self.download_dataset()

    @property
    def name(self):
        return self._name

    @abstractmethod
    def download_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def _check_exists(self) -> bool:
        raise NotImplementedError


class UnsupervisedDownloadableDataset(DownloadableDataset,
                                      UnsupervisedDataset,
                                      ABC):
    def __init__(self,
                 name: str,
                 download_if_missing: bool = True,
                 data_folder: str = None,
                 transformer: Callable = None,
                 target_transformer: Callable = None,
                 **kwargs):
        super().__init__(name=name,
                         transformer=transformer,
                         download_if_missing=download_if_missing,
                         data_folder=data_folder)

        x, (train, test, dev) = self.load_dataset()

        super(DownloadableDataset, self).__init__(x=x,
                                                  train=train,
                                                  test=test,
                                                  dev=dev,
                                                  transformer=transformer,
                                                  target_transformer=
                                                  target_transformer,
                                                  **kwargs)

    @abstractmethod
    def load_dataset(self) -> Tuple[np.ndarray, Tuple[list, list, list]]:
        raise NotImplementedError


class SupervisedDownloadableDataset(DownloadableDataset, SupervisedDataset,
                                    ABC):
    def __init__(self,
                 name: str,
                 download_if_missing: bool = True,
                 data_folder: str = None,
                 transformer: Callable = None,
                 test_transformer: Callable = None,
                 target_transformer: Callable = None,
                 **kwargs):
        super().__init__(name=name,
                         transformer=transformer,
                         download_if_missing=download_if_missing,
                         data_folder=data_folder,
                         **kwargs)

        (x, y), (train, test, dev) = self.load_dataset()

        if kwargs.get('is_path_dataset', False):
            kwargs['images_path'] = os.path.join(self.data_folder,
                                                 kwargs['images_path'])

        super(DownloadableDataset, self).__init__(x=x, y=y, train=train,
                                                  test=test, dev=dev,
                                                  transformer=transformer,
                                                  target_transformer=
                                                  target_transformer,
                                                  test_transformer=
                                                  test_transformer,
                                                  **kwargs)

    @abstractmethod
    def load_dataset(self) -> Tuple[
        Tuple[np.ndarray, np.ndarray], Tuple[list, list, list]]:
        raise NotImplementedError


class ConcatDataset:
    r"""
    modified version of ConcatDataset from torch
    """

    @property
    def cumulative_sizes(self):
        return self.cumsum(self.datasets)

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Sequence[
                 Union[UnsupervisedDataset, SupervisedDataset]]) -> None:

        super(ConcatDataset, self).__init__()

        if len(datasets) == 0:
            raise ValueError('No datase given')

        lens = len(datasets[0][0])
        cond = all([len(d[0]) == lens for d in datasets])

        if not cond:
            raise ValueError('The dataset\'s __getitem__ are not comparable, '
                             'because return different number of values: {}'.
                             format([len(d[0]) for d in datasets]))

        self.datasets = list(datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def get_iterator(self, batch_size, shuffle=True, sampler=None,
                     num_workers=0, pin_memory=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          sampler=sampler, pin_memory=pin_memory,
                          num_workers=num_workers)

    def train(self) -> None:
        for d in self.datasets:
            d.train()

    def dev(self) -> None:
        for d in self.datasets:
            d.dev()

    def test(self) -> None:
        for d in self.datasets:
            d.test()

    def all(self) -> None:
        for d in self.datasets:
            d.all()

    @property
    def current_split(self) -> DatasetSplits:
        return self.datasets[0].current_split

    @current_split.setter
    def current_split(self, v: DatasetSplits) -> None:
        for d in self.datasets:
            d.current_split = v
