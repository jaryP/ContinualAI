import os
from abc import ABC, abstractmethod
from os import makedirs
from os.path import join, dirname, exists
from typing import Callable, Tuple

import numpy as np

from continual_learning.banchmarks import UnsupervisedDataset, SupervisedDataset


class DownloadableDataset(ABC):

    def __init__(self, name, transformer: Callable = None,
                 download_if_missing: bool = True, data_folder: str = None, **kwargs):
        """
        An abstract class used to download the banchmarks.
        :param name: The name of the dataset.
        :param transformer: The transformer function used when a sample is retrieved.
        :param download_if_missing: If the dataset needs to be downloaded if missing.
        :param data_folder: Where the dataset is stored.
        """

        if data_folder is None:
            data_folder = join(dirname(__file__), 'downloaded_datasets', name)

        self.data_folder = data_folder
        self._name = name

        self.transformer = transformer if transformer is not None else lambda x: x

        missing = not self._check_exists()

        if missing:
            if not download_if_missing:
                raise IOError("Data not found and `download_if_missing` is False")
            else:
                if not exists(self.data_folder):
                    makedirs(self.data_folder)

                print('Downloading dataset')
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
    def load_dataset(self) -> Tuple[np.ndarray, Tuple[list, list, list]]:
        raise NotImplementedError


class SupervisedDownloadableDataset(DownloadableDataset, SupervisedDataset, ABC):
    def __init__(self, name, download_if_missing: bool = True, data_folder: str = None,
                 transformer: Callable = None, test_transformer: Callable = None, target_transformer: Callable = None,
                 **kwargs):
        super().__init__(name=name, transformer=transformer,
                         download_if_missing=download_if_missing, data_folder=data_folder, **kwargs)
        (x, y), (train, test, dev) = self.load_dataset()

        if kwargs.get('is_path_dataset', False):
            kwargs['images_path'] = os.path.join(self.data_folder, kwargs['images_path'])

        super(DownloadableDataset, self).__init__(x=x, y=y, train=train, test=test, dev=dev,
                                                  transformer=transformer, target_transformer=target_transformer,
                                                  test_transformer=test_transformer,
                                                  **kwargs)

    @abstractmethod
    def load_dataset(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[list, list, list]]:
        raise NotImplementedError