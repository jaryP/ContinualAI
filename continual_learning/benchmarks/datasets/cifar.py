import tarfile
from typing import Tuple, Callable
import os
from urllib.request import urlretrieve

import numpy as np
import pickle
from continual_learning.benchmarks import SupervisedDownloadableDataset

__all__ = ['CIFAR100', 'CIFAR10']


class CIFAR10(SupervisedDownloadableDataset):
    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.data_folder, self.url.rpartition('/')[2]))

    files = {'train': ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'],
             'test': ['test_batch']}

    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    def __init__(self, download_if_missing=True, data_folder=None, fine_labels=False,
                 transformer: Callable = None, test_transformer: Callable = None, target_transformer: Callable = None):

        self.fine_labels = fine_labels

        super(CIFAR10, self).__init__(name='CIFAR10', download_if_missing=download_if_missing, data_folder=data_folder,
                                      transformer=transformer, target_transformer=target_transformer,
                                      test_transformer=test_transformer)

    def load_dataset(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[list, list, list]]:

        x_train, y_train = [], []
        x_test, y_test = [], []

        with tarfile.open(os.path.join(self.data_folder, self.url.rpartition('/')[2]), 'r') as tar:
            for item in tar:
                name = item.name.rpartition('/')[-1]
                if 'batch' in name:
                    contentfobj = tar.extractfile(item)
                    if contentfobj is not None:
                        entry = pickle.load(contentfobj, encoding='latin1')
                        if 'data' in entry:
                            x = entry['data']
                            if self.fine_labels:
                                y = entry['fine_labels']
                            else:
                                y = entry['labels']

                            if name in self.files['train']:
                                x_train.append(x)
                                y_train.append(y)
                            elif name in self.files['test']:
                                x_test.append(x)
                                y_test.append(y)

        x_train = np.concatenate(x_train)

        idx_train = list(range(len(x_train)))

        x_test = np.concatenate(x_test)

        idx_test = list(range(len(idx_train), len(idx_train) + len(x_test)))

        y = np.concatenate((*y_train, *y_test), 0)
        x = np.concatenate((x_train, x_test), 0) .reshape((-1, 3, 32, 32))
        x = np.transpose(x, (0, 2, 3, 1))

        return (x, y), (idx_train, idx_test, [])

    def download_dataset(self):
        urlretrieve(self.url, os.path.join(self.data_folder, self.url.rpartition('/')[2]))


class CIFAR100(CIFAR10):
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

    def __init__(self, download_if_missing=True, data_folder=None, fine_labels=True,
                 transformer: Callable = None, test_transformer: Callable = None, target_transformer: Callable = None):

        self.file_name = self.url.rpartition('/')[2]
        self.fine_labels = fine_labels

        super(CIFAR10, self).__init__(name='CIFAR100', download_if_missing=download_if_missing,
                                      data_folder=data_folder, test_transformer=test_transformer,
                                      transformer=transformer, target_transformer=target_transformer)

    def load_dataset(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[list, list, list]]:

        x_train, y_train = [], []
        x_test, y_test = [], []

        with tarfile.open(os.path.join(self.data_folder, self.file_name), 'r') as tar:
            for i, item in enumerate(tar):
                name = item.name.rpartition('/')[-1]

                if name not in ['train', 'test']:
                    continue

                contentfobj = tar.extractfile(item)
                if contentfobj is not None:
                    entry = pickle.load(contentfobj, encoding='latin1')
                    if 'train' in name:
                        x_train.append(entry['data'])
                        y_train.append((entry['fine_labels']))
                    else:
                        x_test.append(entry['data'])
                        y_test.append((entry['fine_labels']))

                    contentfobj.close()

        x_train = np.concatenate(x_train)

        idx_train = list(range(len(x_train)))

        x_test = np.concatenate(x_test)

        idx_test = list(range(len(idx_train), len(idx_train) + len(x_test)))

        y = np.concatenate((*y_train, *y_test), 0)
        x = np.concatenate((x_train, x_test), 0).reshape((-1, 3, 32, 32))
        x = np.transpose(x, (0, 2, 3, 1))

        return (x, y), (idx_train, idx_test, [])

    def download_dataset(self):
        urlretrieve(self.url, os.path.join(self.data_folder, self.file_name))
