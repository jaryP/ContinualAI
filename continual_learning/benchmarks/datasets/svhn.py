from os.path import join, exists
from typing import Callable, Tuple
from urllib.request import urlretrieve
import numpy as np
from scipy import io

from continual_learning.benchmarks import SupervisedDownloadableDataset


class SVHN(SupervisedDownloadableDataset):

    url = {'train': {'images': "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"},
           'test': {'images': "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"}}

    def __init__(self, download_if_missing: bool = True, data_folder: str = None,
                 transformer: Callable = None, test_transformer: Callable = None, target_transformer: Callable = None):

        self.file_names = [url.rpartition('/')[2] for url in self.url]

        super().__init__(name='SVHN', download_if_missing=download_if_missing, data_folder=data_folder,
                         transformer=transformer,
                         target_transformer=target_transformer,
                         test_transformer=test_transformer)

    def download_dataset(self):
        for _, type in self.url.items():
            for _, url in type.items():
                urlretrieve(url, join(self.data_folder, url.rpartition('/')[2]))

    def load_dataset(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[list, list, list]]:

        x, y = [], []
        train, dev, test = [], [], []

        for split in 'train', 'test':
            v = self.url[split]
            loaded_mat = io.loadmat(join(self.data_folder, v['images'].rpartition('/')[2]))

            x.append(loaded_mat['X'])
            y.append(loaded_mat['y'].astype(np.int64).squeeze())

            if split == 'train':
                train = list(range(x[-1].shape[-1]))
            else:
                test = list(range(len(train), len(train) + x[-1].shape[-1]))

        x = np.concatenate(x, -1)
        x = np.transpose(x, (3, 0, 1, 2))
        y = np.concatenate(y)

        return (x, y), (train, test, dev)

    def _check_exists(self) -> bool:
        for split in 'train', 'test':
            v = self.url[split]
            if not exists(join(self.data_folder, v['images'].rpartition('/')[2])):
                return False
        return True
