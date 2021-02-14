import codecs
import gzip
from os.path import join, exists
from typing import Callable, Tuple
from urllib.request import urlretrieve
import numpy as np
from continual_learning.benchmarks import SupervisedDownloadableDataset

__all__ = ['MNIST', 'K49MNIST', 'KMNIST']


class MNIST(SupervisedDownloadableDataset):
    url = {'train': {'images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                     'labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'},
           'test': {'images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                    'labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'}}

    def __init__(self, download_if_missing: bool = True, data_folder: str = None,
                 transformer: Callable = None, test_transformer: Callable = None, target_transformer: Callable = None):

        self._get_int = lambda x: int(codecs.encode(x, 'hex'), 16)

        self.file_names = [url.rpartition('/')[2] for url in self.url]

        super().__init__(name='MNIST', download_if_missing=download_if_missing, data_folder=data_folder,
                         transformer=transformer,
                         target_transformer=target_transformer,
                         test_transformer=test_transformer)

    def _load_image(self, data):
        length = self._get_int(data[4:8])
        num_rows = self._get_int(data[8:12])
        num_cols = self._get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        parsed = parsed.reshape((length, num_rows, num_cols))

        return parsed

    def _load_label(self, data):
        length = self._get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        parsed = parsed.reshape((length, -1))
        parsed = np.squeeze(parsed)
        return parsed

    def download_dataset(self):
        for _, type in self.url.items():
            for _, url in type.items():
                urlretrieve(url, join(self.data_folder, url.rpartition('/')[2]))

    def load_dataset(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[list, list, list]]:

        x, y = [], []
        train, dev, test = [], [], []

        for split in 'train', 'test':
            v = self.url[split]
            for j in 'images', 'labels':
                with gzip.GzipFile(join(self.data_folder, v[j].rpartition('/')[2])) as zip_f:
                    data = zip_f.read()

                    if j == 'images':
                        _x = self._load_image(data)[:, None]
                        x.extend(_x)
                        if split == 'train':
                            train = list(range(len(_x)))
                        else:
                            test = list(range(len(train), len(train) + len(_x)))
                    else:
                        y.extend(self._load_label(data))

        x = np.asarray(x)
        x = np.transpose(x, (0, 2, 3, 1))
        y = np.asarray(y)

        return (x, y), (train, test, dev)

    def _check_exists(self) -> bool:
        for split in 'train', 'test':
            v = self.url[split]
            for j in 'images', 'labels':
                if not exists(join(self.data_folder, v[j].rpartition('/')[2])):
                    return False
        return True


class KMNIST(MNIST):
    url = {'train': {'images': 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz',
                     'labels': 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz'},
           'test': {'images': 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz',
                    'labels': 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz'}}

    def __init__(self, download_if_missing: bool = True, data_folder: str = None,
                 transformer: Callable = None, test_transformer: Callable = None, target_transformer: Callable = None):
        self._get_int = lambda x: int(codecs.encode(x, 'hex'), 16)
        self.file_names = [url.rpartition('/')[2] for url in self.url]

        super(MNIST, self).__init__(name='KMNIST', download_if_missing=download_if_missing, data_folder=data_folder,
                                    transformer=transformer, target_transformer=target_transformer,
                                    test_transformer=test_transformer)


class K49MNIST(MNIST):
    url = {'train': {'images': 'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz',
                     'labels': 'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz'},
           'test': {'images': 'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz',
                    'labels': 'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz'}}

    def __init__(self, download_if_missing: bool = True, data_folder: str = None,
                 transformer: Callable = None, test_transformer: Callable = None, target_transformer: Callable = None):

        self._get_int = lambda x: int(codecs.encode(x, 'hex'), 16)

        self.file_names = [url.rpartition('/')[2] for url in self.url]

        super(MNIST, self).__init__(name='K49MNIST', download_if_missing=download_if_missing, data_folder=data_folder,
                                    transformer=transformer, target_transformer=target_transformer,
                                    test_transformer=test_transformer)

    def load_dataset(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[list, list, list]]:

        x, y = [], []
        train, dev, test = [], [], []

        for split in 'train', 'test':
            v = self.url[split]
            for j in 'images', 'labels':
                file_path = join(self.data_folder, v[j].rpartition('/')[2])

                data = np.load(file_path)
                data = data[data.files[0]]

                if j == 'images':
                    x.extend(data)
                    if split == 'train':
                        train = list(range(len(data)))
                    else:
                        test = list(range(len(train), len(train) + len(data)))
                else:
                    y.extend(data)

        x = np.asarray(x)
        y = np.asarray(y)

        return (x, y), (train, test, dev)
