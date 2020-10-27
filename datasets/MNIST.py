import codecs
import gzip
from os.path import join
from typing import Callable
from urllib.request import urlretrieve

import numpy as np
# from datasets import SupervisedDownloadableDataset


# TODO: Aggiungere FashionMNIST
# TODO: Aggiungere QMNIST
# TODO: Aggiungere KMNIST
from datasets.base import SupervisedDownloadableDataset


class MNIST(SupervisedDownloadableDataset):
    url = {'train': {'images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                     'labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'},
           'test': {'images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                    'labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'}}

    def __init__(self, download_if_missing: bool = True, data_folder: str = None,
                 transformer: Callable = None, target_transformer: Callable = None, ):

        self._get_int = lambda x: int(codecs.encode(x, 'hex'), 16)

        self.file_names = [url.rpartition('/')[2] for url in self.url]

        super().__init__(name='MNIST', download_if_missing=download_if_missing, data_folder=data_folder,
                         transformer=transformer, target_transformer=target_transformer)

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

    def load_dataset(self):

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
        y = np.asarray(y)

        return (x, y), (train, test, dev)

    def download_dataset(self):
        for _, type in self.url.items():
            for _, url in type.items():
                urlretrieve(url, join(self.data_folder, url.rpartition('/')[2]))


if __name__ == '__main__':
    m = MNIST()
    m.train_split = None

    m.split_dataset(0.2)

    m.train()
    print(len(m))
    m.test()
    print(len(m))
    i, x, y = m[[1, 20, 30]]
    print(len(x))

    # for i in m:
    #     print(i[0])
    #     pass