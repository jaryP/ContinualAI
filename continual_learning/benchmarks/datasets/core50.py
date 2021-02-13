import os
import pickle
from itertools import chain
from os.path import exists
from typing import Callable, Tuple
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np

from continual_learning.benchmarks import SupervisedDownloadableDataset

scen2dirs = {
    'ni': "batches_filelists/NI_inc/",
    'nc': "batches_filelists/NC_inc/",
    'nic': "batches_filelists/NIC_inc/",
    'nicv2_79': "NIC_v2_79/",
    'nicv2_196': "NIC_v2_196/",
    'nicv2_391': "NIC_v2_391/"
}


class Core50_128(SupervisedDownloadableDataset):
    url = ['http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip',
           'https://vlomonaco.github.io/core50/data/paths.pkl',
           'https://vlomonaco.github.io/core50/data/LUP.pkl',
           'https://vlomonaco.github.io/core50/data/labels.pkl',
           'http://bias.csr.unibo.it/maltoni/download/core50/core50_imgs.npz',
           # 'https://vlomonaco.github.io/core50/data/batches_filelists.zip',
           # 'https://vlomonaco.github.io/core50/data/batches_filelists_NICv2.zip'
           ]

    nbatch = {
        'ni': 8,
        'nc': 9,
        'nic': 79,
        'nicv2_79': 79,
        'nicv2_196': 196,
        'nicv2_391': 391
    }

    def __init__(self, download_if_missing: bool = True, data_folder: str = None,
                 transformer: Callable = None, test_transformer: Callable = None, target_transformer: Callable = None,
                 scenario='ni'):

        self.scenario = scenario
        self.n_batches = self.nbatch[scenario]

        super().__init__(name='CORE50_128', download_if_missing=download_if_missing, data_folder=data_folder,
                         transformer=transformer, target_transformer=target_transformer, is_path_dataset=True,
                         test_transformer=test_transformer,
                         images_path="core50_128x128")

    def load_dataset(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[list, list, list]]:
        x = []
        y = []
        train_idx = []
        test_idx = []

        name = self.url[0].rpartition('/')[2]
        name = os.path.splitext(name)[0]

        with open(os.path.join(self.data_folder, 'paths.pkl'), 'rb') as pathsf, \
                open(os.path.join(self.data_folder, 'labels.pkl'), 'rb') as labelsf, \
                open(os.path.join(self.data_folder, 'LUP.pkl'), 'rb') as LUPf:

            train_test_paths = pickle.load(pathsf)

            all_targets = pickle.load(labelsf)
            train_test_targets = []

            for i in range(self.n_batches + 1):
                train_test_targets += all_targets[self.scenario][0][i]

            LUP = pickle.load(LUPf)

            test_idx = LUP[self.scenario][0][-1]

            for i in range(self.n_batches + 1):
                train_idx += LUP[self.scenario][0][i]

            for idx in chain(train_idx, test_idx):
                x.append(os.path.join(self.data_folder, name, train_test_paths[idx]))
                y.append(train_test_targets[idx])

            y = np.array(y)
            x = np.array(x)

        return (x, y), (train_idx, test_idx, [])

    def _check_exists(self) -> bool:
        for url in self.url:
            name = url.rpartition('/')[2]
            if not exists(os.path.join(self.data_folder, name)):
                return False
        return True

    def download_dataset(self):

        for url in self.url:
            name = url.rpartition('/')[2]
            urlretrieve(url, os.path.join(self.data_folder, name))

            if name.endswith('.zip'):
                with ZipFile(os.path.join(self.data_folder, name), 'r') as zipf:
                    zipf.extractall(self.data_folder)
