import os
import pickle
from itertools import chain
from os.path import exists
from typing import Callable, Tuple
from urllib.request import urlretrieve
from zipfile import ZipFile

import torchvision
import numpy as np

from continual_learning.banchmarks import SupervisedDownloadableDataset


class Core50_128(SupervisedDownloadableDataset):
    url = ['http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip',
           'https://vlomonaco.github.io/core50/data/paths.pkl',
           'https://vlomonaco.github.io/core50/data/LUP.pkl',
           'https://vlomonaco.github.io/core50/data/labels.pkl',
           'http://bias.csr.unibo.it/maltoni/download/core50/core50_imgs.npz',
           # 'https://vlomonaco.github.io/core50/data/batches_filelists.zip',
           # 'https://vlomonaco.github.io/core50/data/batches_filelists_NICv2.zip'
           ]

    def __init__(self, download_if_missing: bool = True, data_folder: str = None,
                 transformer: Callable = None, target_transformer: Callable = None, scenario='ni'):
        self.scenario = scenario
        self.n_batches = 8

        super().__init__(name='CORE50_128', download_if_missing=download_if_missing, data_folder=data_folder,
                         transformer=transformer, target_transformer=target_transformer, is_path_dataset=True,
                         images_path="core50_128x128")

    def load_dataset(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[list, list, list]]:
        x = []
        y = []
        train_idx = []
        test_idx = []

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
                # img = Image.open(os.path.join(self.data_folder, "core50_128x128", train_test_paths[idx]))
                # img = np.asarray(img.convert('RGB'))[None, :]
                # x.append(img)
                x.append(train_test_paths[idx])
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


# TODO: remove main
if __name__ == "__main__":
    # this litte example script can be used to visualize the first image
    # leaded from the dataset.
    from torch.utils.data import DataLoader

    # import matplotlib.pyplot as plt
    # from torchvision import transforms
    # import torch

    t = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(32),
        # torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # nn.Flatten(0)
    ])

    train_data = Core50_128(transformer=t)
    dataloader = DataLoader(train_data, batch_size=1)

    for batch_data in dataloader:
        i, x, y = batch_data
        print(x.shape)

    #     print(x.shape)
    #     plt.imshow(
    #         transforms.ToPILImage()(torch.squeeze(x))
    #     )
    #     plt.show()
    #     print(x.size())
    #     print(len(y))
    #     break
