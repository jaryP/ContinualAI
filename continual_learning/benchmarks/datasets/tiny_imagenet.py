import os
from collections import defaultdict
from operator import add
from os.path import exists
from typing import Callable, Tuple
from zipfile import ZipFile

import numpy as np

from continual_learning.benchmarks import SupervisedDownloadableDataset
from urllib.request import urlretrieve


class TinyImagenet(SupervisedDownloadableDataset):
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

    def __init__(self,
                 download_if_missing: bool = True,
                 data_folder: str = None,
                 transformer: Callable = None,
                 test_transformer: Callable = None,
                 target_transformer: Callable = None):

        super().__init__(name='TINY_IMAGENET',
                         download_if_missing=download_if_missing,
                         data_folder=data_folder,
                         transformer=transformer,
                         target_transformer=target_transformer,
                         test_transformer=test_transformer,
                         is_path_dataset=True,
                         images_path="tiny-imagenet-200")

    def download_dataset(self):
        """ Downloads the TintImagenet Dataset """

        name = self.url.rpartition('/')[2]
        urlretrieve(self.url, os.path.join(self.data_folder, name))

        with ZipFile(os.path.join(self.data_folder, name), 'r') as f:
            f.extractall(self.data_folder)

    def load_dataset(self) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                    Tuple[list, list, list]]:

        name = self.url.rpartition('/')[2]
        name = os.path.splitext(name)[0]

        val_images = os.path.join(self.data_folder,
                                  name, 'val', 'images')
        val_classes = os.path.join(self.data_folder,
                                   name, 'val', 'val_annotations.txt')

        labels_map = {}

        labels = set()
        with open(os.path.join(self.data_folder, name, 'wnids.txt'), 'r') as f:
            for line in f:
                label = line.strip()
                labels.add(label)
                if label not in labels_map:
                    labels_map[label] = len(labels_map)

        x = []
        y = []

        for c, i in labels_map.items():
            train_img_folder = os.path.join(self.data_folder, name,
                                            'train', c, 'images')

            img_paths = [os.path.join(train_img_folder, f)
                         for f in os.listdir(train_img_folder)
                         if os.path.isfile(os.path.join(train_img_folder, f))]

            x.extend(img_paths)
            y.extend([i] * len(img_paths))

            # test_x = test_paths[c]
            # x_test.extend(test_x)
            # y_test.extend([i] * len(test_x))

        x_test = []
        y_test = []

        test_paths = defaultdict(list)
        with open(val_classes, 'r') as f:
            for line in f:
                img, c, _, _, _, _ = line.strip().split()
                test_paths[c].append(os.path.join(val_images, img))
                if c in labels_map:
                    x_test.append(os.path.join(val_images, img))
                    y_test.append(labels_map[c])

        train_idx = list(range(len(x)))
        test_idx = list(map(lambda x: x + len(train_idx), range(len(x_test))))

        x.extend(x_test)
        y.extend(y_test)

        x = np.asarray(x)
        y = np.asarray(y, dtype=int)

        return (x, y), (train_idx, test_idx, [])

    def _check_exists(self) -> bool:
        name = self.url.rpartition('/')[2]
        if not exists(os.path.join(self.data_folder, name)):
            return False
        return True
