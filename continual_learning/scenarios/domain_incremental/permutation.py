from functools import reduce
from operator import mul
from typing import Union, List, Sequence

import numpy as np
import torch
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import ToPILImage, ToTensor, Compose

from continual_learning.benchmarks import SupervisedDataset, \
    UnsupervisedDataset, DatasetSplits
from continual_learning.scenarios.base import IncrementalSupervisedProblem, \
    DomainIncremental
from continual_learning.scenarios.tasks import Task, \
    UnsupervisedTransformerTask, SupervisedTransformerTask


class PixelsPermutation(object):
    def __init__(self, index_permutation: Sequence[int]):
        self.permutation = index_permutation

    def __call__(self, img: Union[Image, Tensor, np.ndarray]):
        if isinstance(img, np.ndarray):
            img = img.reshape(-1)[self.permutation].reshape(*img.shape)
        else:
            is_image = isinstance(img, Image)

            if is_image:
                img = ToTensor()(img)

            img = img.view(-1)[self.permutation].view(*img.shape)

            if is_image:
                img = ToPILImage()(img)

        return img


class Permutation(DomainIncremental):
    def __init__(self, dataset: Union[UnsupervisedDataset, SupervisedDataset],
                 permutation_n: int,
                 shuffle_datasets: bool = False,
                 random_state: Union[np.random.RandomState, int] = None,
                 **kwargs):

        self.base_train_t = dataset.transformer
        self.base_test_t = dataset.test_transformer
        self._permutations = []

        super().__init__(dataset,
                         random_state=random_state,
                         shuffle_datasets=shuffle_datasets,
                         permutation_n=permutation_n,
                         **kwargs)

    def generate_tasks(self, dataset:
    Union[UnsupervisedDataset, SupervisedDataset],
                       random_state: Union[np.random.RandomState, int] = None,
                       **kwargs) -> List[Union[UnsupervisedTransformerTask,
                                               SupervisedTransformerTask]]:

        permutation_n = kwargs['permutation_n']
        img = dataset[0][1]

        if isinstance(img, Image):
            img = ToTensor()(img)

        if isinstance(img, torch.Tensor):
            img_shape = img.numel()
        elif isinstance(img, np.ndarray):
            img_shape = reduce(mul, img.shape)

        tasks = []
        for i in range(permutation_n):
            if i > 0:
                perm = random_state.permutation(img_shape)
                perm = PixelsPermutation(perm)
            else:
                perm = lambda x: x

            if isinstance(dataset, SupervisedDataset):
                task = SupervisedTransformerTask(base_dataset=dataset,
                                                 train=dataset.get_indexes(
                                                     DatasetSplits.TRAIN),
                                                 test=dataset.get_indexes(
                                                     DatasetSplits.TEST),
                                                 dev=dataset.get_indexes(
                                                     DatasetSplits.TRAIN),
                                                 index=len(tasks),
                                                 labels_mapping=None,
                                                 transformer=perm)

            elif isinstance(dataset, UnsupervisedDataset):
                task = UnsupervisedTransformerTask(base_dataset=dataset,
                                                   train=dataset.get_indexes(
                                                       DatasetSplits.TRAIN),
                                                   test=dataset.get_indexes(
                                                       DatasetSplits.TEST),
                                                   dev=dataset.get_indexes(
                                                       DatasetSplits.TRAIN),
                                                   index=len(tasks),
                                                   transformer=perm)

            else:
                assert False

            tasks.append(task)

        return tasks

    def __iter__(self):
        for t in self._tasks:
            yield t


if __name__ == '__main__':
    from continual_learning.benchmarks import MNIST
    import matplotlib.pyplot as plt

    d = MNIST(download_if_missing=True,
              transformer=None,
              test_transformer=None, data_folder='/media/jary/Data/progetti/CL/'
                                                 'cl_framework/'
                                                 'continual_learning/'
                                                 'tests/training/mnist')
    perm = Permutation(d, permutation_n=10)
    print(len(perm))

    for t in perm:
        _, x, y = t[0]
        print(y, x.sum())
        plt.imshow(x)

        plt.show()
