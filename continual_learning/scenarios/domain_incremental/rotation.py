from functools import reduce
from operator import mul
from typing import Union, List

import numpy as np
import torch
from PIL.Image import Image
from torchvision.transforms import ToTensor, RandomRotation, Compose, ToPILImage

from continual_learning.benchmarks import UnsupervisedDataset, \
    SupervisedDataset, DatasetSplits
from continual_learning.scenarios.base import DomainIncremental
from continual_learning.scenarios.tasks import Task, SupervisedTransformerTask, \
    UnsupervisedTransformerTask


class RotationWrapper:
    def __init__(self, degree):
        self.t = RandomRotation(degree)

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = ToPILImage()(x)
            x = self.t(x)
            x = np.asarray(x)
        else:
            is_image = isinstance(x, Image)

            if is_image:
                x = ToTensor()(x)

            x = self.t(x)

            if is_image:
                x = ToPILImage()(x)

        return x

class Rotation(DomainIncremental):
    def __init__(self, dataset: Union[UnsupervisedDataset, SupervisedDataset],
                 rotations_n: int,
                 rotations_degree: List[float] = None,
                 shuffle_datasets: bool = False,
                 random_state: Union[np.random.RandomState, int] = None,
                 **kwargs):

        self.base_train_t = dataset.transformer
        self.base_test_t = dataset.test_transformer
        self._permutations = []

        super().__init__(dataset,
                         random_state=random_state,
                         shuffle_datasets=shuffle_datasets,
                         rotations_n=rotations_n,
                         rotations_degree=rotations_degree,
                         **kwargs)

    def generate_tasks(self, dataset:
    Union[UnsupervisedDataset, SupervisedDataset],
                       random_state: Union[np.random.RandomState, int] = None,
                       **kwargs) -> List[Union[UnsupervisedTransformerTask,
                                               SupervisedTransformerTask]]:

        rotations = kwargs['rotations_n']
        degrees = kwargs['rotations_degree']

        if degrees is None:
            assert rotations > 1
            degrees = [0] + [random_state.randint(0, 359)
                             for _ in range(rotations - 1)]

        tasks = []
        for i in range(rotations):
            if i > 0:
                print(degrees[i])
                perm = RotationWrapper(degrees[i])
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
              test_transformer=None,
              data_folder='/media/jary/Data/progetti/CL/'
                          'cl_framework/'
                          'continual_learning/'
                          'tests/training/mnist')
    perm = Rotation(d, rotations_n=10)
    print(len(perm))

    for t in perm:
        _, x, y = t[0]
        # print(y, x.sum())
        plt.imshow(x)

        plt.show()
