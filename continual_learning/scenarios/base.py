__all__ = ['StreamDataset',
           'Task',
           'TasksGenerator']

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from continual_learning.datasets.base import SupervisedDataset, \
    UnsupervisedDataset, DatasetSplits


class StreamDataset(ABC):
    def __init__(self):
        pass


class Task(ABC):

    def __init__(self,
                 *,
                 base_dataset: Union[SupervisedDataset, UnsupervisedDataset],
                 task_index: int,
                 **kwargs):

        super().__init__(**kwargs)

        self._base_dataset = base_dataset
        self.task_index = task_index

    # @abstractmethod
    def __len__(self):
        return len(self.base_dataset)

    # @abstractmethod
    def __getitem__(self, item):
        return self.base_dataset[item]

    @property
    def base_dataset(self):
        return self._base_dataset

    @property
    def current_split(self) -> DatasetSplits:
        return self.base_dataset.current_split

    @current_split.setter
    def current_split(self, v: Union[DatasetSplits, int, str]) -> None:
        self.base_dataset.current_split = v

    def get_split_len(self, v: DatasetSplits = None) -> int:
        return self.base_dataset.get_split_len(v)

    @property
    def current_indexes(self) -> np.ndarray:
        return self.base_dataset.get_indexes(self.current_split)

    def get_indexes(self, v: DatasetSplits = None) -> np.ndarray:
        return self.base_dataset.get_indexes()

    def train(self) -> None:
        self.base_dataset.current_split = DatasetSplits.TRAIN

    def test(self) -> None:
        self.base_dataset.current_split = DatasetSplits.TEST

    def dev(self) -> None:
        self.base_dataset.current_split = DatasetSplits.DEV

    def all(self) -> None:
        self.base_dataset.current_split = DatasetSplits.ALL

    def get_split(self,
                  split: DatasetSplits,
                  **kwargs):
        return self.base_dataset.get_split(split)



class TasksGenerator(ABC):
    def __init__(self,
                 dataset: Union[SupervisedDataset, UnsupervisedDataset],
                 random_state: Union[np.random.RandomState, int] = None,
                 **kwargs):

        super().__init__()

        if random_state is not None:
            if isinstance(random_state, int):
                random_state = np.random.RandomState(random_state)
        else:
            random_state = np.random.RandomState(None)

        self.dataset = dataset
        self.random_state = random_state

    @abstractmethod
    def generate_task(self, dataset: Union[UnsupervisedDataset,
                                           SupervisedDataset],
                      random_state: Union[np.random.RandomState, int] = None,
                      **kwargs) \
            -> Union[Task, None]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, i: int):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

