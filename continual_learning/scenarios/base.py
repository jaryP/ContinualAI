from abc import ABC, abstractmethod
from typing import Union, List, Sequence

import numpy as np
from torch.utils.data import DataLoader

from continual_learning.banchmarks.base import IndexesContainer, SupervisedDataset, DatasetSplits, UnsupervisedDataset


class Task(IndexesContainer):
    def __init__(self, *, index: int, base_dataset, train: Union[list, np.ndarray], test: [list, np.ndarray] = None,
                 dev: [list, np.ndarray] = None, **kwargs):
        super().__init__(train=train, dev=dev, test=test, **kwargs)
        self.base_dataset = base_dataset
        self.index = index

    def __getitem__(self, item):
        idx = self.current_indexes
        i = idx[item]
        return self.base_dataset[i]

    @IndexesContainer.current_split.setter
    def current_split(self, v: DatasetSplits) -> None:
        self._current_split = v
        self.base_dataset.current_split = v

    def get_iterator(self, batch_size, shuffle=True, sampler=None, num_workers=0, pin_memory=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          sampler=sampler, pin_memory=pin_memory, num_workers=num_workers)


class SupervisedTask(Task):
    def __init__(self, *, index: int, base_dataset: SupervisedDataset, labels_mapping: dict,
                 train: Union[list, np.ndarray], test: [list, np.ndarray] = None,
                 dev: [list, np.ndarray] = None, **kwargs):

        super().__init__(index=index, base_dataset=base_dataset, train=train, dev=dev, test=test, **kwargs)
        self._task_labels = True
        self.labels_mapping = labels_mapping
        self.current_split = DatasetSplits.TRAIN

    def set_task_labels(self):
        self._task_labels = True

    def set_dataset_labels(self):
        self._task_labels = False

    @property
    def labels(self):
        if self._task_labels:
            return list(self.labels_mapping.values())
        else:
            return list(self.labels_mapping.keys())

    def _map_labels(self, y):
        if not isinstance(y, list):
            return self.labels_mapping[y]

        if self._task_labels:
            y = [self.labels_mapping[i] for i in y]

        return y

    @property
    def data(self):
        return self.x, self.y

    def y(self, split: DatasetSplits = None):
        if split is None:
            split = self.current_split

        a = self.get_indexes(split)
        _, _, y = self.base_dataset[a]
        return self._map_labels(y)

    def x(self, split: DatasetSplits = None):
        if split is None:
            split = self.current_split

        a = self.get_indexes(split)
        _, x, _ = self.base_dataset[a]
        return x

    def __getitem__(self, item):
        idx = self.current_indexes
        i = idx[item]
        i, x, y = self.base_dataset[i]
        y = self._map_labels(y)
        return i, x, y


class IncrementalProblem(ABC):
    def __init__(self, dataset: SupervisedDataset, labels_per_task: int, shuffle_labels: bool = False,
                 random_state: Union[np.random.RandomState, int] = None):
        super().__init__()

        if random_state is not None:
            if isinstance(random_state, int):
                random_state = np.random.RandomState(random_state)
        else:
            random_state = np.random.RandomState(None)

        self._tasks = self.generate_tasks(dataset=dataset, labels_per_task=labels_per_task,
                                          shuffle_labels=shuffle_labels, random_state=random_state)

    @abstractmethod
    def generate_tasks(self, dataset: SupervisedDataset, labels_per_task: int, shuffle_labels: bool = False,
                       random_state: Union[np.random.RandomState, int] = None) -> List[SupervisedTask]:
        raise NotImplementedError

    def __len__(self):
        return len(self._tasks)

    def __getitem__(self, i: int):
        return self._tasks[i]

    def __iter__(self):
        for t in self._tasks:
            yield t


class StreamProblem(ABC):
    def __init__(self, dataset: Union[Sequence[IndexesContainer], IndexesContainer],
                 shuffle_datasets: bool = False, random_state: Union[np.random.RandomState, int] = None):
        super().__init__()
        if random_state is not None:
            if isinstance(random_state, int):
                random_state = np.random.RandomState(random_state)
        else:
            random_state = np.random.RandomState(None)

        self._tasks = self.generate_tasks(dataset=dataset, shuffle_datasets=shuffle_datasets, random_state=random_state)

    @abstractmethod
    def generate_tasks(self, dataset: Union[Sequence[IndexesContainer], IndexesContainer],
                       shuffle_datasets: bool = False, random_state: Union[np.random.RandomState, int] = None)\
            -> List[Task]:
        raise NotImplementedError

    def __len__(self):
        return len(self._tasks)

    def __getitem__(self, i: int):
        return self._tasks[i]

    def __iter__(self):
        for t in self._tasks:
            yield t

