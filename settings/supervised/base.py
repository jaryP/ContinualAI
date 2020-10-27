from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, List, Union

import numpy as np

from datasets import SupervisedDataset


class ClassificationTask(SupervisedDataset):
    def __init__(self, x: np.ndarray, task_y: np.ndarray, dataset_y: np.ndarray, train: Union[list, np.ndarray],
                 test: Union[list, np.ndarray] = None, dev: Union[list, np.ndarray] = None,
                 transformer: Callable = None,
                 target_transformer: Callable = None, **kwargs):

        self._current_labels = 'task'
        self._task_y = task_y
        self._dataset_y = dataset_y

        self._dataset_labels = sorted(set(dataset_y))
        self._task_labels = sorted(set(task_y))

        self.task_labels()

        super().__init__(x=x, y=task_y, train=train, test=test, dev=dev, transformer=transformer,
                         target_transformer=target_transformer, **kwargs)

    def task_labels(self):
        self._current_labels = 'task'
        self._y = self._task_y
        self._labels = self._task_labels

    def dataset_labels(self):
        self._current_labels = 'dataset'
        self._y = self._dataset_y
        self._labels = self._dataset_labels


class ContinualLearningSupervisedProblem(ABC):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, i: int):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError


class IncrementalProblem(ContinualLearningSupervisedProblem, ABC):
    def __init__(self, dataset: SupervisedDataset, labels_per_task: int, shuffle_labels: bool = False,
                 random_state: Union[np.random.RandomState, int] = None):
        super().__init__()

        self._tasks = self.generate_tasks(dataset=dataset, labels_per_task=labels_per_task,
                                          shuffle_labels=shuffle_labels, random_state=random_state)

    @abstractmethod
    def generate_tasks(self, dataset: SupervisedDataset, labels_per_task: int, shuffle_labels: bool = False,
                       random_state: Union[np.random.RandomState, int] = None) -> List[ClassificationTask]:
        raise NotImplementedError

    def __len__(self):
        return len(self._tasks)

    def __getitem__(self, i: int):
        return self._tasks[i]

    def __iter__(self):
        for t in self._tasks:
            yield t


class TransformationProblem(ContinualLearningSupervisedProblem, ABC):
    def __init__(self, dataset: SupervisedDataset, number_of_tasks: int, tasks_values: list = None):
        super().__init__()

        self._base_transformer = deepcopy(dataset.transformer)
        self._transformers = self.get_transformers(number_of_tasks, tasks_values)
        self.dataset = dataset

    @abstractmethod
    def get_transformers(self, number_of_tasks, tasks_values) -> List[Callable]:
        raise NotImplementedError

    def __len__(self):
        return len(self._transformers)

    def __getitem__(self, i: int):
        self.dataset.transformer = self._transformers[i]
        return self.dataset

    def __iter__(self):
        for t in self._transformers:
            self.dataset.transformer = t
            yield self.dataset
