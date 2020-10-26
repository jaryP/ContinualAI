from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, List, Union

import numpy as np

from datasets import SupervisedDataset


# class AbstractTask(ABC):
#     pass
#     # def __init__(self, original_dataset: AbstractBaseDataset):
#     #     pass
#     #
#     # @abstractmethod
#     # def generate_tasks(self, original_dataset: AbstractBaseDataset):
#     #     raise NotImplementedError


class ClassificationTask(SupervisedDataset):
    def __init__(self, x, task_y, dataset_y, train, test=None, dev=None, transformer: Callable = None,
                 target_transformer: Callable = None, **kwargs):

        self._current_labels = 'task'
        self._task_y = task_y

        super().__init__(x=x, y=dataset_y, train=train, test=test, dev=dev, transformer=transformer,
                         target_transformer=target_transformer, **kwargs)

    def __setattr__(self, key, value):
        if key in ['_current_labels', '_task_y'] \
                and hasattr(self, '__initialized'):
            raise ValueError()

        super().__setattr__(key, value)

    @property
    def y(self):
        if self._current_labels == 'dataset':
            return super().y
        else:
            return [self._task_y[i] for i in self._current_split_idx]

    def __getitem__(self, item):
        _, x = super().__getitem__(item)
        y = self._y if self._current_labels == 'dataset' else self._task_y
        return item, x, self.target_transformer(y[self._current_split_idx[item]])

    def task_labels(self):
        self._current_labels = 'task'

    def dataset_labels(self):
        self._current_labels = 'dataset'


# class ClassificationTask(AbstractTask):
#     def __init__(self, original_dataset: LabeledDataset, task_labels: list, task_labels_map: list):
#         super().__init__()
#
#         if len(task_labels) != len(task_labels_map):
#             raise ValueError('The task labels and the associated map should have the same dimension: '
#                              '{} <> {}'.format(len(task_labels), len(task_labels_map)))
#
#         self.task_labels = task_labels
#         self.task_labels_map = task_labels_map
#         self._split = 'train'
#         self._current_labels = 'task'
#         self.transformer = original_dataset.transformer
#         self.target_transformer = original_dataset.target_transformer
#
#         self._x = []
#         self._task_y = []
#         self._dataset_y = []
#
#         self._train_split = []
#         self._test_split = []
#         self._dev_split = []
#
#         self._current_split_idx = self._train_split
#
#     def task_labels(self):
#         self._current_labels = 'task'
#
#     def dataset_labels(self):
#         self._current_labels = 'dataset'
#
#     @property
#     def x(self):
#         return [self._x[i] for i in self._current_split_idx]
#
#     @property
#     def y(self):
#         return [getattr(self, '{}_y'.format(self._current_labels))[i] for i in self._current_split_idx]
#
#     def __getitem__(self, item):
#         ci = self._current_split_idx[item]
#         return item, self.transformer(self._x[ci]), self.target_transformer(
#             getattr(self, '{}_y'.format(self._current_labels))[ci])
#
#     def __len__(self):
#         return len(self._current_split_idx)
#
#     def train(self):
#         self._split = 'train'
#         self._current_split_idx = self._train_split
#
#     def test(self):
#         self._split = 'test'
#         self._current_split_idx = self._test_split
#
#     def dev(self):
#         self._split = 'dev'
#         self._current_split_idx = self._dev_split
#
#     def all(self):
#         self._split = 'all'
#         self._current_split_idx = self._train_split + self._train_split + self._dev_split


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


# class TasksContainer(ABC):
#     def __init__(self, dataset: ClassificationDataset, labels_per_task: int, batch_size: int,
#                  shuffle_labels: bool = False,
#                  random_state: Union[np.random.RandomState, int] = None):
#
#         self._tasks = list()
#         self._current_task = 0
#         self.current_batch_size = None
#
#         if random_state is None or isinstance(random_state, int):
#             self.RandomState = np.random.RandomState(random_state)
#         elif isinstance(random_state, np.random.RandomState):
#             self.RandomState = random_state
#
#         self.generate_tasks(dataset, labels_per_task=labels_per_task, batch_size=batch_size,
#                             shuffle_labels=shuffle_labels)
#
#         for i, t in enumerate(self._tasks):
#             setattr(t, 'index', i)
#
#     def __len__(self):
#         return len(self._tasks)
#
#     @abstractmethod
#     def generate_tasks(self, dataset: ClassificationDataset, labels_per_task: int, batch_size: int,
#                        shuffle_labels: bool = False):
#         raise NotImplementedError
#
#     def add_task(self, task):
#         self._tasks.append(task)
#
#     @property
#     def task(self):
#         return self._tasks[self._current_task]
#
#     @task.setter
#     def task(self, v: int):
#         if v > len(self):
#             raise ValueError('ERROR (MODIFICARE)')
#         self._current_task = v
#
#     def __getitem__(self, i: int):
#         if i > len(self):
#             raise ValueError('ERROR (MODIFICARE)')
#         return self._tasks[i]
#
#     def __iter__(self):
#         for t in self._tasks:
#             yield t
