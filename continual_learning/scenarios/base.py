from abc import ABC, abstractmethod
from typing import Union, List, Type

import numpy as np
import torch

from continual_learning.datasets.base import SupervisedDataset, \
    UnsupervisedDataset


# from continual_learning.scenarios.classification.tasks import SupervisedTask


class StreamDataset(ABC):
    def __init__(self):
        pass


class Task(ABC):

    def __init__(self,
                 *,
                 base_dataset: SupervisedDataset,
                 task_index: int,
                 # train: Union[list, np.ndarray],
                 # test: [list, np.ndarray] = None,
                 # dev: [list, np.ndarray] = None,
                 **kwargs):

        super().__init__(**kwargs)

        self._base_dataset = base_dataset
        self.task_index = task_index
    #
    @property
    def base_dataset(self):
        return self._base_dataset

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    # def __getitem__(self, item):
    #     idx = self.current_indexes
    #     i = idx[item]
    #     return self.base_dataset[i]
    #
    # @IndexesContainer.current_split.setter
    # def current_split(self, v: DatasetSplits) -> None:
    #     self._current_split = v
    #     self.base_dataset.current_split = v


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

    # @abstractmethod
    # def __next__(self):
    #     raise NotImplementedError


# class IncrementalSupervisedProblem(ABC):
#     def __init__(self,
#                  dataset: SupervisedDataset,
#                  labels_per_task: int,
#                  shuffle_labels: bool = False,
#                  random_state: Union[np.random.RandomState, int] = None,
#                  **kwargs):
#
#         if random_state is not None:
#             if isinstance(random_state, int):
#                 random_state = np.random.RandomState(random_state)
#         else:
#             random_state = np.random.RandomState(None)
#
#         self._tasks = self.generate_tasks(dataset=dataset,
#                                           labels_per_task=labels_per_task,
#                                           shuffle_labels=shuffle_labels,
#                                           random_state=random_state,
#                                           **kwargs
#                                           )
#
#     @abstractmethod
#     def generate_tasks(self,
#                        dataset: SupervisedDataset,
#                        labels_per_task: int,
#                        shuffle_labels: bool = False,
#                        random_state: Union[np.random.RandomState, int] = None,
#                        **kwargs) -> List[SupervisedTask]:
#         raise NotImplementedError
#
#     def __len__(self):
#         return len(self._tasks)
#
#     def __getitem__(self, i: int):
#         return self._tasks[i]
#
#     def __iter__(self):
#         for t in self._tasks:
#             yield t
#
#
# class AbstractTrainer(ABC):
#     @abstractmethod
#     def train_epoch(self, **kwargs):
#         raise NotImplementedError
#
#     @abstractmethod
#     def train_task(self, **kwargs):
#         raise NotImplementedError
#
#     @abstractmethod
#     def train_full(self, **kwargs):
#         raise NotImplementedError
#
#     def add_parameters_to_optimizer(self, optimizer: torch.optim.Optimizer,
#                                     parameters: dict):
#         optimizer.add_param_group({'params': parameters})
#
#     def change_optimizer_parameters(self, optimizer: torch.optim.Optimizer,
#                                     parameters):
#         optimizer.param_groups[0]['params'] = parameters

