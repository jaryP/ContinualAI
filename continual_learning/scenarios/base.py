from abc import ABC, abstractmethod
from typing import Union, List, Type

import numpy as np
import torch

from continual_learning.benchmarks.base import UnsupervisedDataset, \
    SupervisedDataset
from continual_learning.scenarios.tasks import Task, SupervisedTask


class StreamDataset(ABC):
    def __init__(self):
        pass


class DomainIncremental(ABC):
    def __init__(self,
                 dataset: Union[SupervisedDataset, UnsupervisedDataset],
                 shuffle_datasets: bool = False,
                 random_state: Union[np.random.RandomState, int] = None,
                 **kwargs):

        super().__init__()

        if random_state is not None:
            if isinstance(random_state, int):
                random_state = np.random.RandomState(random_state)
        else:
            random_state = np.random.RandomState(None)

        self._tasks = self.generate_tasks(dataset=dataset,
                                          shuffle_datasets=shuffle_datasets,
                                          random_state=random_state,
                                          **kwargs)
        if shuffle_datasets:
            random_state.shuffle(self._tasks)

    @abstractmethod
    def generate_tasks(self, dataset: Union[UnsupervisedDataset,
                                            SupervisedDataset],
                       random_state: Union[np.random.RandomState, int] = None,
                       **kwargs) \
            -> List[Type[Task]]:
        raise NotImplementedError

    def __len__(self):
        return len(self._tasks)

    def __getitem__(self, i: int):
        return self._tasks[i]

    def __iter__(self):
        for t in self._tasks:
            yield t


class IncrementalSupervisedProblem(ABC):
    def __init__(self,
                 dataset: SupervisedDataset,
                 labels_per_task: int,
                 shuffle_labels: bool = False,
                 random_state: Union[np.random.RandomState, int] = None,
                 **kwargs):

        if random_state is not None:
            if isinstance(random_state, int):
                random_state = np.random.RandomState(random_state)
        else:
            random_state = np.random.RandomState(None)

        self._tasks = self.generate_tasks(dataset=dataset,
                                          labels_per_task=labels_per_task,
                                          shuffle_labels=shuffle_labels,
                                          random_state=random_state,
                                          **kwargs
                                          )

    @abstractmethod
    def generate_tasks(self,
                       dataset: SupervisedDataset,
                       labels_per_task: int,
                       shuffle_labels: bool = False,
                       random_state: Union[np.random.RandomState, int] = None,
                       **kwargs) -> List[SupervisedTask]:
        raise NotImplementedError

    def __len__(self):
        return len(self._tasks)

    def __getitem__(self, i: int):
        return self._tasks[i]

    def __iter__(self):
        for t in self._tasks:
            yield t


class AbstractTrainer(ABC):
    @abstractmethod
    def train_epoch(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train_task(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train_full(self, **kwargs):
        raise NotImplementedError

    def add_parameters_to_optimizer(self, optimizer: torch.optim.Optimizer,
                                    parameters: dict):
        optimizer.add_param_group({'params': parameters})

    def change_optimizer_parameters(self, optimizer: torch.optim.Optimizer,
                                    parameters):
        optimizer.param_groups[0]['params'] = parameters
