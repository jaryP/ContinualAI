from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np

from continual_learning.banchmarks import SupervisedDataset
from continual_learning.scenarios.supervised import ClassificationTask


class IncrementalProblem(ABC):
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
