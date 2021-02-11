from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np

from continual_learning.banchmarks import SupervisedDataset
from continual_learning.scenarios.base import SupervisedTask


class IncrementalProblem(ABC):
    def __init__(self, dataset: SupervisedDataset, labels_per_task: int, shuffle_labels: bool = False,
                 random_state: Union[np.random.RandomState, int] = None):
        super().__init__()

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


def get_labels_set(labels: Union[tuple, list, np.ndarray], labels_per_task: int, shuffle_labels: bool = False,
                   random_state: Union[np.random.RandomState, int] = None):
    if shuffle_labels:
        if random_state is not None:
            if isinstance(random_state, int):
                random_state = np.random.RandomState(random_state)
            random_state.shuffle(labels)
        else:
            np.random.shuffle(labels)

    labels_sets = [list(labels[i:i + labels_per_task]) for i in range(0, len(labels), labels_per_task)]

    if len(labels_sets[-1]) == 1:
        labels_sets[-2].extend(labels_sets[-1])
        labels_sets = labels_sets[:-1]

    return np.asarray(labels_sets)