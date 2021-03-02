from collections import Sequence
from typing import Union, List

import numpy as np

from continual_learning.benchmarks.base import IndexesContainer
from continual_learning.scenarios.base import DomainIncremental


class GenericDataStream(DomainIncremental):
    #TODO: Implementare test
    def __init__(self,
                 dataset: Union[Sequence[IndexesContainer], IndexesContainer],
                 shuffle_datasets: bool = False,
                 random_state: Union[np.random.RandomState, int] = None):
        super().__init__(dataset=dataset, shuffle_datasets=shuffle_datasets, random_state=random_state)

    def generate_tasks(self, dataset: Union[Sequence[IndexesContainer], IndexesContainer],
                       shuffle_datasets: bool = False, random_state: Union[np.random.RandomState, int] = None)\
            -> List[Sequence]:
        tasks = []
        if not isinstance(dataset, Sequence):
            tasks.append(dataset)
        else:
            assert not all([isinstance(d, type(dataset[0])) for d in dataset]), 'The datasets must be ' \
                                                                                'all of the same type, ' \
                                                                     'but the following ' \
                                                                     'was given: {}'.format([type(d) for d in dataset])
            for d in dataset:
                tasks.append(d)

            if shuffle_datasets:
                random_state.shuffle(tasks)

        return tasks
