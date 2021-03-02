from typing import Union, List

import numpy as np

from continual_learning.benchmarks import SupervisedDataset, DatasetSplits
from continual_learning.scenarios.base import IncrementalSupervisedProblem
from continual_learning.scenarios.tasks import SupervisedTask
from continual_learning.scenarios.supervised.utils import get_labels_set


class MultiTask(IncrementalSupervisedProblem):
    def generate_tasks(self,
                       dataset: SupervisedDataset,
                       labels_per_task: int,
                       shuffle_labels: bool = False,
                       random_state: Union[np.random.RandomState, int] = None,
                       **kwargs) -> List[SupervisedTask]:

        if labels_per_task <= 1:
            raise ValueError('Parameter labels_per_task must be > 1 '
                             '{}'.format(labels_per_task))

        labels = dataset.labels
        labels_sets = get_labels_set(labels=labels,
                                     labels_per_set=labels_per_task,
                                     shuffle_labels=shuffle_labels,
                                     random_state=random_state)

        if len(labels_sets[-1]) == 1:
            labels_sets[-2].extend(labels_sets[-1])
            labels_sets = labels_sets[:-1]

        labels_map = np.zeros(len(labels), dtype=int)

        for i in labels_sets:
            for j in range(len(i)):
                labels_map[i[j]] = j

        tasks = []

        for task_labels in labels_sets:
            lm = {i: labels_map[i] for i in task_labels}

            train_indexes = \
                np.arange(len(dataset.get_indexes(DatasetSplits.TRAIN)))[
                    np.in1d(dataset.y(DatasetSplits.TRAIN), task_labels)]

            test_indexes = \
                np.arange(len(dataset.get_indexes(DatasetSplits.TEST)))[
                    np.in1d(dataset.y(DatasetSplits.TEST), task_labels)]

            dev_indexes = \
                np.arange(len(dataset.get_indexes(DatasetSplits.DEV)))[
                    np.in1d(dataset.y(DatasetSplits.DEV), task_labels)]

            task = SupervisedTask(base_dataset=dataset,
                                  train=train_indexes,
                                  test=test_indexes,
                                  dev=dev_indexes,
                                  labels_mapping=lm,
                                  index=len(tasks))

            tasks.append(task)

        return tasks
