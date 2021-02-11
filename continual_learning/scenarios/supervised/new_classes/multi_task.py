from typing import Union, List

import numpy as np

from continual_learning.banchmarks import SupervisedDataset, DatasetSplits
from continual_learning.scenarios.base import SupervisedTask
from continual_learning.scenarios.supervised.base import IncrementalProblem, get_labels_set


class MultiTask(IncrementalProblem):
    def generate_tasks(self, dataset: SupervisedDataset, labels_per_task: int, shuffle_labels: bool = False,
                       random_state: Union[np.random.RandomState, int] = None) -> List[SupervisedTask]:

        labels = dataset.labels
        labels_sets = get_labels_set(labels, labels_per_task=labels_per_task, shuffle_labels=shuffle_labels,
                                     random_state=random_state)

        labels_map = np.zeros(len(labels), dtype=int)

        for i in labels_sets:
            for j in range(len(i)):
                labels_map[i[j]] = j

        tasks = []

        for task_labels in labels_sets:

            lm = {i: labels_map[i] for i in task_labels}

            train_indexes = np.arange(len(dataset.get_indexes(DatasetSplits.TRAIN)))[
                np.in1d(dataset.y(DatasetSplits.TRAIN), task_labels)]

            test_indexes = np.arange(len(dataset.get_indexes(DatasetSplits.TEST)))[
                np.in1d(dataset.y(DatasetSplits.TEST), task_labels)]

            dev_indexes = np.arange(len(dataset.get_indexes(DatasetSplits.DEV)))[
                np.in1d(dataset.y(DatasetSplits.DEV), task_labels)]

            task = SupervisedTask(base_dataset=dataset, train=train_indexes, test=test_indexes, dev=dev_indexes,
                                  labels_mapping=lm, index=len(tasks))

            tasks.append(task)

        return tasks