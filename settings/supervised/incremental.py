from typing import List, Union

import numpy as np

from datasets import SupervisedDataset
from settings.supervised.base import IncrementalProblem, ClassificationTask


class MultiTask(IncrementalProblem):
    def generate_tasks(self, dataset: SupervisedDataset, labels_per_task: int, shuffle_labels: bool = False,
                       random_state: Union[np.random.RandomState, int] = None) -> List[ClassificationTask]:
        """
            def generate_tasks(self, dataset, labels_per_task: int, batch_size: int, shuffle_labels: bool = True):

                labels = dataset.labels

                if shuffle_labels:
                    self.RandomState.shuffle(dataset.labels)

                labels_sets = [list(labels[i:i + labels_per_task]) for i in range(0, len(labels), labels_per_task)]

                if len(labels_sets[-1]) == 1:
                    labels_sets[-2].append(labels_sets[-1][0])
                    labels_sets = labels_sets[:-1]

                labels_sets = np.asarray(labels_sets)

                labels_map = np.zeros(len(labels), dtype=int)

                for i in labels_sets:
                    for j in range(len(i)):
                        labels_map[i[j]] = j

                for i, ls in enumerate(labels_sets):
                    self._tasks.append(ClassificationTask(batch_size=batch_size, base_dataset=dataset,
                                                          labels_map=labels_map, task_labels=ls))
        """
        tasks = []
        labels = dataset.labels

        if shuffle_labels:
            if random_state is not None:
                if isinstance(random_state, int):
                    random_state = np.random.RandomState(random_state)
            random_state.shuffle(dataset.labels)

        labels_sets = [list(labels[i:i + labels_per_task]) for i in range(0, len(labels), labels_per_task)]

        if len(labels_sets[-1]) == 1:
            labels_sets[-2].extend(labels_sets[-1])
            labels_sets = labels_sets[:-1]

        labels_map = np.zeros(len(labels), dtype=int)

        for i in labels_sets:
            for j in range(len(i)):
                labels_map[i[j]] = j

        for task_labels in labels_sets:
            tx, ty, dy = [], [], []
            indexes = []

            for i, x, y in dataset:
                if y in task_labels:
                    dy.append(y)
                    ty.append(labels_map[y])
                    tx.append(x)

                    indexes.append(i)

            train = list(filter(lambda z: z in indexes, dataset.train_split))
            test = list(filter(lambda z: z in indexes, dataset.test_split))
            dev = list(filter(lambda z: z in indexes, dataset.dev_split))

            task = ClassificationTask(x=tx, dataset_y=dy, task_y=ty, train=train, test=test, dev=dev,
                                      transformer=dataset.transformer, target_transformer=dataset.target_transformer)
            tasks.append(task)

        return tasks




class SingleIncrementalTask(IncrementalProblem):
    def generate_tasks(self, dataset: SupervisedDataset, labels_per_task: int, shuffle_labels: bool = False,
                       random_state: Union[np.random.RandomState, int] = None) -> List[ClassificationTask]:
        pass
