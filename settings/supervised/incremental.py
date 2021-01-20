from typing import List, Union

import numpy as np

from datasets import SupervisedDataset
from settings.supervised.base import IncrementalProblem, ClassificationTask


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


class MultiTask(IncrementalProblem):
    def generate_tasks(self, dataset: SupervisedDataset, labels_per_task: int, shuffle_labels: bool = False,
                       random_state: Union[np.random.RandomState, int] = None) -> List[ClassificationTask]:

        dataset.apply_transformer(False)

        labels = dataset.labels
        labels_sets = get_labels_set(labels, labels_per_task=labels_per_task, shuffle_labels=shuffle_labels,
                                     random_state=random_state)

        labels_map = np.zeros(len(labels), dtype=int)

        for i in labels_sets:
            for j in range(len(i)):
                labels_map[i[j]] = j

        dataset.all()
        y = dataset.y

        tasks = []
        for task_labels in labels_sets:
            indexes = np.where(np.in1d(y, task_labels))[0]

            _, x, dataset_y = dataset[indexes]
            task_y = labels_map[dataset_y]

            train_i, test_i, dev_i = dataset.train_indices, dataset.test_indices, dataset.dev_indices
            train, dev, test = [], [], []

            for j, i in enumerate(indexes):
                if i in train_i:
                    train.append(j)
                elif dev_i is not None and i in dev_i:
                    dev.append(j)
                elif test_i is not None and i in test_i:
                    test.append(j)
                else:
                    assert False

            # train = list(filter(lambda z: z in indexes, dataset.train_indices))
            # test = list(filter(lambda z: z in indexes, dataset.test_indices))
            # dev = list(filter(lambda z: z in indexes, dataset.dev_indices))

            task = ClassificationTask(x=x, dataset_y=dataset_y, task_y=task_y, train=train, test=test, dev=dev,
                                      task_index=len(tasks),
                                      transformer=dataset.transformer, target_transformer=dataset.target_transformer)

            tasks.append(task)

        return tasks


class SingleIncrementalTask(IncrementalProblem):

    def generate_tasks(self, dataset: SupervisedDataset, labels_per_task: int, shuffle_labels: bool = False,
                       random_state: Union[np.random.RandomState, int] = None) -> List[ClassificationTask]:

        labels = dataset.labels
        labels_sets = get_labels_set(labels, labels_per_task=labels_per_task, shuffle_labels=shuffle_labels,
                                     random_state=random_state)

        labels_map = np.zeros(len(labels), dtype=int)

        offset = 0
        for i in labels_sets:
            for j in range(len(i)):
                labels_map[i[j]] = j + offset
            offset += len(i)

        dataset.all()
        y = dataset.y

        tasks = []
        for task_labels in labels_sets:
            indexes = np.where(np.in1d(y, task_labels))[0]

            _, x, dataset_y = dataset[indexes]
            task_y = labels_map[dataset_y]

            train = list(filter(lambda z: z in indexes, dataset.train_indices))
            test = list(filter(lambda z: z in indexes, dataset.test_indices))
            dev = list(filter(lambda z: z in indexes, dataset.dev_indices))

            task = ClassificationTask(x=x, dataset_y=dataset_y, task_y=task_y, train=train, test=test, dev=dev,
                                      transformer=dataset._transformer, target_transformer=dataset.target_transformer)

            tasks.append(task)

        return tasks
