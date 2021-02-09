from typing import Union, List

import numpy as np

from continual_learning.banchmarks import SupervisedDataset, DatasetSplits
from continual_learning.scenarios.base import SupervisedTask
from continual_learning.scenarios.supervised_new.base import IncrementalProblem


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
                       random_state: Union[np.random.RandomState, int] = None) -> List[SupervisedTask]:

        labels = dataset.labels
        labels_sets = get_labels_set(labels, labels_per_task=labels_per_task, shuffle_labels=shuffle_labels,
                                     random_state=random_state)

        labels_map = np.zeros(len(labels), dtype=int)

        for i in labels_sets:
            for j in range(len(i)):
                labels_map[i[j]] = j

        # dataset.all()
        # y = dataset.y

        # is_path_dataset = dataset.is_path_dataset
        # images_path = dataset.images_path

        tasks = []

        for task_labels in labels_sets:

            lm = {i: labels_map[i] for i in task_labels}

            train_indexes = np.arange(len(dataset.get_indexes(DatasetSplits.TRAIN)))[
                np.in1d(dataset.y(DatasetSplits.TRAIN), task_labels)]

            test_indexes = np.arange(len(dataset.get_indexes(DatasetSplits.TEST)))[
                np.in1d(dataset.y(DatasetSplits.TEST), task_labels)]

            dev_indexes = np.arange(len(dataset.get_indexes(DatasetSplits.DEV)))[
                np.in1d(dataset.y(DatasetSplits.DEV), task_labels)]

            # print(len(train_indexes), len(test_indexes), len(dev_indexes))
            # dataset.test()
            # test_indexes = dataset.current_indexes[np.in1d(dataset.y, task_labels)]
            #
            # dataset.dev()
            # dev_indexes = dataset.current_indexes[np.in1d(dataset.y, task_labels)]
            # print(lm)

            task = SupervisedTask(base_dataset=dataset, train=train_indexes, test=test_indexes, dev=dev_indexes,
                                  labels_mapping=lm, index=len(tasks))

            # task.train()
            # print(task.labels_mapping)
            #
            # task.set_task_labels()
            # print(task.labels)
            #
            # task.set_dataset_labels()
            # print(task.labels)
            #
            # task.all()
            # print(len(task))
            #
            # for i in range(len(task)):
            #     a = task[i]
            #
            # print()

            tasks.append(task)

        print(sum([len(t.get_indexes(DatasetSplits.ALL)) for t in tasks]))
        # for task_labels in labels_sets:
        #     indexes = np.where(np.in1d(y, task_labels))[0]
        #     if is_path_dataset:
        #         x, dataset_y = [os.path.join(dataset.images_path, dataset._x[item]) for item in indexes], \
        #                        dataset._target_transformer(dataset._y[dataset.current_indices[indexes]])
        #         x = np.array(x)
        #     else:
        #         _, x, dataset_y = dataset[indexes]
        #
        #     task_y = labels_map[dataset_y]
        #
        #     train_i, test_i, dev_i = dataset.train_indices, dataset.test_indices, dataset.dev_indices
        #     train, dev, test = [], [], []
        #
        #     for j, i in enumerate(indexes):
        #         if i in train_i:
        #             train.append(j)
        #         elif dev_i is not None and i in dev_i:
        #             dev.append(j)
        #         elif test_i is not None and i in test_i:
        #             test.append(j)
        #         else:
        #             assert False
        #
        #     task = ClassificationTask(x=x, dataset_y=dataset_y, task_y=task_y, train=train, test=test, dev=dev,
        #                               task_index=len(tasks),
        #                               transformer=dataset.transformer, target_transformer=dataset.target_transformer)
        #
        #     tasks.append(task)

        return tasks


if __name__ == '__main__':
    from continual_learning.banchmarks import MNIST

    d = MNIST(download_if_missing=True, data_folder='continual_learning/tests/downloaded_dataset')
    MultiTask(dataset=d, labels_per_task=2)