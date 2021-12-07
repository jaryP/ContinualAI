import unittest

import numpy as np

from continual_learning.datasets.base import BaseDataset, \
    DatasetSplitsContainer, create_dataset_with_new_split
from continual_learning.scenarios.tasks import Task


class base_test(unittest.TestCase):
    def get_dataset(self, n, target=False):
        x = np.random.uniform(-1, 10, n)
        y = None
        if target:
            y = np.random.randint(0, 10, n)
        return x, y

    def test_instantiating(self):
        x, y = self.get_dataset(100)

        dataset = BaseDataset(values=x)

        self.assertTrue(dataset.classes is None)
        self.assertTrue(dataset.targets is None)

        sample = dataset[0]
        all = [i for i in dataset]

        self.assertTrue(len(all) == 100)
        self.assertTrue(len(dataset) == 100)
        self.assertTrue(len(sample) == 2)

    def test_subset_extraction(self):
        x, y = self.get_dataset(100)

        dataset = BaseDataset(values=x)

        subset = dataset.get_subset([0, 1, 2])
        self.assertTrue(isinstance(subset, BaseDataset))

        sample = subset[0]

        all = [i for i in subset]

        self.assertTrue(len(all) == 3)
        self.assertTrue(len(subset) == 3)
        self.assertTrue(len(sample) == 2)

    def test_transform(self):
        x, y = self.get_dataset(100)
        t = lambda x: 0

        dataset = BaseDataset(values=x, transform=t)

        self.assertTrue(all([i[1] == 0 for i in dataset]))
        self.assertTrue(not all([i == 0 for i in dataset.values]))

    def test_subset_from_dataset(self):
        x, y = self.get_dataset(100)
        train_subset = [0, 1, 20, 33]
        test_subset = [0, 1, 20, 33]

        dataset = BaseDataset(values=x, transform=None)
        splitted_dataset = DatasetSplitsContainer(base_dataset=dataset,
                                                  train=train_subset,
                                                  test=test_subset)
        # splitted_dataset.train()
        # print(len(splitted_dataset))
        # splitted_dataset.test()
        # print(len(splitted_dataset))

        self.assertTrue(all([dataset[i][1] == x[i]
                             for i, v in enumerate([0, 1, 20, 33])]))

        train, test, dev = splitted_dataset.get_subset([0, 1])

        all_train = [i for i in train]
        self.assertTrue(len(all_train) == 2)

        all_test = [i for i in test]
        self.assertTrue(len(all_test) == 0)

        sub_subset = splitted_dataset.get_subset([0, 1], test=[0],
                                                 as_splitted_dataset=
                                                 True)
        print(len(sub_subset.train_split()))

    def test_dev(self):
        x, y = self.get_dataset(100)
        dataset = BaseDataset(values=x, transform=None)

        train, test, dev = create_dataset_with_new_split(dataset,
                                                         dev_percentage=0.1)

        self.assertTrue(len(test) == 20 and len(train) == 70 and len(dev) == 10)

        train_subset = [0, 1, 20, 33, 5, 8, 22, 66, 88, 12, 64, 55]
        test_subset = [0, 1, 20, 33]

        splitted_dataset = DatasetSplitsContainer(base_dataset=dataset,
                                                  train=train_subset,
                                                  test=test_subset)

        tr, te, de = create_dataset_with_new_split(
            splitted_dataset, dev_percentage=0.1)

        print(len(tr), len(te), len(de))

    def test_da_elimiare(self):
        x, y = self.get_dataset(100, target=True)
        dataset = BaseDataset(values=x, transform=None)

        task = Task(base_dataset=dataset, task_index=0,
                    labels_mapping={i: i for i in set(y)})
        dataset.get_subset([0, 1])

        task.get_subset([0, 1])

        a = task.current_split