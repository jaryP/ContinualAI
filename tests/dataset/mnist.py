from collections import defaultdict

import numpy as np
import unittest
from continual_learning.datasets import MNIST

# from continual_learning.datasets.base import create_dataset_with_dev_split, \
#     DatasetView, DatasetSplitContexView, create_dataset_with_new_split, \
#     SupervisedDataset, DatasetSubsetView, DatasetSplits
# from continual_learning.scenarios.classification.utils import \
#     get_dataset_subset_using_labels
from continual_learning.datasets.base import DatasetSplits, \
    create_dev_dataset


class mnist_tests(unittest.TestCase):
    def test_loading(self):
        dataset = MNIST(download_if_missing=True,
                        data_folder='../downloaded_dataset/mnist/')

    def test_iteration(self):
        dataset = MNIST(download_if_missing=True,
                        data_folder='../downloaded_dataset/mnist/')

        for v, l in [('train', 60000), ('test', 10000), ('dev', 0)]:
            dataset.current_split = DatasetSplits(v)
            print(len(dataset))
            self.assertTrue(len(dataset) == l)

        self.assertTrue(len(dataset.test_split()) == 10000)
        self.assertTrue(len(dataset.train_split()) == 60000)
        self.assertTrue(len(dataset.dev_split()) == 0)

        dataset.train()
        print(dataset[0][1].shape)

    def test_add_dev(self):
        from collections import Counter

        dataset = MNIST(download_if_missing=True,
                        data_folder='../downloaded_dataset/mnist/')
        print(Counter(dataset.get_split('train').targets))

        dataset = create_dev_dataset(dataset, 0.1,
                                     from_test=False,
                                     balanced=True)

        lens = defaultdict(int)
        for s in ['dev', 'train', 'test']:
            split = dataset.get_split(s)
            ln = len([_ for i, _ in enumerate(split)])
            lens[s] = ln
            print(lens)
            ctr = Counter(split.targets)
            print(s, sum(ctr.values()), ctr, len(split.targets))

        dataset = MNIST(download_if_missing=True,
                        data_folder='../downloaded_dataset/mnist/')
        train, test, dev = create_dev_dataset(dataset, 0.1,
                                              from_test=False,
                                              balanced=True, as_container=False)
        print()
        print(Counter(train.targets))

        for split in [train, test, dev]:
            ln = len([_ for i, _ in enumerate(split)])
            lens[s] = ln
            print(lens)
            ctr = Counter(split.targets)
            print(sum(ctr.values()), ctr, len(split.targets))

    def test_subset(self):
        dataset = MNIST(download_if_missing=True,
                        data_folder='../downloaded_dataset/mnist/')

        ln = dataset.get_split_len(DatasetSplits.ALL)

        subset = dataset.get_subset([0, 1], [0], None)

        ln1 = subset.get_split_len(DatasetSplits.ALL)

        self.assertTrue(ln1 == 3)
        subset.current_split = DatasetSplits.DEV

        task = get_dataset_subset_using_labels(dataset, [0, 1])
        task.current_split = DatasetSplits.ALL
        print(task.labels)
        print(len(task))

    def test_context_split(self):
        dataset = MNIST(download_if_missing=True,
                        data_folder='../downloaded_dataset/mnist/')

        dataset.train()
        ln = len(dataset)

        with DatasetSplitContexView(dataset, DatasetSplits.TEST):
            ln1 = len(dataset)

        self.assertTrue(ln != ln1)

    def test_split_view(self):
        dataset = MNIST(download_if_missing=True,
                        data_folder='../downloaded_dataset/mnist/')
        dataset.train()

        view = DatasetView(dataset, DatasetSplits.TEST)

        a = [view[i][0] for i in range(len(view))]
        self.assertTrue(len(a) == len(dataset.get_indexes(DatasetSplits.TEST)))

        ln = len(dataset)
        ln1 = len(view)

        self.assertTrue(ln != ln1)

    def test_subset_view(self):
        dataset = MNIST(download_if_missing=True,
                        data_folder='../downloaded_dataset/mnist/')
        dataset.train()

        ti = dataset.get_indexes(DatasetSplits.TEST)
        tri = dataset.get_indexes(DatasetSplits.TRAIN)
        di = dataset.get_indexes(DatasetSplits.DEV)

        subset = DatasetSubsetView(dataset, train_subset=tri[:10])

        self.assertTrue(len(subset) == 10)
        self.assertTrue(len(subset.get_indexes(DatasetSplits.TEST)) == 0)
        self.assertTrue(len(subset.get_indexes(DatasetSplits.DEV)) == 0)

        a = [subset[i][0] for i in range(len(subset))]
        self.assertTrue(len(a) == 10)

    def test_transform(self):
        dataset = MNIST(download_if_missing=True,
                        data_folder='../downloaded_dataset/mnist/',
                        transform=lambda x: x + 1,
                        test_transform=lambda x: x + 1)
        a = dataset[0][1]

        dataset = MNIST(download_if_missing=True,
                        data_folder='../downloaded_dataset/mnist/',
                        transform=None)
        b = dataset[0][1]

        self.assertTrue((a != b).all())

    def test_dev_split(self):
        dataset = MNIST(download_if_missing=True,
                        data_folder='../downloaded_dataset/mnist/',
                        transform=lambda x: x + 1,
                        test_transform=lambda x: x + 1)

        new_dataset = create_dataset_with_dev_split(dataset=dataset)

        dataset.all()
        new_dataset.all()

        # a = dataset[0][1]
        #
        # dataset = MNIST(download_if_missing=True, data_folder='../downloaded_dataset/mnist/',
        #                 transformer=None)
        # b = dataset[0][1]

        self.assertTrue(len(dataset) == len(new_dataset))
        self.assertTrue(len(dataset.get_indexes(DatasetSplits.DEV))
                        < len(new_dataset.get_indexes(DatasetSplits.DEV)))

    def test_new_splits(self):
        dataset = MNIST(download_if_missing=True,
                        data_folder='../downloaded_dataset/mnist/',
                        transform=lambda x: x + 1,
                        test_transform=lambda x: x + 1)

        new_dataset = create_dataset_with_new_split(dataset=dataset)

        dataset.all()
        new_dataset.all()

        # a = dataset[0][1]
        #
        # dataset = MNIST(download_if_missing=True, data_folder='../downloaded_dataset/mnist/',
        #                 transformer=None)
        # b = dataset[0][1]

        print(len(new_dataset.get_indexes(DatasetSplits.TEST)),
              len(new_dataset.get_indexes(DatasetSplits.TRAIN)))

        print(len(dataset.get_indexes(DatasetSplits.TEST)),
              len(dataset.get_indexes(DatasetSplits.TRAIN)))

        self.assertTrue(len(dataset) == len(new_dataset))
        # self.assertTrue(len(dataset.get_indexes(DatasetSplits.DEV))
        #                 < len(new_dataset.get_indexes(DatasetSplits.DEV)))
