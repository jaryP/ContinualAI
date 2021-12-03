import numpy as np
import unittest
from continual_learning.datasets import MNIST

from continual_learning.datasets.base import create_dataset_with_dev_split, \
    DatasetView, DatasetSplits, DatasetSplitContexView, create_dataset_with_new_split, \
    SupervisedDataset, DatasetSubsetView
from continual_learning.scenarios.classification.utils import \
    get_dataset_subset_using_labels


class mnist_tests(unittest.TestCase):
    def test_classes(self):
        dataset = MNIST(download_if_missing=True,
                        data_folder='../downloaded_dataset/mnist/')

        print(MNIST.mro())
        print(isinstance(dataset, SupervisedDataset))
        print(type(dataset))
        print(super(MNIST))

    def test_loading(self):
        dataset = MNIST(download_if_missing=True,
                        data_folder='../downloaded_dataset/mnist/')

    def test_iteration(self):
        dataset = MNIST(download_if_missing=True,
                        data_folder='../downloaded_dataset/mnist/')

        for v, l in [('train', 60000), ('test', 10000), ('dev', 0)]:
            dataset.current_split = DatasetSplits(v)
            i = [_ for _, _, _ in dataset]
            self.assertTrue(len(i) == l)

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
                        transformer=lambda x: x + 1,
                        test_transformer=lambda x: x + 1)
        a = dataset[0][1]

        dataset = MNIST(download_if_missing=True,
                        data_folder='../downloaded_dataset/mnist/',
                        transformer=None)
        b = dataset[0][1]

        self.assertTrue((a != b).all())

    def test_dev_split(self):
        dataset = MNIST(download_if_missing=True,
                        data_folder='../downloaded_dataset/mnist/',
                        transformer=lambda x: x + 1,
                        test_transformer=lambda x: x + 1)

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
                        transformer=lambda x: x + 1,
                        test_transformer=lambda x: x + 1)

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
