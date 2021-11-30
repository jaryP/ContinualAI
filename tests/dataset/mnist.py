import numpy as np
import unittest
from continual_learning.datasets import MNIST

from continual_learning.datasets.base import create_dataset_with_dev_split, \
    DatasetView, DatasetSplits, DatasetSplitView, create_dataset_with_new_split


class mnist_tests(unittest.TestCase):
    def test_loading(self):
        dataset = MNIST(download_if_missing=True,
                        data_folder='../downloaded_dataset/mnist/')
        dataset.all()
        self.assertTrue(len(dataset) == 70000)
        dataset.train()
        self.assertTrue(len(dataset) == 60000)
        dataset.test()
        self.assertTrue(len(dataset) == 10000)

    def test_context_split(self):
        dataset = MNIST(download_if_missing=True,
                        data_folder='../downloaded_dataset/mnist/')

        dataset.train()
        ln = len(dataset)

        with DatasetSplitView(dataset, DatasetSplits.TEST):
            ln1 = len(dataset)

        self.assertTrue(ln != ln1)

    def test_split_view(self):
        dataset = MNIST(download_if_missing=True,
                        data_folder='../downloaded_dataset/mnist/')
        dataset.train()

        view = DatasetView(dataset, DatasetSplits.TEST)

        ln = len(dataset)
        ln1 = len(view)

        self.assertTrue(ln != ln1)

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
