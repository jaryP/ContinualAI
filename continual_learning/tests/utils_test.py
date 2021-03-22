import os
import unittest

from continual_learning.benchmarks import MNIST
from continual_learning.benchmarks.base.utils import ConcatDataset


class datasets_concatenation_tests(unittest.TestCase):
    # dataset_class = MNIST
    dataset_class = MNIST

    def setUp(self) -> None:
        self.data_folder = os.path.join('downloaded_dataset', self.dataset_class.__name__)

    def test_concatenation(self):
        dataset1 = self.dataset_class(download_if_missing=True, data_folder=self.data_folder)
        dataset2 = self.dataset_class(download_if_missing=True, data_folder=self.data_folder)
        cd = ConcatDataset([dataset2, dataset1])
        self.assertTrue(len(cd) == 120000)
        cd.test()
        self.assertTrue(len(cd) == 20000)

    def test_iterations(self):
        dataset1 = self.dataset_class(download_if_missing=True, data_folder=self.data_folder)
        dataset2 = self.dataset_class(download_if_missing=True, data_folder=self.data_folder)
        cd = ConcatDataset([dataset2, dataset1])

        cd.train()
        i = 0
        for i, _ in enumerate(cd):
            pass
        self.assertTrue(i + 1 == 120000)

        cd.test()
        i = 0
        for i, _ in enumerate(cd):
            pass
        self.assertTrue(i + 1 == 20000)

    def test_dataloader(self):
        def modify_dataset(d):
            dataset2 = self.dataset_class(download_if_missing=True, data_folder=self.data_folder)
            d = ConcatDataset([d, dataset2])

        dataset1 = self.dataset_class(download_if_missing=True, data_folder=self.data_folder)
        # dataset2 = self.dataset_class(download_if_missing=True, data_folder=self.data_folder)
        # cd = ConcatDataset([dataset2, dataset1])
        print(len(dataset1))
        modify_dataset(dataset1)
        print(len(dataset1))
        
        # cd.train()
        # i = 0
        # for j, _, _ in cd.get_iterator(128):
        #     i += len(j)
        # self.assertTrue(i == 120000)
        #
        # cd.test()
        # i = 0
        # for j, _, _ in cd.get_iterator(128):
        #     i += len(j)
        # self.assertTrue(i == 20000)
