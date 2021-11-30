import os
import unittest

import torchvision
from torch import nn

from continual_learning.benchmarks import MNIST, DatasetSplits
from continual_learning.scenarios.supervised.supervised_train_supervised_test.multi_task import MultiTask


class Multi_Task_tests(unittest.TestCase):
    dataset_class = MNIST
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                                nn.Flatten(0)
                                                ])

    def setUp(self) -> None:
        self.data_folder = os.path.join('downloaded_dataset', self.dataset_class.__name__)

    def test_creation(self):
        dataset = self.dataset_class(download_if_missing=True, data_folder=self.data_folder)

        mt = MultiTask(dataset=dataset, labels_per_task=2)
        ln = len(dataset.get_indexes(DatasetSplits.ALL))
        s = 0
        ss = []

        for t in mt:
            l = len(t.get_indexes(DatasetSplits.ALL))
            l1 = sum([len(t.get_indexes(s)) for s in DatasetSplits if s != DatasetSplits.ALL])
            self.assertTrue(l == l1)

            ss.append(set(t.get_indexes(DatasetSplits.ALL)))

            s += len(t.get_indexes(DatasetSplits.ALL))

        self.assertTrue(sum(map(len, ss)) == ln)
        self.assertTrue(ln == s)

    def test_creation_after_split(self):
        dataset = self.dataset_class(download_if_missing=True, data_folder=self.data_folder)
        dataset.create_dev_split(dev_split=0.1)

        mt = MultiTask(dataset=dataset, labels_per_task=2)
        ln = len(dataset.get_indexes(DatasetSplits.ALL))
        s = 0
        ss = []

        for t in mt:
            l = len(t.get_indexes(DatasetSplits.ALL))
            l1 = sum([len(t.get_indexes(s)) for s in DatasetSplits if s != DatasetSplits.ALL])
            self.assertTrue(l == l1)

            s += len(t.get_indexes(DatasetSplits.ALL))
            ss.append(set(t.get_indexes(DatasetSplits.ALL)))

        self.assertTrue(sum(map(len, ss)) == ln)
        self.assertTrue(ln == s)

    def test_transformer(self):
        dataset = self.dataset_class(download_if_missing=True, data_folder=self.data_folder,
                                     transformer=self.transform)
        dataset.create_dev_split(dev_split=0.1)

        mt = MultiTask(dataset=dataset, labels_per_task=2)
        ln = len(dataset.get_indexes(DatasetSplits.ALL))
        s = 0
        for t in mt:
            l = len(t.get_indexes(DatasetSplits.ALL))
            l1 = sum([len(t.get_indexes(s)) for s in DatasetSplits if s != DatasetSplits.ALL])
            self.assertTrue(l == l1)

            s += len(t.get_indexes(DatasetSplits.ALL))
            t.all()
            for _ in t.get_iterator(batch_size=129):
                pass

        self.assertTrue(ln == s)

    def test_iteration(self):
        dataset = self.dataset_class(download_if_missing=True, data_folder=self.data_folder,
                                     transformer=self.transform)
        mt = MultiTask(dataset=dataset, labels_per_task=2)
        # ln = len(dataset.get_indexes(DatasetSplits.ALL))
        # s = 0

        for t in mt:
            t.train()
            if len(t) > 0:
                for _ in t.get_iterator(batch_size=128):
                    pass
            t.test()
            if len(t) > 0:
                for _ in t.get_iterator(batch_size=128):
                    pass
            t.dev()
            if len(t) > 0:
                for _ in t.get_iterator(batch_size=128):
                    pass

            # l = len(t.get_indexes(DatasetSplits.ALL))
            # l1 = sum([len(t.get_indexes(s)) for s in DatasetSplits if s != DatasetSplits.ALL])
            # self.assertTrue(l == l1)
            #
            # s += len(t.get_indexes(DatasetSplits.ALL))

        self.assertTrue(True)
