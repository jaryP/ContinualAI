import os
import unittest

import torchvision
from torch import nn

from continual_learning.banchmarks import *


class MNIST_tests(unittest.TestCase):
    dataset_class = MNIST
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                                nn.Flatten(0)
                                                ])

    def setUp(self) -> None:
        self.data_folder = os.path.join('downloaded_dataset', self.dataset_class.__name__)
        
    def test_download(self):
        try:
            self.dataset_class(download_if_missing=False, data_folder='.')
        except OSError:
            self.assertTrue(True)
            return
        self.assertTrue(False)

    def test_iterate_subset(self):
        d = self.dataset_class(download_if_missing=True, transformer=self.transform, data_folder=self.data_folder)
        for v in DatasetSplits:
            d.current_split = v
            if len(d) > 0:
                for _ in d.get_iterator(batch_size=128, pin_memory=False, num_workers=0):
                    pass

    def test_iteration(self):
        d = self.dataset_class(download_if_missing=True, data_folder=self.data_folder)
        d.test()
        i = 0
        for _, x, y in d:
            i += 1
        self.assertTrue(i == len(d))
        d.all()
        for i, (_, x, y) in enumerate(d):
            pass
        self.assertTrue(i + 1 == len(d))

    def test_loading(self):
        s = None
        try:
            d = self.dataset_class(download_if_missing=True, data_folder=self.data_folder)
            for s in DatasetSplits:
                d.current_split = s
                d.x
                d.y
        except Exception as e:
            print(s, e)
            self.assertTrue(False)

        self.assertTrue(True)

    def test_dev_split(self):
        d = self.dataset_class(download_if_missing=True, data_folder=self.data_folder)
        d.create_dev_split(dev_split=0.1)
        d.all()
        len_all = len(d)
        self.assertTrue(len(d.get_indexes(DatasetSplits.TRAIN)) +
                        len(d.get_indexes(DatasetSplits.TEST)) +
                        len(d.get_indexes(DatasetSplits.DEV)) == len_all)

    def test_random_split(self):
        d = self.dataset_class(download_if_missing=True, data_folder=self.data_folder)
        d.split_dataset(test_split=0.5, dev_split=0.1)
        d.all()
        len_all = len(d)
        self.assertTrue(len(d.get_indexes(DatasetSplits.TRAIN)) +
                        len(d.get_indexes(DatasetSplits.TEST)) +
                        len(d.get_indexes(DatasetSplits.DEV)) == len_all)

    def test_slice(self):
        d = self.dataset_class(download_if_missing=True, data_folder=self.data_folder)
        for v in DatasetSplits:
            d.current_split = v
            if len(d) > 0:
                a = d[[1, 2, 3, 4]][0]
        self.assertTrue(True)

    def test_slice_transform(self):
        tt = lambda a: a + 1
        d = self.dataset_class(download_if_missing=True, data_folder=self.data_folder, transformer=self.transform,
                               target_transformer=tt)
        try:
            for v in DatasetSplits:
                d.current_split = v
                if len(d) > 0:
                    i, x, y = d[[1, 2, 3, 4]]
        except Exception as e:
            self.assertTrue(False)
        self.assertTrue(True)

    def test_dataloader(self):
        tt = lambda a: a + 1
        d = self.dataset_class(download_if_missing=True, data_folder=self.data_folder, transformer=self.transform,
                               target_transformer=tt)
        try:
            dataset = d.get_iterator(batch_size=64)
            for i in dataset:
                pass
        except Exception as e:
            self.assertTrue(False)

        self.assertTrue(True)

    def test_transformers(self):
        t = lambda x: x * 0 - 1000
        tt = lambda x: x * 0
        d = self.dataset_class(download_if_missing=True, transformer=t, test_transformer=tt,
                               data_folder=self.data_folder)
        _, x, y = d[0]
        d.test()
        _, test_x, _ = d[0]
        # print(x)
        # print(test_x)
        self.assertTrue((x == test_x).sum() == 0)

        d = self.dataset_class(download_if_missing=True, transformer=None, data_folder=self.data_folder)

        _, x2, _ = d[0]
        self.assertTrue((x == x2).sum() == 0)


class KMNIST_tests(MNIST_tests):
    dataset_class = KMNIST


class K49MNIST_tests(MNIST_tests):
    dataset_class = K49MNIST


class CIFAR10_tests(MNIST_tests):
    dataset_class = CIFAR10
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.50, 0.50, 0.50,),
                                                                                 (0.50, 0.50, 0.50))])


class CIFAR100_tests(CIFAR10_tests):
    dataset_class = CIFAR100


class SVHN_tests(MNIST_tests):
    dataset_class = SVHN
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.50, 0.50, 0.50,),
                                                                                 (0.50, 0.50, 0.50))])

class CORE50_tests(MNIST_tests):
    dataset_class = Core50_128
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.50, 0.50, 0.50,),
                                                                                 (0.50, 0.50, 0.50))])


if __name__ == '__main__':
    unittest.main()

#     # suite = unittest.TestSuite()
#     loader = unittest.TestLoader()
#     suites_list = []
#
#     for dataset, t, tt in [
#         (MNIST, torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
#                                                 torchvision.transforms.Normalize((0.1307,), (0.3081,)),
#                                                 nn.Flatten(0)
#                                                 ]), lambda x: x + 1)
#     ]:
# #         # unittest.main(dataset_tests(MNIST))
#         suite = loader.loadTestsFromTestCase(dataset_tests)
#         suites_list.append(suite)
#         # suite.addTest(dataset_tests(dataset, t, tt))
#
#     big_suite = unittest.TestSuite(suites_list)
#
#     runner = unittest.TextTestRunner()
#     results = runner.run(big_suite)
#
#     # unittest.TextTestRunner(verbosity=2).run(suite)
#         # suite = dataset_tests()
#         #
#         # dataset_tests.dataset_class = dataset
#         #
#         # dataset_tests.transform = t
#         # dataset_tests.target_transform = tt
#         #
#         # # unittest.main()
#         # runner = unittest.TextTestRunner()
#         # runner.run(suite)
