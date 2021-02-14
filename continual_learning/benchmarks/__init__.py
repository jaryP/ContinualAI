# __all__ = ['UnsupervisedDataset', 'SupervisedDataset',
#            'split_dataset']
#
from .base import UnsupervisedDataset, SupervisedDataset, DatasetSplits
from .utils import DownloadableDataset, UnsupervisedDownloadableDataset, SupervisedDownloadableDataset
from .split_functions import split_dataset

from .datasets.mnist import MNIST, K49MNIST, KMNIST
from .datasets.cifar import CIFAR10, CIFAR100
from .datasets.core50 import Core50_128
from .datasets.svhn import SVHN
from .datasets.tiny_imagenet import TinyImagenet

