__all__ = ['mnist', 'cifar',
           'UnsupervisedDataset', 'SupervisedDataset',
           'split_dataset',
           'SupervisedDownloadableDataset', 'UnsupervisedDownloadableDataset']
#
from .base import UnsupervisedDataset, SupervisedDataset, DownloadableDataset, UnsupervisedDownloadableDataset, \
    SupervisedDownloadableDataset
from .utils import split_dataset

# import \.mnist
# import cifar
