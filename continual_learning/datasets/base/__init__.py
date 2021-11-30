from enum import unique, Enum
from aenum import MultiValueEnum


@unique
class DatasetSplits(MultiValueEnum):
    TRAIN = 0, 'train'
    TEST = 1, 'test'
    DEV = 2, 'dev'
    ALL = 3, 'all'


@unique
class DatasetType(Enum):
    SUPERVISED = 0


from .base import \
    SupervisedDataset, \
    UnsupervisedDataset, \
    IndexesContainer, \
    DatasetView

from .utils import \
    DownloadableDataset, \
    UnsupervisedDownloadableDataset, \
    SupervisedDownloadableDataset, \
    DatasetSplitView

from .split_functions import create_dataset_with_dev_split, \
    create_dataset_with_new_split
