from enum import unique, Enum

from PIL import Image


def path_image_loading(path):
    return Image.open(path).convert('RGB')

@unique
class DatasetSplits(Enum):
    TRAIN = 'train'
    TEST = 'test'
    DEV = 'dev'
    ALL = 'all'


@unique
class DatasetType(Enum):
    SUPERVISED = 0
    UNSUPERVISED = 1


@unique
class DatasetProblem(Enum):
    CLASSIFICATION = 0
    REGRESSION = 1

from .base import \
    SupervisedDataset, \
    UnsupervisedDataset, \
    IndexesContainer

from .dataset_definition import AbstractDataset, \
    DatasetSplitsContainer, \
    BaseDataset, \
    DatasetSubset

from .utils import  DownloadableDataset\
    , \
    UnsupervisedDownloadableDataset, \
    SupervisedDownloadableDataset, \
    DatasetSplitContexView

from .split_functions import create_dataset_with_dev_split, \
    create_dataset_with_new_split
