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
    SUPERVISED = 'supervised'
    UNSUPERVISED = 'unsupervised'


@unique
class DatasetProblem(Enum):
    CLASSIFICATION = 0
    REGRESSION = 1


from .dataset_definition import AbstractDataset, \
    DatasetSplitsContainer, \
    BaseDataset, \
    DatasetSubset, \
    DownloadableDataset

# from .utils import  DownloadableDataset\
#     , \
#     UnsupervisedDownloadableDataset, \
#     SupervisedDownloadableDataset, \
#     DatasetSplitContexView

from .split_functions import create_dataset_with_new_split, \
    create_dev_dataset
