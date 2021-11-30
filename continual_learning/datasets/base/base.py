import warnings

from abc import ABC, abstractmethod
from typing import Callable, Union, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from . import DatasetSplits, DatasetType

IndexesType = Union[list, np.ndarray]


class Dataset(ABC):
    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self, item):
        raise NotImplementedError


class IndexesContainer(object):
    def __init__(self,
                 *,
                 train: IndexesType,
                 test: IndexesType = None,
                 dev: IndexesType = None,
                 **kwargs):

        if dev is None:
            dev = []
        if test is None:
            test = []

        self._splits = \
            {
                DatasetSplits.TRAIN: np.asarray(train, dtype=int),
                DatasetSplits.TEST: np.asarray(test, dtype=int),
                DatasetSplits.DEV: np.asarray(dev, dtype=int),
            }

        self._current_split = DatasetSplits.TRAIN

    def __len__(self):
        return len(self.current_indexes)

    @property
    def current_split(self) -> DatasetSplits:
        return self._current_split

    @current_split.setter
    def current_split(self, v: DatasetSplits) -> None:
        self._current_split = v

    @property
    def current_indexes(self) -> np.ndarray:
        return self.get_indexes(self.current_split)

    def get_indexes(self, v: DatasetSplits = None) -> np.ndarray:
        if v is None:
            return self.current_indexes
        if v == DatasetSplits.ALL:
            return np.concatenate((self._splits[DatasetSplits.TRAIN],
                                   self._splits[DatasetSplits.TEST],
                                   self._splits[DatasetSplits.DEV]))
        return self._splits[v]

    def train(self) -> None:
        self.current_split = DatasetSplits.TRAIN

    def test(self) -> None:
        self.current_split = DatasetSplits.TEST
        if len(self._splits[DatasetSplits.TEST]) == 0:
            warnings.warn('The dataset does not have Test split.',
                          RuntimeWarning)

    def dev(self) -> None:
        self.current_split = DatasetSplits.DEV
        if len(self._splits[DatasetSplits.DEV]) == 0:
            warnings.warn('The dataset does not have Development split.',
                          RuntimeWarning)

    def all(self) -> None:
        self.current_split = DatasetSplits.ALL


class AbstractDataset(ABC, IndexesContainer):
    def __init__(self, *,
                 train: IndexesType,
                 test: IndexesType = None,
                 dev: IndexesType = None,
                 transformer: Callable = None,
                 test_transformer: Callable = None,
                 is_path_dataset: bool = False,
                 images_path: str = None,
                 **kwargs):

        super().__init__(train=train, test=test, dev=dev, **kwargs)

        self.is_path_dataset = is_path_dataset
        if is_path_dataset and images_path is None:
            raise ValueError('If is_path_dataset=True, '
                             'then images_path must be not None')

        self.images_path = images_path

        if transformer is not None and test_transformer is None:
            raise ValueError('Train transformer provided '
                             'but test_transformer is None. '
                             'Please proved both or none. ')

        self.transformer = transformer \
            if transformer is not None else lambda z: z

        self.test_transformer = test_transformer \
            if test_transformer is not None else lambda z: z

    def _get_transformer(self, split: DatasetSplits = None):
        if split is None:
            split = self.current_split

        if split == DatasetSplits.TRAIN:
            return self.transformer
        else:
            return self.test_transformer

    @abstractmethod
    def __getitem__(self, item: Union[slice, int, list, np.ndarray]) -> \
            Tuple[Union[list, tuple, int, list], Union[np.ndarray, list]]:
        raise NotImplementedError

    def get_iterator(self, batch_size, shuffle=True, sampler=None,
                     num_workers=0, pin_memory=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          sampler=sampler, pin_memory=pin_memory,
                          num_workers=num_workers)

    @abstractmethod
    def data(self, split: DatasetSplits = None):
        raise NotImplementedError
        # return self.x(split)

    @property
    def x(self) -> np.ndarray:
        raise AttributeError('The dataset does not have x property')

    @property
    def y(self) -> np.ndarray:
        raise AttributeError('The dataset does not have y property')


class DatasetView(object):
    def __init__(self,
                 dataset: AbstractDataset,
                 split: DatasetSplits):

        self._dataset = dataset
        self._subset = dataset.get_indexes(split)

    @property
    def dataset(self):
        return self.dataset

    @property
    def indexes(self):
        return self._subset

    def __len__(self) -> int:
        return len(self._subset)

    def __getitem__(self, item: Union[slice, int, list, np.ndarray]) -> \
            Tuple[Union[list, tuple, int, list], Union[np.ndarray, list]]:
        return self.dataset[self._subset[item]]


class UnsupervisedDataset(AbstractDataset):

    def __init__(self, x,
                 train: Union[list, np.ndarray],
                 test: [list, np.ndarray] = None,
                 dev: [list, np.ndarray] = None,
                 transformer: Callable = None,
                 test_transformer: Callable = None,
                 is_path_dataset: bool = False,
                 images_path: str = None,
                 **kwargs):

        """
        The init function, used to instantiate the class.
        :param x: The samples of the dataset.
        :param train: The list of the training set indexes.
        :param test: The list of the testing set indexes, if present.
        :param dev: The list of the dev set indexes, if present.
        :param transformer: A callable function f(x), which takes as input a sample and transforms it. In the case it is undefined, the identity function is used,
        :param is_path_dataset: If the dataset contains paths instead of images.
        :param images_path: the path from the root of the dataset, in which the images are stored.
        :param kwargs: Additional parameters.
        """
        assert len(x) == sum(map(len, [train, test, dev]))

        super().__init__(transformer=transformer,
                         test_transformer=test_transformer,
                         train=train,
                         test=test,
                         dev=dev,
                         is_path_dataset=is_path_dataset,
                         images_path = images_path,
                         **kwargs)

        self._x = x

    def __getitem__(self, item: Union[slice, int, list, np.ndarray]) -> \
            Tuple[Union[list, tuple, int, list], Union[np.ndarray, list]]:

        """
        Given an index, or a list of indexes (defined as list, tuple o slice), return the associated samples.
        :param item: The index, or more than one, used to fetch the samples in the dataset.
        :return: Return :param item: and the associated samples, modified by the transformer function.
        """
        to_map = False

        if not isinstance(item, (np.integer, int)):
            to_map = True
            if isinstance(item, slice):
                s = item.start if item.start is not None else 0
                e = item.stop
                step = item.step if item.step is not None else 1
                item = list(range(s, e, step))
            elif isinstance(item, tuple):
                item = list(item)

        idxs = self.current_indexes[item]

        if self.is_path_dataset:
            if to_map:
                x = [Image.open(self._x[i]).convert('RGB') for i in idxs]
            else:
                x = Image.open(self._x[idxs]).convert('RGB')
        else:
            x = self._x[idxs]

        if to_map:
            x = list(map(self._get_transformer(), x))
        else:
            x = self._get_transformer()(x)

        return item, x

    @property
    def x(self):
        return self._x[self.current_indexes]

    def data(self, split: DatasetSplits = None):
        return self.x


class SupervisedDataset(UnsupervisedDataset):
    """
    This class contains all the functions to operate with an _supervised dataset.
    It allows to use transformation (pytorch style) and to have all the dataset split (train, test, split) in one place.
    it also possible to split the dataset using custom percentages of the whole dataset.
    :param x: The samples of the dataset.
    :param y: The labels associated to x
    :param train: The list of the training set indexes.
    :param test: The list of the testing set indexes, if present.
    :param dev: The list of the dev set indexes, if present.
    :param transformer: A callable function f(x), which takes as input a sample and transforms it.
    In the case it is undefined, the identity function is used,
    :param kwargs: Additional parameters.
    """

    dataset_type = DatasetType.SUPERVISED

    def __init__(self,
                 x,
                 y,
                 train,
                 test=None,
                 dev=None,
                 transformer: Callable = None,
                 test_transformer: Callable = None,
                 target_transformer: Callable = None, **kwargs):

        """
        :param x: The samples of the dataset.
        :param y: The labels associated to x
        :param train: The list of the training set indexes.
        :param test: The list of the testing set indexes, if present.
        :param dev: The list of the dev set indexes, if present.
        :param transformer: A callable function f(x), which takes as input a sample and transforms it.
        In the case it is undefined, the identity function is used,
        :param kwargs: Additional parameters.
        """

        super().__init__(x=x,
                         train=train,
                         test=test,
                         dev=dev,
                         transformer=transformer,
                         test_transformer=test_transformer,
                         **kwargs)

        self._y = y
        assert len(self._x) == len(self._y)

        self.target_transformer = target_transformer \
            if target_transformer is not None else lambda z: z

        self._labels = tuple(sorted(list(set(y))))

    def __getitem__(self, item) -> Tuple[Union[list, tuple, int, list], np.ndarray, np.ndarray]:
        """
        Given an index, or a list of indexes (defined as list, tuple o slice), return the associated samples.
        :param item: The index, or more than one, used to fetch the samples in the dataset.
        :return: Return :param item: and the associated samples, x and y, modified by the transformer function.
        """

        to_map = False

        if not isinstance(item, (np.integer, int)):
            to_map = True

        item, x = super().__getitem__(item)

        if to_map:
            y = list(map(self.target_transformer,
                         self._y[self.current_indexes[item]]))
        else:
            y = self.target_transformer(self._y[self.current_indexes[item]])

        return item, x, y

    @property
    def labels(self) -> tuple:
        """
        :return: The set of labels of the dataset.
        """
        return self._labels

    @property
    def y(self):
        return self._y[self.current_indexes]

    def data(self, split: DatasetSplits = None):
        return self.x, self.y

    def preprocess_targets(self, f: Callable):
        """
        Apply the input function f to the current labels in the split.
        :param f: The callable function: f(x).
        """
        self._y = f(self._y)
