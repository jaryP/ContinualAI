import os
from enum import unique, Enum
from typing import Callable, Union, Tuple, List

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from .split_functions import extract_dev, split_dataset


@unique
class DatasetSplits(Enum):
    TRAIN = 0
    TEST = 1
    DEV = 2
    ALL = 3


class IndexesContainer(object):
    def __init__(self, *, train: Union[list, np.ndarray], test: [list, np.ndarray] = None,
                 dev: [list, np.ndarray] = None, **kwargs):

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

    def dev(self) -> None:
        self.current_split = DatasetSplits.DEV

    def all(self) -> None:
        self.current_split = DatasetSplits.ALL


class UnsupervisedDataset(IndexesContainer):
    # """
    # This class contains all the functions to operate with an unsupervised dataset
    # (a dataset which does not contains labels).
    # It allows to use transformation (pytorch style) and to have all the dataset split (train, test, split) in one place.
    # it also possible to split the dataset using custom percentages of the whole dataset.
    # :param x: The samples of the dataset.
    # :param train: The list of the training set indexes.
    # :param test: The list of the testing set indexes, if present.
    # :param dev: The list of the dev set indexes, if present.
    # :param transformer: A callable function f(x), which takes as input a sample and transforms it.
    # In the case it is undefined, the identity function is used,
    # :param kwargs: Additional parameters.
    # """

    def __init__(self, x, train: Union[list, np.ndarray], test: [list, np.ndarray] = None,
                 dev: [list, np.ndarray] = None,
                 transformer: Callable = None, test_transformer: Callable = None,
                 is_path_dataset: bool = False, images_path: str = '', **kwargs):

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

        super().__init__(train=train, test=test, dev=dev, **kwargs)

        self.is_path_dataset = is_path_dataset
        self.images_path = images_path

        self._x = x

        if transformer is not None:
            assert test_transformer is not None

        self._transformer = transformer if transformer is not None else lambda z: z
        self._test_transformer = test_transformer if test_transformer is not None else lambda z: z

        # self.transformer = transformer if transformer is not None else lambda z: z

    # def apply_transformer(self, v: bool = True):
    #     if v:
    #         self._transformer = self.transformer
    #     else:
    #         self._transformer = lambda x: x

    def _get_transformer(self, split: DatasetSplits = None):
        if split is None:
            split = self.current_split

        if split == DatasetSplits.TRAIN:
            return self._transformer
        else:
            return self._test_transformer

    def __getitem__(self, item: Union[slice, int, list, np.ndarray]) -> \
            Tuple[Union[list, tuple, int, list], Union[np.ndarray, list]]:

        """
        Given an index, or a list of indexes (defined as list, tuple o slice), return the associated samples.
        :param item: The index, or more than one, used to fetch the samples in the dataset.
        :return: Return :param item: and the associated samples, modified by the transformer function.
        """
        # We do not want to index if we need to use all the datasets
        # if self.current_split != DatasetSplits.ALL:
        #     f = lambda x, i: x[self.current_indexes[i]]
        # else:
        #     f = lambda x, i: x

        if self.is_path_dataset:
            # TODO: implement the case in which item is instance of list or slice
            img = Image.open(os.path.join(self.images_path, self._x[item]))
            img = img.convert('RGB')
            img = np.asarray(img)
            return item, self._transformer(img)
        else:
            if isinstance(item, (np.integer, int)):
                return item, self._get_transformer()(self._x[self.current_indexes[item]])

            if isinstance(item, slice):
                s = item.start if item.start is not None else 0
                e = item.stop
                step = item.step if item.step is not None else 1
                item = list(range(s, e, step))
            elif isinstance(item, tuple):
                item = list(item)

            a = list(map(self._get_transformer(), self._x[self.current_indexes[item]]))

            return item, a

    def get_iterator(self, batch_size, shuffle=True, sampler=None, num_workers=0, pin_memory=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          sampler=sampler, pin_memory=pin_memory, num_workers=num_workers)

    def x(self, split: DatasetSplits = None):
        if split is None:
            split = self.current_split

        return list(map(self._get_transformer(), self._x[self.get_indexes(split)]))

    def data(self, split: DatasetSplits = None):
        return self.x(split)

    # @property
    # def x(self) -> Union[np.ndarray, list]:
    #     """
    #     Return the samples of the current split.
    #     """
    #     return list(map(self._transformer, self._x[self.current_indexes]))
    #
    # @property
    # def data(self) -> Union[np.ndarray, list]:
    #     """
    #     Alias for self.x.
    #     """
    #     return self.x

    def preprocess(self, f: Callable) -> None:
        """
        Apply the input function f to the current split.
        :param f: The callable function: f(x).
        """
        self._x = f(self._x)

    def split_dataset(self, test_split: float = 0.2, dev_split: float = 0.0,
                      random_state: Union[np.random.RandomState, int] = None) -> None:
        """
        When called, split the dataset according to the values passed to the function.
        Modify the dataset in-place.
        :param test_split: The percentage of the data to be used in the test set. Must be 0 <= test_split <= 1
        :param dev_split: The percentage of the data to be used in the dev set. Must be 0 <= dev_split <= 1.
        :param random_state: The random state used to shuffle the dataset.
        If it isn't a numpy RandomState object, the object is retrieved by doing
        np.random.RandomState(:param random_state:).
        """
        assert test_split >= 0, 'The test_split must be 0 <= test_split <= 1. The current value is {}' \
            .format(test_split)
        assert dev_split >= 0, 'The dev_split must be 0 <= dev_split <= 1. The current value is {}'.format(dev_split)
        assert test_split + dev_split < 1, 'The sum of test_split and dev_split must be less than 1. ' \
                                           'The current value is {}'.format(dev_split + test_split)

        _train_split, _test_split, _dev_split = \
            split_dataset(self._x, test_split=test_split, dev_split=dev_split, random_state=random_state)

        super(UnsupervisedDataset, self)._splits = \
            {
                DatasetSplits.TRAIN: np.asarray(_train_split, dtype=int),
                DatasetSplits.TEST: np.asarray(_test_split, dtype=int),
                DatasetSplits.DEV: np.asarray(_dev_split, dtype=int),
            }

    def create_dev_split(self, dev_split: float = 0.1,
                         random_state: Union[np.random.RandomState, int] = None):
        """
        When called, extract the dev split from the training set
        Modify the dataset in-place.
        :param dev_split: The percentage of the data to be used in the dev set. Must be 0 <= dev_split <= 1.
        :param random_state: The random state used to shuffle the dataset.
        If it isn't a numpy RandomState object, the object is retrieved by doing
        np.random.RandomState(:param random_state:).
        """
        assert dev_split >= 0

        _train_split, _dev_split = extract_dev(y=self.get_indexes(DatasetSplits.TRAIN), dev_split=dev_split,
                                               random_state=random_state)

        self._splits = \
            {
                DatasetSplits.TRAIN: np.asarray(_train_split, dtype=int),
                DatasetSplits.TEST: np.asarray(self.get_indexes(DatasetSplits.TEST), dtype=int),
                DatasetSplits.DEV: np.asarray(_dev_split, dtype=int),
            }


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

    def __init__(self, x, y, train, test=None, dev=None, transformer: Callable = None,
                 test_transformer: Callable = None, target_transformer: Callable = None, **kwargs):
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
        super().__init__(x=x, train=train, test=test, dev=dev, transformer=transformer,
                         test_transformer=test_transformer, **kwargs)

        self._y = y
        assert len(self._x) == len(self._y)

        # self.target_transformer = target_transformer if target_transformer is not None else lambda z: z
        self._target_transformer = target_transformer if target_transformer is not None else lambda z: z

        self._labels = tuple(sorted(list(set(y))))

    def __getitem__(self, item) -> Tuple[Union[list, tuple, int, list], np.ndarray, np.ndarray]:
        """
        Given an index, or a list of indexes (defined as list, tuple o slice), return the associated samples.
        :param item: The index, or more than one, used to fetch the samples in the dataset.
        :return: Return :param item: and the associated samples, x and y, modified by the transformer function.
        """
        i, x = super().__getitem__(item)
        # We do not want to index if we need to use all the datasets
        # if self.current_split != DatasetSplits.ALL:
        #     f = lambda x: x[self.current_indexes[item]]
        # else:
        #     f = lambda x: x

        if isinstance(item, (np.integer, int)):
            # return item, self._transformer(self._x[self.current_indexes[item]])
            y = self._target_transformer(self._y[self.current_indexes[item]])
        else:
            if isinstance(item, slice):
                s = item.start if item.start is not None else 0
                e = item.stop
                step = item.step if item.step is not None else 1
                item = list(range(s, e, step))
            elif isinstance(item, tuple):
                item = list(item)

            y = list(map(self._target_transformer, self._y[self.current_indexes[item]]))

        return i, x, y

    @property
    def labels(self) -> tuple:
        """
        :return: The set of labels of the dataset.
        """
        return self._labels

    def y(self, split: DatasetSplits = None):
        if split is None:
            split = self.current_split

        return list(map(self._target_transformer, self._y[self.get_indexes(split)]))

    def data(self, split: DatasetSplits = None):
        return self.x(split), self.y(split)

    # @property
    # def y(self) -> Union[np.ndarray, list]:
    #     """
    #     :return: The labels of the current split.
    #     """
    #     return list(map(self._target_transformer, self._y[self.current_indexes]))
    #
    # @property
    # def data(self):
    #     """
    #     Alias for (self.x, self.y)
    #     :return:
    #     """
    #     return self.x, self.y

    # def apply_transformer(self, v: bool = True):
    #     super().apply_transformer(v)
    #     if v:
    #         self._target_transformer = self.target_transformer
    #     else:
    #         self._target_transformer = lambda x: x

    def preprocess_targets(self, f: Callable):
        """
        Apply the input function f to the current labels in the split.
        :param f: The callable function: f(x).
        """
        self._y = f(self._y)

    def split_dataset(self, test_split: float = 0.2, dev_split: float = 0.0, balanced_split: bool = True,
                      random_state: Union[np.random.RandomState, int] = None):
        """
        When called, split the dataset according to the values passed to the function.
        Modify the dataset in-place.
        :param test_split: The percentage of the data to be used in the test set. Must be 0 <= test_split <= 1
        :param dev_split: The percentage of the data to be used in the dev set. Must be 0 <= dev_split <= 1.
        :param balanced_split: If the resulting splits need to have balanced number of samples for each label
        :param random_state: The random state used to shuffle the dataset.
        If it isn't a numpy RandomState object, the object is retrieved by doing
        np.random.RandomState(:param random_state:).
        """
        assert test_split >= 0
        assert dev_split >= 0
        assert test_split + dev_split < 1

        _train_split, _test_split, _dev_split = \
            split_dataset(self._y, balance_labels=balanced_split,
                          test_split=test_split, dev_split=dev_split, random_state=random_state)

        self._splits = \
            {
                DatasetSplits.TRAIN: np.asarray(_train_split, dtype=int),
                DatasetSplits.TEST: np.asarray(_test_split, dtype=int),
                DatasetSplits.DEV: np.asarray(_dev_split, dtype=int),
            }
