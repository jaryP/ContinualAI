import warnings

from abc import ABC, abstractmethod
from typing import Callable, Union, Tuple, List, Any, Sequence

import numpy as np
from PIL import Image

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
    def current_split(self, v: Union[DatasetSplits, int, str]) -> None:
        if isinstance(v, (str, int)):
            v = DatasetSplits(v)

        if v == DatasetSplits.TRAIN:
            self.train()
        elif v == DatasetSplits.TEST:
            self.test()
        elif v == DatasetSplits.DEV:
            self.dev()
        else:
            self.all()

    def get_split_len(self, v: DatasetSplits = None) -> int:
        if v == DatasetSplits.ALL:
            return sum(map(len, [v for v in self._splits.values()]))
        if v is None:
            return self.current_split

        return len(self._splits[v])

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
        self._current_split = DatasetSplits.TRAIN

    def test(self) -> None:
        self._current_split = DatasetSplits.TEST
        if len(self._splits[DatasetSplits.TEST]) == 0:
            warnings.warn('The dataset does not have Test split.',
                          RuntimeWarning)

    def dev(self) -> None:
        self._current_split = DatasetSplits.DEV
        if len(self._splits[DatasetSplits.DEV]) == 0:
            warnings.warn('The dataset does not have Development split.',
                          RuntimeWarning)

    def all(self) -> None:
        self._current_split = DatasetSplits.ALL


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
    def get_subset(self,
                   train_subset: IndexesType,
                   test_subset: IndexesType,
                   dev_subset: IndexesType,
                   **kwargs):
        raise NotImplementedError
        # return self.x(split)

    @abstractmethod
    def __getitem__(self, item: Union[tuple, slice, int, list, np.ndarray]) -> \
            Tuple[Union[list, tuple, int, list], Union[np.ndarray, list]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def data(self, split: DatasetSplits = None):
        raise NotImplementedError
        # return self.x(split)

    @property
    @abstractmethod
    def x(self) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def y(self) -> np.ndarray:
        raise NotImplementedError

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


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
        # assert len(x) == sum(map(len, [train, test, dev]))

        super().__init__(transformer=transformer,
                         test_transformer=test_transformer,
                         train=train,
                         test=test,
                         dev=dev,
                         is_path_dataset=is_path_dataset,
                         images_path=images_path,
                         **kwargs)

        self._x = x

    def __getitem__(self, item: Union[slice, int, list, np.ndarray]) -> \
            Tuple[Sequence[int], Sequence[Any]]:

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

    @property
    def data(self, split: DatasetSplits = None):
        return self.x

    def get_subset(self,
                   train_subset: IndexesType,
                   test_subset: IndexesType = None,
                   dev_subset: IndexesType = None,
                   **kwargs):

        _train, _test, _dev = None, None, None

        if len(train_subset) > 0:
            _train = self.get_indexes(DatasetSplits.TRAIN)[train_subset]

        if test_subset is not None and len(test_subset) > 0:
            _test = self.get_indexes(DatasetSplits.TEST)[test_subset]

        if dev_subset is not None and len(dev_subset) > 0:
            _dev = self.get_indexes(DatasetSplits.DEV)[dev_subset]

        return UnsupervisedDataset(x=self._x,
                                   train=_train,
                                   test=_test,
                                   dev=_dev,
                                   is_path_dataset=self.is_path_dataset,
                                   images_path=self.images_path,
                                   transformer=self.transformer,
                                   test_transformer=self.test_transformer)


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

        labels = set()

        for s in self._splits.values():
            sy = y[s]
            labels.update(list(sy))

        self._labels = tuple(sorted(list(labels)))

    def __getitem__(self, item: Union[slice, int, list, np.ndarray]) -> \
            Tuple[Sequence[int], Sequence[Any], Sequence[Any]]:

        """
        Given an index, or a list of indexes (defined as list, tuple o slice), return the associated samples.
        :param item: The index, or more than one, used to fetch the samples in the dataset.
        :return: Return :param item: and the associated samples, x and y, modified by the transformer function.
        """

        to_map = False

        if not isinstance(item, (np.integer, int)):
            to_map = True

        item, x = super().__getitem__(item)

        idx = self.current_indexes[item]

        if to_map:
            y = list(map(self.target_transformer,
                         self._y[idx]))
        else:
            y = self.target_transformer(self._y[idx])

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

    @property
    def data(self, split: DatasetSplits = None):
        return self.x, self.y

    def get_subset(self,
                   train_subset: IndexesType,
                   test_subset: IndexesType = None,
                   dev_subset: IndexesType = None,
                   **kwargs):

        _train, _test, _dev = None, None, None
        # all_indexes = []

        if len(train_subset) > 0:
            _train = self.get_indexes(DatasetSplits.TRAIN)[train_subset]
            # all_indexes.extend(_train)

        if test_subset is not None and len(test_subset) > 0:
            _test = self.get_indexes(DatasetSplits.TEST)[test_subset]
            # all_indexes.extend(_test)

        if dev_subset is not None and len(dev_subset) > 0:
            _dev = self.get_indexes(DatasetSplits.DEV)[dev_subset]
            # all_indexes.extend(_dev)

        # _train, _test, _dev = np.asarray(_train),\
        #                       np.asarray(_test), \
        #                       np.asarray(_dev)
        #
        # all_indexes = np.concatenate((_train, _test, _dev), 1)

        # x = self._x[all_indexes]
        # y = self._y[all_indexes]
        # input(x.base is self._x)
        # input(y.base is self._y)

        return SupervisedDataset(x=self._x,
                                 y=self._y,
                                 train=_train,
                                 test=_test,
                                 dev=_dev,
                                 is_path_dataset=self.is_path_dataset,
                                 images_path=self.images_path,
                                 transformer=self.transformer,
                                 target_transformer=self.target_transformer,
                                 test_transformer=self.test_transformer)
