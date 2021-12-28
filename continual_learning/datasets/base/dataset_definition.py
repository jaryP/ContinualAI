import warnings
from abc import abstractmethod, ABC
from os import makedirs
from os.path import exists, join, dirname

from typing import Callable, Union, Tuple, List, Any, Sequence, Type, TypeVar

import numpy as np
from . import path_image_loading, DatasetSplits, DatasetType

IndexesType = Union[list, np.ndarray]

D = TypeVar('D', bound='AbstractDataset')

Va_T = TypeVar('Va_T')
Ta_T = TypeVar('Ta_T')

Tva_T = TypeVar('Tva_T')

C_R_T = Union[Sequence[int], None]
Ta_R_T = Union[Sequence[Ta_T], None]
Va_R_T = Union[Sequence[Va_T], None]


class AbstractDataset(ABC):

    def __init__(self):
        self.dataset_type = None

    @property
    @abstractmethod
    def classes(self) -> C_R_T:
        raise NotImplementedError

    @property
    @abstractmethod
    def values(self) -> Va_R_T:
        raise NotImplementedError

    @property
    @abstractmethod
    def targets(self) -> Ta_R_T:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item) -> Sequence[Union[int, Va_T, Ta_T]]:
        raise NotImplementedError

    @abstractmethod
    def get_subset(self, items, **kwargs) -> D:
        raise NotImplementedError


class BaseDataset(AbstractDataset):
    def __init__(self,
                 *,
                 values: Sequence[Va_T],
                 transform: Callable = None,
                 target_transform: Callable = None,
                 targets: Sequence[Ta_T] = None,
                 is_path_dataset: bool = False,
                 images_path: str = None,
                 path_loading_function: Callable = None,
                 **kwargs):

        super(BaseDataset, self).__init__()

        if is_path_dataset and images_path is None:
            raise ValueError('If is_path_dataset=True, '
                             'then images_path must be not None')

        if targets is None:
            self.dataset_type = DatasetType.UNSUPERVISED
        else:
            if len(targets) != len(values):
                raise ValueError(f'The len of values ({len(values)}) '
                                 f'and targets ({len(targets)}) '
                                 f'are different.')

            self.dataset_type = DatasetType.SUPERVISED

        self.images_path: str = images_path
        self.is_path_dataset: bool = is_path_dataset
        self.path_loading_function: Callable[[str], Va_T] = \
            path_loading_function \
                if path_loading_function is not None else path_image_loading

        self.transform: Callable[[Va_T], Any] = transform \
            if transform is not None else lambda z: z

        self.target_transform: Callable[[Ta_T], Any] = target_transform \
            if target_transform is not None else lambda z: z

        self._use_transform: bool = True

        self._targets: Ta_T = targets
        self._values: Va_T = values

    @property
    def base_dataset_indexes(self) -> Sequence[int]:
        return np.arange(len(self))

    @property
    def classes(self) -> C_R_T:
        if self.targets is None:
            return None
        return sorted(list(set(self.targets)))

    @property
    def values(self) -> Va_R_T:
        return self._values

    @property
    def targets(self) -> Ta_R_T:
        if self.dataset_type != DatasetType.SUPERVISED:
            return None
        return self._targets

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, item: Union[slice, int, list, np.ndarray]) -> \
            Tuple[Sequence[int], Union[Tuple[Sequence[Any], Sequence[Any]],
                                       Sequence[Any]]]:

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
        else:
            item = [item]

        rets = []
        for i in item:
            if self.is_path_dataset:
                x = self.path_loading_function(self._values[i])
            else:
                x = self._values[i]

            if self._use_transform:
                x = self.transform(x)

            if self.dataset_type == DatasetType.SUPERVISED:
                y = self._targets[i]

                if self._use_transform:
                    y = self.target_transform(y)

                t = (i, x, y)
            else:
                t = (i, x)

            rets.append(t)

        return rets if to_map else rets[0]

    def use_transform(self, v: bool) -> None:
        self._use_transform = v

    def get_subset(self,
                   subset: IndexesType,
                   **kwargs) -> 'DatasetSubset':

        if subset is None or len(subset) == 0:
            raise ValueError('The parameter subset is empty or None.')

        return DatasetSubset(values=self._values,
                             subset=subset,
                             transform=self.transform,
                             target_transform=self.target_transform,
                             targets=self._targets,
                             is_path_dataset=self.is_path_dataset,
                             images_path=self.images_path,
                             path_loading_function=self.path_loading_function)


class DatasetSubset(BaseDataset):
    def __init__(self, *,
                 values: Sequence[Va_T],

                 subset: IndexesType,
                 transform: Callable = None,
                 target_transform: Callable = None,

                 targets: Sequence[Ta_T] = None,
                 is_path_dataset: bool = False,
                 images_path: str = None,
                 path_loading_function: Callable = None,
                 **kwargs):

        if len(subset) >= len(values):
            raise ValueError(f'The len of values ({len(values)}) '
                             f'is higher than length of'
                             f'f subset ({len(subset)})')

        if len(subset) > 0:
            if max(subset) > len(values) or min(subset) < 0:
                raise ValueError(f'The parameter subset contains a value that '
                                 f'is greater ({max(subset)}) than '
                                 f'the length of values ({len(values) - 1}), or '
                                 f'the minimum value is '
                                 f'lower that 0 ({min(subset)})')

        self._subset: Sequence[int] = np.asarray(subset)

        super().__init__(values=values, transform=transform,
                         target_transform=target_transform, targets=targets,
                         is_path_dataset=is_path_dataset,
                         images_path=images_path,
                         path_loading_function=path_loading_function, **kwargs)

    @property
    def base_dataset_indexes(self) -> Sequence[int]:
        return self._subset

    @property
    def classes(self) -> C_R_T:
        if self.targets is None:
            return None
        return sorted(list(set(self.targets)))

    @property
    def values(self) -> Va_R_T:
        return [self._values[s] for s in self._subset]

    @property
    def targets(self) -> Ta_R_T:
        if self.dataset_type != DatasetType.SUPERVISED:
            return None
        #     warnings.warn('The dataset is not supervised.',
        #                   RuntimeWarning)
        return [self._targets[s] for s in self._subset]

    def __len__(self) -> int:
        return len(self._subset)

    def __getitem__(self, item: Union[slice, int, list, np.ndarray]) -> \
            Tuple[Sequence[int], Sequence[Any]]:

        if not isinstance(item, (np.integer, int)):
            if isinstance(item, slice):
                s = item.start if item.start is not None else 0
                e = item.stop
                step = item.step if item.step is not None else 1
                item = list(range(s, e, step))
            elif isinstance(item, tuple):
                item = list(item)
            s = [self._subset[i] for i in item]
        else:
            s = self._subset[item]

        if self.dataset_type == DatasetType.SUPERVISED:
            _, x, y = super().__getitem__(s)
            return item, x, y
        else:
            _, x = super().__getitem__(s)
            return item, x


class DatasetSplitsContainer(AbstractDataset):
    def __init__(self, *,
                 train: Union[IndexesType, D],
                 values: IndexesType = None,
                 transform: Callable = None,
                 base_dataset: Union[DatasetSubset, D] = None,
                 target_transform: Callable = None,
                 test_transform: Callable = None,
                 targets: IndexesType = None,
                 test: Union[IndexesType, D] = None,
                 dev: Union[IndexesType, D] = None,
                 is_path_dataset: bool = False,
                 images_path: str = None,
                 path_loading_function: Callable = None,
                 **kwargs):

        super().__init__()

        def _duplicate_check(l: Sequence):
            return len(set(l)) != len(l)

        if transform is not None and test_transform is None:
            raise ValueError('Train transform provided '
                             'but test_transform is None. '
                             'Please proved both or none. ')

        if (values is None and base_dataset is None) and \
                any([not isinstance(s, AbstractDataset)
                     for s in [train, test, dev]]):
            raise ValueError('You passed indexes as input but'
                             ' value and base_dataset parameter are None. '
                             'Please proved one of them.')

        if base_dataset is not None and \
                (values is not None or targets is not None):
            raise ValueError('Base dataset is not None but one, or both, '
                             'values and targets are note None. '
                             'PLease set both to none to use the base dataset.')

        if base_dataset is not None:
            targets = base_dataset.targets
            values = base_dataset.values

        if dev is None:
            dev = []
        if test is None:
            test = []

        if not isinstance(train, (BaseDataset, DatasetSubset)):
            if _duplicate_check(train):
                raise ValueError('Train indexes list contains duplicates.')

            d_train = DatasetSubset(values=values,
                                    subset=train,
                                    transform=transform,
                                    target_transform=target_transform,
                                    targets=targets,
                                    is_path_dataset=is_path_dataset,
                                    images_path=images_path,
                                    path_loading_function=path_loading_function)
        else:
            d_train = train

        if not isinstance(test, (BaseDataset, DatasetSubset)):
            if _duplicate_check(test):
                raise ValueError('Test indexes list contains duplicates.')

            d_test = DatasetSubset(values=values,
                                   subset=test,
                                   transform=transform,
                                   target_transform=target_transform,
                                   targets=targets,
                                   is_path_dataset=is_path_dataset,
                                   images_path=images_path,
                                   path_loading_function=path_loading_function)
        else:
            d_test = test

        if not isinstance(dev, (BaseDataset, DatasetSubset)):
            if _duplicate_check(dev):
                raise ValueError('Dev indexes list contains duplicates.')

            d_dev = DatasetSubset(values=values,
                                  subset=dev,
                                  transform=transform,
                                  target_transform=target_transform,
                                  targets=targets,
                                  is_path_dataset=is_path_dataset,
                                  images_path=images_path,
                                  path_loading_function=path_loading_function)
        else:
            d_dev = dev

        self._splits = \
            {
                DatasetSplits.TRAIN: d_train,
                DatasetSplits.TEST: d_test,
                DatasetSplits.DEV: d_dev,
            }

        self._current_split: DatasetSplits = DatasetSplits.TRAIN

    def __len__(self) -> int:
        return len(self._splits[self._current_split])

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

    @property
    def current_dataset(self) -> AbstractDataset:
        return self._splits[self.current_split]

    @property
    def classes(self) -> C_R_T:
        targets = self.current_dataset.targets
        if targets is None:
            return None
        return sorted(list(set(targets)))

    @property
    def values(self) -> Va_R_T:
        return self.current_dataset.values

    @property
    def targets(self) -> Ta_R_T:
        return self.current_dataset.targets

    @property
    def base_dataset_indexes(self) -> Sequence[int]:
        return self.current_split.base_dataset_indexes

    def get_dataset_indexes(self, v: DatasetSplits) -> Sequence[int]:
        return self.get_split(v).base_dataset_indexes

    def train(self) -> None:
        self._current_split = DatasetSplits.TRAIN

    def train_split(self) -> D:
        return self.get_split(DatasetSplits.TRAIN)

    def test(self) -> None:
        self._current_split = DatasetSplits.TEST
        if len(self._splits[DatasetSplits.TEST]) == 0:
            warnings.warn('The dataset does not have Test split.',
                          RuntimeWarning)

    def test_split(self) -> D:
        return self.get_split(DatasetSplits.TEST)

    def dev(self) -> None:
        self._current_split = DatasetSplits.DEV
        if len(self._splits[DatasetSplits.DEV]) == 0:
            warnings.warn('The dataset does not have Development split.',
                          RuntimeWarning)

    def dev_split(self) -> D:
        return self.get_split(DatasetSplits.DEV)

    def get_subset(self,
                   train_subset: IndexesType,
                   test_subset: IndexesType = None,
                   dev_subset: IndexesType = None,
                   as_splitted_dataset: bool = False,
                   **kwargs) -> Union['DatasetSplitsContainer',
                                      Tuple[D, D, D]]:

        if train_subset is None:
            train_subset = []

        if test_subset is None:
            test_subset = []

        if dev_subset is None:
            dev_subset = []

        if len(train_subset) == 0 \
                and len(test_subset) == 0 \
                and len(dev_subset) == 0:
            raise ValueError('One of train_subset, test_subset and dev_subset '
                             'must be not None (or non empty sequence).')

        train = self.train_split()
        train = train.get_subset(train_subset)

        test = self.test_split()
        if len(test) > 0:
            test = test.get_subset(test_subset)

        dev = self.dev_split()
        if len(dev) > 0:
            dev = dev.get_subset(dev_subset)

        if as_splitted_dataset:
            return DatasetSplitsContainer(train=train,
                                          test=test,
                                          dev=dev)
        else:
            return train, test, dev

    def use_transform(self, v: bool) -> None:
        self.current_split.use_transform(v)

    def get_split(self, split: Union[DatasetSplits, str]) -> D:
        if isinstance(split, str):
            split = DatasetSplits(split)

        return self._splits[split]

    def __getitem__(self, item: Union[tuple, slice, int, list, np.ndarray]):
        a = self.current_dataset
        return a[item]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class DownloadableDataset(DatasetSplitsContainer, ABC):
    # TODO: Types Hints
    def __init__(self,
                 *,
                 name: str,

                 transform: Callable = None,
                 test_transform: Callable = None,
                 target_transform: Callable = None,

                 download_if_missing: bool = True,
                 data_folder: str = None,

                 is_path_dataset: bool = False,
                 images_path: str = None,
                 path_loading_function: Callable = None,

                 **kwargs):

        if data_folder is None:
            data_folder = join(dirname(__file__), 'downloaded_datasets', name)

        self.data_folder = data_folder
        self._name = name

        missing = not self._check_exists()

        if missing:
            if not download_if_missing:
                raise IOError("Data not found and "
                              "`download_if_missing` is False")
            else:
                if not exists(self.data_folder):
                    makedirs(self.data_folder)

                print('Downloading dataset {}'.format(self.name))
                self.download_dataset()

        values, (train, test, dev) = self.load_dataset()

        if isinstance(values, tuple):
            x, y = values
        else:
            x = values
            y = None

        super().__init__(values=x,
                         targets=y,
                         train=train,
                         test=test,
                         dev=dev,
                         transform=transform,
                         target_transform=
                         target_transform,
                         test_transform=
                         test_transform,
                         is_path_dataset=is_path_dataset,
                         images_path=images_path,
                         path_loading_function=path_loading_function,
                         **kwargs)

    @property
    def name(self):
        return self._name

    @abstractmethod
    def download_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def _check_exists(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load_dataset(self) -> Tuple[Union[np.ndarray,
                                          Tuple[np.ndarray, np.ndarray]],
                                    Tuple[list, list, list]]:
        raise NotImplementedError
