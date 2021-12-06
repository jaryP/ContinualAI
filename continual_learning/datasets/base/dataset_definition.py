import warnings
from abc import abstractmethod, ABC

from typing import Callable, Union, Tuple, List, Any, Sequence, Type, TypeVar

import numpy as np
from . import path_image_loading, DatasetSplits, DatasetType

IndexesType = Union[list, np.ndarray]
D = TypeVar('D', bound='AbstractDataset')


class AbstractDataset(ABC):

    @property
    @abstractmethod
    def classes(self, **kwargs):
        raise NotImplementedError

    @property
    @abstractmethod
    def values(self, **kwargs):
        raise NotImplementedError

    @property
    @abstractmethod
    def targets(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abstractmethod
    def get_subset(self, items, **kwargs):
        raise NotImplementedError


class BaseDataset(AbstractDataset):
    def __init__(self,
                 *,
                 values: IndexesType,
                 transform: Callable = None,
                 target_transform: Callable = None,
                 targets: IndexesType = None,
                 is_path_dataset: bool = False,
                 images_path: str = None,
                 path_loading_function: Callable = None,
                 **kwargs):

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

        self.images_path = images_path
        self.is_path_dataset = is_path_dataset
        self.path_loading_function = path_loading_function \
            if path_loading_function is not None else path_image_loading

        self.transform = transform \
            if transform is not None else lambda z: z

        self.target_transform = target_transform \
            if target_transform is not None else lambda z: z

        self._targets = targets
        self._values = values

    @property
    def classes(self):
        if self.targets is None:
            return None
        return sorted(list(set(self.targets)))

    @property
    def values(self):
        return self._values

    @property
    def targets(self):
        if self.dataset_type != DatasetType.SUPERVISED:
            return None
            # warnings.warn('The dataset is not supervised.',
            #               RuntimeWarning)
        return self._targets

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item: Union[slice, int, list, np.ndarray]) -> \
            Tuple[Sequence[int], Sequence[Any]]:

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
                x = self.path_loading_function(self.values[i])
            else:
                x = self.values[i]

            x = self.transform(x)

            if self.dataset_type == DatasetType.SUPERVISED:
                y = self.targets[i]
                x = self.target_transform(y)

                t = (i, x, y)
            else:
                t = (i, x)

            rets.append(t)

        return rets if to_map else rets[0]

    def get_subset(self,
                   subset: IndexesType,
                   **kwargs) -> D:

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
                 values: IndexesType,
                 subset: IndexesType,
                 transform: Callable = None,
                 target_transform: Callable = None,
                 targets: IndexesType = None,
                 is_path_dataset: bool = False,
                 images_path: str = None,
                 path_loading_function: Callable = None,
                 **kwargs):

        super().__init__(values=values, transform=transform,
                         target_transform=target_transform, targets=targets,
                         is_path_dataset=is_path_dataset,
                         images_path=images_path,
                         path_loading_function=path_loading_function, **kwargs)

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

        self._subset = subset

    @property
    def values(self):
        return [self._values[s] for s in self._subset]

    @property
    def targets(self):
        if self.dataset_type != DatasetType.SUPERVISED:
            return None
        #     warnings.warn('The dataset is not supervised.',
        #                   RuntimeWarning)

        return [self._targets[s] for s in self._subset]

    def __len__(self):
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

        return super().__getitem__(s)


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

        def _duplicate_check(l: Sequence):
            return len(set(l)) == len(l)

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

        if not isinstance(train, AbstractDataset):
            if _duplicate_check(train):
                raise ValueError('Train indexes list contains duplicates.')

            train = DatasetSubset(values=values,
                                  subset=train,
                                  transform=transform,
                                  target_transform=target_transform,
                                  targets=targets,
                                  is_path_dataset=is_path_dataset,
                                  images_path=images_path,
                                  path_loading_function=path_loading_function)

        if not isinstance(test, AbstractDataset):
            if _duplicate_check(test):
                raise ValueError('Test indexes list contains duplicates.')

            test = DatasetSubset(values=values,
                                 subset=test,
                                 transform=transform,
                                 target_transform=target_transform,
                                 targets=targets,
                                 is_path_dataset=is_path_dataset,
                                 images_path=images_path,
                                 path_loading_function=path_loading_function)

        if not isinstance(dev, AbstractDataset):
            if _duplicate_check(dev):
                raise ValueError('Dev indexes list contains duplicates.')

            dev = DatasetSubset(values=values,
                                subset=dev,
                                transform=transform,
                                target_transform=target_transform,
                                targets=targets,
                                is_path_dataset=is_path_dataset,
                                images_path=images_path,
                                path_loading_function=path_loading_function)

        self._splits = \
            {
                DatasetSplits.TRAIN: train,
                DatasetSplits.TEST: test,
                DatasetSplits.DEV: dev,
            }

        self._current_split = DatasetSplits.TRAIN

    def __len__(self):
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
    def current_dataset(self) :
        return self._splits[self.current_split]

    @property
    def classes(self):
        targets = self.current_dataset.targets
        if targets is None:
            return None
        return sorted(list(set(targets)))

    @property
    def values(self):
        return self.current_dataset.values

    @property
    def targets(self):
        return self.current_dataset.targets

    def train(self) -> None:
        self._current_split = DatasetSplits.TRAIN

    def train_split(self) -> D:
        return self.get_dataset(DatasetSplits.TRAIN)

    def test(self) -> None:
        self._current_split = DatasetSplits.TEST
        if len(self._splits[DatasetSplits.TEST]) == 0:
            warnings.warn('The dataset does not have Test split.',
                          RuntimeWarning)

    def test_split(self) -> D:
        return self.get_dataset(DatasetSplits.TEST)

    def dev(self) -> None:
        self._current_split = DatasetSplits.DEV
        if len(self._splits[DatasetSplits.DEV]) == 0:
            warnings.warn('The dataset does not have Development split.',
                          RuntimeWarning)

    def dev_split(self) -> D:
        return self.get_dataset(DatasetSplits.DEV)

    def get_subset(self,
                   train_subset: IndexesType,
                   test_subset: IndexesType = None,
                   dev_subset: IndexesType = None,
                   as_splitted_dataset: bool = False,
                   **kwargs) -> Union[D, Tuple[D, D, D]]:

        if train_subset is None:
            train_subset = []

        if test_subset is None:
            test_subset = []

        if dev_subset is None:
            dev_subset = []

        train = self.train_split().get_subset(train_subset)
        test = self.test_split().get_subset(test_subset)
        dev = self.dev_split().get_subset(dev_subset)

        if as_splitted_dataset:
            return DatasetSplitsContainer(train=train,
                                          test=test,
                                          dev=dev)
        else:
            return train, test, dev

    def get_dataset(self, split: Union[DatasetSplits, str],
                    **kwargs) -> BaseDataset:
        if isinstance(split, str):
            split = DatasetType(split)

        return self._splits[split]

    def __getitem__(self, item: Union[tuple, slice, int, list, np.ndarray]):
        return self.get_dataset(self.current_split)[item]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
