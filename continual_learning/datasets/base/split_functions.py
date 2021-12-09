# __all__ = ['split_dataset', 'extract_dev']

from collections import defaultdict
from copy import deepcopy

from typing import Union, Tuple

import numpy as np
#
# from continual_learning.datasets.base import AbstractDataset, \
#     DatasetSubset, \
#     BaseDataset, \
#     DatasetSplitsContainer, SupervisedDataset, UnsupervisedDataset, \
#     DatasetSplits


# SupervisedDataset, \
# UnsupervisedDataset, \
# DatasetSplits, \
# DatasetSplitContexView, \
from continual_learning.datasets.base import DatasetSplitsContainer, \
    AbstractDataset, BaseDataset, DatasetSubset


# def _get_balanced_index(y: Union[list, np.ndarray],
#                         test_split: float,
#                         dev_split: float = 0,
#                         random_state: np.random.RandomState = None) \
#         -> Tuple[list, list, list]:
#     train = []
#     test = []
#     dev = []
#
#     d = defaultdict(list)
#
#     for i, _y in enumerate(y):
#         d[_y].append(i)
#
#     for index_list in d.values():
#
#         if random_state is not None:
#             random_state.shuffle(index_list)
#         else:
#             np.random.shuffle(index_list)
#
#         ln = len(index_list)
#
#         _test_split = int(ln * test_split)
#         _dev_split = int(ln * dev_split)
#
#         train.extend(index_list[_test_split + _dev_split:])
#         test.extend(index_list[:_test_split])
#         dev.extend(index_list[_test_split:_test_split + _dev_split])
#
#     return train, test, dev
#
#
# def _get_split_index(y: Union[list, np.ndarray],
#                      test_split: float,
#                      dev_split: float = 0,
#                      random_state: np.random.RandomState = None) \
#         -> Tuple[list, list, list]:
#     index_list = np.arange(len(y))
#
#     if random_state is not None:
#         random_state.shuffle(index_list)
#     else:
#         np.random.shuffle(index_list)
#
#     ln = len(index_list)
#
#     _test_split = int(ln * test_split)
#     _dev_split = int(ln * dev_split)
#
#     train = index_list[_test_split + _dev_split:]
#     test = index_list[:_test_split]
#     dev = index_list[_test_split:_test_split + _dev_split]
#
#     return train, test, dev
#
#
# def extract_dev(y: Union[list, np.ndarray],
#                 dev_split: float = 0.1,
#                 random_state: Union[np.random.RandomState, int] = None):
#     if not isinstance(random_state, np.random.RandomState):
#         random_state = np.random.RandomState(random_state)
#
#     y = np.asarray(y)
#
#     index_list = np.arange(len(y))
#
#     random_state.shuffle(index_list)
#
#     _dev_split = int(len(index_list) * (1 - dev_split))
#
#     train = y[index_list[:_dev_split]]
#     dev = y[index_list[_dev_split:]]
#
#     return train, dev
#
#
# def split_dataset(y: Union[list, np.ndarray],
#                   test_split: float,
#                   dev_split: float = 0,
#                   balance_labels: bool = True,
#                   random_state: Union[np.random.RandomState, int] = None) -> \
#         Tuple[list, list, list]:
#     if isinstance(random_state, int):
#         random_state = np.random.RandomState(random_state)
#
#     if balance_labels:
#         train, test, dev = _get_balanced_index(y, test_split, dev_split,
#                                                random_state)
#     else:
#         train, test, dev = _get_split_index(y, test_split, dev_split,
#                                             random_state)
#
#     assert sum(map(len, [train, test, dev])) == len(y)
#
#     return train, test, dev
#
#
# def create_dataset_with_dev_split(dataset: Union[SupervisedDataset,
#                                                  UnsupervisedDataset],
#                                   dev_percentage: float = 0.1,
#                                   balance_labels: bool = False,
#                                   random_state: Union[np.random.RandomState,
#                                                       int] = None):
#     if isinstance(random_state, int):
#         random_state = np.random.RandomState(random_state)
#     else:
#         random_state = np.random.RandomState(None)
#
#     is_supervised = False
#
#     train_split = dataset.get_indexes(DatasetSplits.TRAIN)
#     test_split = dataset.get_indexes(DatasetSplits.TEST)
#     dev_split = dataset.get_indexes(DatasetSplits.DEV)
#
#     if isinstance(dataset, SupervisedDataset):
#         is_supervised = True
#         c = SupervisedDataset
#         y = dataset.y
#     else:
#         c = UnsupervisedDataset
#         y = dataset.x
#
#     indexes = deepcopy(np.concatenate((train_split, dev_split)))
#     index_list = np.arange(len(indexes))
#
#     random_state.shuffle(index_list)
#
#     _dev_split = int(len(index_list) * (1 - dev_percentage))
#
#     train = indexes[index_list[:_dev_split]]
#     dev = indexes[index_list[_dev_split:]]
#
#     new_dataset = c(x=dataset._x,
#                     y=dataset._y if is_supervised else None,
#                     train=train,
#                     test=test_split,
#                     dev=dev,
#                     transformer=dataset.transformer,
#                     test_transformer=dataset.
#                     test_transformer,
#                     target_transformer=dataset.
#                     test_transformer,
#                     is_path_dataset=dataset.is_path_dataset,
#                     images_path=dataset.images_path)
#
#     return new_dataset


def _base_dataset_split(dataset: Union[DatasetSubset, BaseDataset],
                        random_state: np.random.RandomState,
                        test_percentage: float = 0.2,
                        dev_percentage: float = 0) -> Tuple[DatasetSubset,
                                                            DatasetSubset,
                                                            DatasetSubset]:
    ln = len(dataset)

    index_list = np.arange(ln)

    random_state.shuffle(index_list)

    _dev_split = int(len(index_list) * dev_percentage)
    _test_split = int(len(index_list) * test_percentage)

    dev = index_list[: _dev_split]
    test = index_list[_dev_split: _dev_split + _test_split]
    train = index_list[_dev_split + _test_split:]

    train_subset = dataset.get_subset(train)
    test_subset = dataset.get_subset(test)
    dev_subset = dataset.get_subset(dev)

    return train_subset, test_subset, dev_subset


def create_dataset_with_new_split(dataset: AbstractDataset,
                                  test_percentage: float = 0.2,
                                  dev_percentage: float = 0,
                                  balance_labels: bool = False,
                                  random_state: Union[np.random.RandomState,
                                                      int] = None,
                                  as_dataset_container: bool = False):

    if 0 >= test_percentage > 1:
        raise ValueError(f'test_percentage must be in [0, 1) '
                         f'({test_percentage})')

    if 0 >= dev_percentage > 1:
        raise ValueError(f'dev_percentage must be in [0, 1) '
                         f'({dev_percentage})')

    if test_percentage + dev_percentage >= 1:
        raise ValueError(f'resulting value of '
                         f'dev_percentage plus test_percentage '
                         f'must be in [0, 1) '
                         f'({test_percentage + dev_percentage})')

    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        random_state = np.random.RandomState(None)

    if isinstance(dataset, (BaseDataset, DatasetSubset)):
        train, test, dev = _base_dataset_split(dataset=dataset,
                                               dev_percentage=dev_percentage,
                                               test_percentage=test_percentage,
                                               random_state=random_state)
    elif isinstance(dataset, DatasetSplitsContainer):
        train, test, dev = \
            add_dev_split_to_container(dataset=dataset,
                                       dev_percentage=dev_percentage,
                                       as_container=False,
                                       random_state=random_state)
    else:
        assert False

    if as_dataset_container:
        return DatasetSplitsContainer(train=train, test=test, dev=dev)
    else:
        return train, test, dev


def add_dev_split_to_container(dataset: DatasetSplitsContainer,
                               dev_percentage: float = 0.1,
                               from_test: bool = False,
                               random_state: Union[np.random.RandomState,
                                                   int] = None,
                               as_container: bool = True,
                               ) -> Union[DatasetSplitsContainer,
                                          Tuple[AbstractDataset,
                                                AbstractDataset,
                                                AbstractDataset]]:
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        random_state = np.random.RandomState(None)

    if 0 >= dev_percentage > 1:
        raise ValueError(f'dev_percentage must be in [0, 1) '
                         f'({dev_percentage})')

    dev = dataset.dev_split()
    if len(dev) > 0:
        raise ValueError(f'The dataset already has a dev split')

    if from_test and len(dataset.test_split()) == 0:
        raise ValueError(f'The parameter from_test is set to True but'
                         f' the dataset hasn\'t  a dev split.')

    if from_test:
        source = dataset.test_split()
    else:
        source = dataset.train_split()

    index_list = np.arange(len(source))
    random_state.shuffle(index_list)

    _dev_split = int(len(index_list) * dev_percentage)

    dev_idx = index_list[: _dev_split]
    source_idx = index_list[_dev_split:]

    dev = source.get_subset(dev_idx)
    source_subset = source.get_subset(source_idx)

    if from_test:
        train = dataset.train_split()
        test = source_subset
    else:
        train = source_subset
        test = dataset.test_split()

    if as_container:
        return DatasetSplitsContainer(train=train, test=test, dev=dev)
    else:
        return train, test, dev


# def _create_dataset_with_new_split(dataset: AbstractDataset,
#                                    test_percentage: float = 0.2,
#                                    dev_percentage: float = 0,
#                                    balance_labels: bool = False,
#                                    random_state: Union[np.random.RandomState,
#                                                        int] = None):
#     if 0 >= test_percentage > 1:
#         raise ValueError(f'test_percentage must be in [0, 1) '
#                          f'({test_percentage})')
#
#     if 0 >= dev_percentage > 1:
#         raise ValueError(f'dev_percentage must be in [0, 1) '
#                          f'({dev_percentage})')
#
#     if test_percentage + dev_percentage >= 1:
#         raise ValueError(f'resulting value of '
#                          f'dev_percentage plus test_percentage '
#                          f'must be in [0, 1) '
#                          f'({test_percentage + dev_percentage})')
#
#     if isinstance(random_state, int):
#         random_state = np.random.RandomState(random_state)
#     else:
#         random_state = np.random.RandomState(None)
#
#     is_supervised = False
#
#     train_split = dataset.get_indexes(DatasetSplits.TRAIN)
#     test_split = dataset.get_indexes(DatasetSplits.TEST)
#     dev_split = dataset.get_indexes(DatasetSplits.DEV)
#
#     if isinstance(dataset, SupervisedDataset):
#         is_supervised = True
#         c = SupervisedDataset
#         # y = dataset.y
#     else:
#         c = UnsupervisedDataset
#         # y = dataset.x
#
#     indexes = deepcopy(np.concatenate((train_split, dev_split, test_split)))
#     index_list = np.arange(len(indexes))
#
#     random_state.shuffle(index_list)
#
#     _dev_split = int(len(index_list) * dev_percentage)
#     _test_split = int(len(index_list) * test_percentage)
#
#     dev = indexes[index_list[: _dev_split]]
#     test = indexes[index_list[_dev_split: _dev_split + _test_split]]
#     train = indexes[index_list[_dev_split + _test_split:]]
#
#     new_dataset = c(x=dataset._x,
#                     y=dataset._y if is_supervised else None,
#                     train=train,
#                     test=test,
#                     dev=dev,
#                     transformer=dataset.transformer,
#                     test_transformer=dataset.
#                     test_transformer,
#                     target_transformer=dataset.
#                     test_transformer,
#                     is_path_dataset=dataset.is_path_dataset,
#                     images_path=dataset.images_path)
#     # else:
#     #     new_dataset = UnsupervisedDataset(x=dataset.x,
#     #                                       y=dataset.y,
#     #                                       train=train,
#     #                                       test=test_split,
#     #                                       dev=dev,
#     #                                       transformer=dataset.transformer,
#     #                                       test_transformer=dataset.
#     #                                       test_transformer,
#     #                                       target_transformer=dataset.
#     #                                       test_transformer,
#     #                                       is_path_dataset=dataset.
#     #                                       is_path_dataset,
#     #                                       images_path=dataset.images_path)
#     #
#     # if isinstance(dataset, SupervisedDataset):
#     #
#     #     y = dataset.y
#     #
#     #     index_list = np.arange(len(y))
#     #
#     #     random_state.shuffle(index_list)
#     #
#     #     _dev_split = int(len(index_list) * (1 - dev_split))
#     #
#     #     train = y[index_list[:_dev_split]]
#     #     dev = y[index_list[_dev_split:]]
#     #
#     #     new_dataset = SupervisedDataset(x=dataset.x,
#     #                                     y=dataset.y,
#     #                                     train=train,
#     #                                     test=dataset.get_indexes(
#     #                                         DatasetSplits.TEST),
#     #                                     dev=dev,
#     #                                     transformer=dataset.transformer,
#     #                                     test_transformer=dataset.
#     #                                     test_transformer,
#     #                                     target_transformer=dataset.
#     #                                     test_transformer,
#     #                                     is_path_dataset=dataset.is_path_dataset,
#     #                                     images_path=dataset.images_path)
#     #
#     # else:
#     #
#     #     x = dataset.x
#     #
#     #     index_list = np.arange(len(x))
#     #
#     #     random_state.shuffle(index_list)
#     #
#     #     _dev_split = int(len(index_list) * (1 - dev_split))
#     #
#     #     train = x[index_list[:_dev_split]]
#     #     dev = x[index_list[_dev_split:]]
#     #
#     #     new_dataset = UnsupervisedDataset(x=dataset.x,
#     #                                       train=train,
#     #                                       test=dataset.get_indexes(
#     #                                           DatasetSplits.TEST),
#     #                                       dev=dev,
#     #                                       transformer=
#     #                                       dataset.transformer,
#     #                                       test_transformer=dataset.
#     #                                       test_transformer,
#     #                                       is_path_dataset=
#     #                                       dataset.is_path_dataset,
#     #                                       images_path=dataset.images_path)
#
#     return new_dataset
