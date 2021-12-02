__all__ = ['split_dataset', 'extract_dev']

from collections import defaultdict
from copy import deepcopy

from typing import Union, Tuple

import numpy as np

from continual_learning.datasets.base import SupervisedDataset, \
    UnsupervisedDataset, DatasetSplits, DatasetSplitContexView


def _get_balanced_index(y: Union[list, np.ndarray],
                        test_split: float,
                        dev_split: float = 0,
                        random_state: np.random.RandomState = None) \
        -> Tuple[list, list, list]:
    train = []
    test = []
    dev = []

    d = defaultdict(list)

    for i, _y in enumerate(y):
        d[_y].append(i)

    for index_list in d.values():

        if random_state is not None:
            random_state.shuffle(index_list)
        else:
            np.random.shuffle(index_list)

        ln = len(index_list)

        _test_split = int(ln * test_split)
        _dev_split = int(ln * dev_split)

        train.extend(index_list[_test_split + _dev_split:])
        test.extend(index_list[:_test_split])
        dev.extend(index_list[_test_split:_test_split + _dev_split])

    return train, test, dev


def _get_split_index(y: Union[list, np.ndarray],
                     test_split: float,
                     dev_split: float = 0,
                     random_state: np.random.RandomState = None) \
        -> Tuple[list, list, list]:
    index_list = np.arange(len(y))

    if random_state is not None:
        random_state.shuffle(index_list)
    else:
        np.random.shuffle(index_list)

    ln = len(index_list)

    _test_split = int(ln * test_split)
    _dev_split = int(ln * dev_split)

    train = index_list[_test_split + _dev_split:]
    test = index_list[:_test_split]
    dev = index_list[_test_split:_test_split + _dev_split]

    return train, test, dev


def extract_dev(y: Union[list, np.ndarray],
                dev_split: float = 0.1,
                random_state: Union[np.random.RandomState, int] = None):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    y = np.asarray(y)

    index_list = np.arange(len(y))

    random_state.shuffle(index_list)

    _dev_split = int(len(index_list) * (1 - dev_split))

    train = y[index_list[:_dev_split]]
    dev = y[index_list[_dev_split:]]

    return train, dev


def split_dataset(y: Union[list, np.ndarray],
                  test_split: float,
                  dev_split: float = 0,
                  balance_labels: bool = True,
                  random_state: Union[np.random.RandomState, int] = None) -> \
        Tuple[list, list, list]:
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    if balance_labels:
        train, test, dev = _get_balanced_index(y, test_split, dev_split,
                                               random_state)
    else:
        train, test, dev = _get_split_index(y, test_split, dev_split,
                                            random_state)

    assert sum(map(len, [train, test, dev])) == len(y)

    return train, test, dev


def create_dataset_with_dev_split(dataset: Union[SupervisedDataset,
                                                 UnsupervisedDataset],
                                  dev_percentage: float = 0.1,
                                  balance_labels: bool = False,
                                  random_state: Union[np.random.RandomState,
                                                      int] = None):

    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        random_state = np.random.RandomState(None)

    is_supervised = False

    train_split = dataset.get_indexes(DatasetSplits.TRAIN)
    test_split = dataset.get_indexes(DatasetSplits.TEST)
    dev_split = dataset.get_indexes(DatasetSplits.DEV)

    if isinstance(dataset, SupervisedDataset):
        is_supervised = True
        c = SupervisedDataset
        y = dataset.y
    else:
        c = UnsupervisedDataset
        y = dataset.x

    indexes = deepcopy(np.concatenate((train_split, dev_split)))
    index_list = np.arange(len(indexes))

    random_state.shuffle(index_list)

    _dev_split = int(len(index_list) * (1 - dev_percentage))

    train = indexes[index_list[:_dev_split]]
    dev = indexes[index_list[_dev_split:]]

    new_dataset = c(x=dataset._x,
                    y=dataset._y if is_supervised else None,
                    train=train,
                    test=test_split,
                    dev=dev,
                    transformer=dataset.transformer,
                    test_transformer=dataset.
                    test_transformer,
                    target_transformer=dataset.
                    test_transformer,
                    is_path_dataset=dataset.is_path_dataset,
                    images_path=dataset.images_path)
    # else:
    #     new_dataset = UnsupervisedDataset(x=dataset.x,
    #                                       y=dataset.y,
    #                                       train=train,
    #                                       test=test_split,
    #                                       dev=dev,
    #                                       transformer=dataset.transformer,
    #                                       test_transformer=dataset.
    #                                       test_transformer,
    #                                       target_transformer=dataset.
    #                                       test_transformer,
    #                                       is_path_dataset=dataset.
    #                                       is_path_dataset,
    #                                       images_path=dataset.images_path)
    #
    # if isinstance(dataset, SupervisedDataset):
    #
    #     y = dataset.y
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
    #     new_dataset = SupervisedDataset(x=dataset.x,
    #                                     y=dataset.y,
    #                                     train=train,
    #                                     test=dataset.get_indexes(
    #                                         DatasetSplits.TEST),
    #                                     dev=dev,
    #                                     transformer=dataset.transformer,
    #                                     test_transformer=dataset.
    #                                     test_transformer,
    #                                     target_transformer=dataset.
    #                                     test_transformer,
    #                                     is_path_dataset=dataset.is_path_dataset,
    #                                     images_path=dataset.images_path)
    #
    # else:
    #
    #     x = dataset.x
    #
    #     index_list = np.arange(len(x))
    #
    #     random_state.shuffle(index_list)
    #
    #     _dev_split = int(len(index_list) * (1 - dev_split))
    #
    #     train = x[index_list[:_dev_split]]
    #     dev = x[index_list[_dev_split:]]
    #
    #     new_dataset = UnsupervisedDataset(x=dataset.x,
    #                                       train=train,
    #                                       test=dataset.get_indexes(
    #                                           DatasetSplits.TEST),
    #                                       dev=dev,
    #                                       transformer=
    #                                       dataset.transformer,
    #                                       test_transformer=dataset.
    #                                       test_transformer,
    #                                       is_path_dataset=
    #                                       dataset.is_path_dataset,
    #                                       images_path=dataset.images_path)

    return new_dataset



def create_dataset_with_new_split(dataset: Union[SupervisedDataset,
                                                 UnsupervisedDataset],
                                  test_percentage: float = 0.2,
                                  dev_percentage: float = 0,
                                  balance_labels: bool = False,
                                  random_state: Union[np.random.RandomState,
                                                      int] = None):

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

    is_supervised = False

    train_split = dataset.get_indexes(DatasetSplits.TRAIN)
    test_split = dataset.get_indexes(DatasetSplits.TEST)
    dev_split = dataset.get_indexes(DatasetSplits.DEV)

    if isinstance(dataset, SupervisedDataset):
        is_supervised = True
        c = SupervisedDataset
        # y = dataset.y
    else:
        c = UnsupervisedDataset
        # y = dataset.x

    indexes = deepcopy(np.concatenate((train_split, dev_split, test_split)))
    index_list = np.arange(len(indexes))

    random_state.shuffle(index_list)

    _dev_split = int(len(index_list) * dev_percentage)
    _test_split = int(len(index_list) * test_percentage)

    dev = indexes[index_list[: _dev_split]]
    test = indexes[index_list[_dev_split: _dev_split + _test_split]]
    train = indexes[index_list[ _dev_split + _test_split:]]

    new_dataset = c(x=dataset._x,
                    y=dataset._y if is_supervised else None,
                    train=train,
                    test=test,
                    dev=dev,
                    transformer=dataset.transformer,
                    test_transformer=dataset.
                    test_transformer,
                    target_transformer=dataset.
                    test_transformer,
                    is_path_dataset=dataset.is_path_dataset,
                    images_path=dataset.images_path)
    # else:
    #     new_dataset = UnsupervisedDataset(x=dataset.x,
    #                                       y=dataset.y,
    #                                       train=train,
    #                                       test=test_split,
    #                                       dev=dev,
    #                                       transformer=dataset.transformer,
    #                                       test_transformer=dataset.
    #                                       test_transformer,
    #                                       target_transformer=dataset.
    #                                       test_transformer,
    #                                       is_path_dataset=dataset.
    #                                       is_path_dataset,
    #                                       images_path=dataset.images_path)
    #
    # if isinstance(dataset, SupervisedDataset):
    #
    #     y = dataset.y
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
    #     new_dataset = SupervisedDataset(x=dataset.x,
    #                                     y=dataset.y,
    #                                     train=train,
    #                                     test=dataset.get_indexes(
    #                                         DatasetSplits.TEST),
    #                                     dev=dev,
    #                                     transformer=dataset.transformer,
    #                                     test_transformer=dataset.
    #                                     test_transformer,
    #                                     target_transformer=dataset.
    #                                     test_transformer,
    #                                     is_path_dataset=dataset.is_path_dataset,
    #                                     images_path=dataset.images_path)
    #
    # else:
    #
    #     x = dataset.x
    #
    #     index_list = np.arange(len(x))
    #
    #     random_state.shuffle(index_list)
    #
    #     _dev_split = int(len(index_list) * (1 - dev_split))
    #
    #     train = x[index_list[:_dev_split]]
    #     dev = x[index_list[_dev_split:]]
    #
    #     new_dataset = UnsupervisedDataset(x=dataset.x,
    #                                       train=train,
    #                                       test=dataset.get_indexes(
    #                                           DatasetSplits.TEST),
    #                                       dev=dev,
    #                                       transformer=
    #                                       dataset.transformer,
    #                                       test_transformer=dataset.
    #                                       test_transformer,
    #                                       is_path_dataset=
    #                                       dataset.is_path_dataset,
    #                                       images_path=dataset.images_path)

    return new_dataset