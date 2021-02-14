__all__ = ['split_dataset', 'extract_dev']

from collections import defaultdict
from typing import Union, Tuple

import numpy as np


def _get_balanced_index(y: Union[list,  np.ndarray], test_split: float, dev_split: float = 0,
                        random_state: np.random.RandomState = None) -> Tuple[list, list, list]:
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


def _get_split_index(y: Union[list,  np.ndarray], test_split: float, dev_split: float = 0,
                     random_state: np.random.RandomState = None) -> Tuple[list, list, list]:
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


def extract_dev(y: Union[list,  np.ndarray], dev_split: float = 0.1, random_state: Union[np.random.RandomState, int] = None):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    y = np.asarray(y)

    index_list = np.arange(len(y))

    random_state.shuffle(index_list)

    _dev_split = int(len(index_list) * (1 - dev_split))

    train = y[index_list[:_dev_split]]
    dev = y[index_list[_dev_split:]]

    return train, dev


def split_dataset(y: Union[list,  np.ndarray], test_split: float, dev_split: float = 0,
                  balance_labels: bool = True,
                  random_state: Union[np.random.RandomState, int] = None) -> Tuple[list, list, list]:
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    if balance_labels:
        train, test, dev = _get_balanced_index(y, test_split, dev_split, random_state)
    else:
        train, test, dev = _get_split_index(y, test_split, dev_split, random_state)

    assert sum(map(len, [train, test, dev])) == len(y)

    return train, test, dev
