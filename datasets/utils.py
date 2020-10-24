# __all__ = ['split_dataset']


# def _get_balanced_index(y: np.ndarray, test_split: float, dev_split: float = 0,
#                         random_state: np.random.RandomState = None) -> Tuple[list, list, list]:
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
#         # print(test_split, dev_split)
#
#         train.extend(index_list[_test_split + _dev_split:])
#         test.extend(index_list[:_test_split])
#         dev.extend(index_list[_test_split:_test_split + _dev_split])
#
#     assert sum(map(len, [train, test, dev])) == len(y)
#     # print(len(train), len(test), len(dev))
#
#     return train, dev, test
#
#
# def _get_split_index(y: np.ndarray, test_split: float, dev_split: float = 0,
#                      random_state: np.random.RandomState = None) -> Tuple[list, list, list]:
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
#     return train, dev, test
#
#
# def split_dataset(x: np.ndarray, y: np.ndarray, test_split: float, dev_split: float = 0,
#                   balance_labels: bool = True, old_y: np.ndarray = None,
#                   random_state: np.random.RandomState = None) -> Tuple[tuple, tuple, tuple]:
#
#     if balance_labels:
#         train, dev, test = _get_balanced_index(y, test_split, dev_split, random_state)
#     else:
#         train, dev, test = _get_split_index(y, test_split, dev_split, random_state)
#
#     train = (x[train], y[train], old_y[train] if old_y is not None else None)
#     test = (x[test], y[test], old_y[test] if old_y is not None else None)
#     dev = (x[dev], y[dev], old_y[dev] if old_y is not None else None)
#
#     return train, dev, test
