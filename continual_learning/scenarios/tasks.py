# from typing import Union, Callable
#
# import numpy as np
# from torch.utils.data import DataLoader
#
# from continual_learning.datasets.base import IndexesContainer, \
#     UnsupervisedDataset, SupervisedDataset, DatasetSplits
#
#
# class Task(IndexesContainer):
#     def __init__(self,
#                  *,
#                  base_dataset: Union[UnsupervisedDataset, SupervisedDataset],
#                  index: int,
#                  train: Union[list, np.ndarray],
#                  test: [list, np.ndarray] = None,
#                  dev: [list, np.ndarray] = None,
#                  **kwargs):
#
#         super().__init__(train=train, dev=dev, test=test, **kwargs)
#
#         self.base_dataset = base_dataset
#         self.index = index
#
#     def __len__(self):
#         return len(self.base_dataset)
#
#     def __getitem__(self, item):
#         idx = self.current_indexes
#         i = idx[item]
#         a = self.base_dataset[i]
#         return a
#
#     @IndexesContainer.current_split.setter
#     def current_split(self, v: DatasetSplits) -> None:
#         self._current_split = v
#         self.base_dataset.current_split = v
#
#     # def get_iterator(self,
#     #                  batch_size: int,
#     #                  shuffle: bool = True,
#     #                  sampler=None,
#     #                  num_workers: int = 0,
#     #                  pin_memory: bool = False):
#     #     return DataLoader(self,
#     #                       batch_size=batch_size,
#     #                       shuffle=shuffle,
#     #                       sampler=sampler,
#     #                       pin_memory=pin_memory,
#     #                       num_workers=num_workers)
#
#
# class SupervisedTask(Task):
#     def __init__(self,
#                  *,
#                  base_dataset: SupervisedDataset,
#                  index: int,
#                  labels_mapping: dict,
#                  train: Union[list, np.ndarray],
#                  test: [list, np.ndarray] = None,
#                  dev: [list, np.ndarray] = None,
#                  **kwargs):
#
#         super().__init__(index=index,
#                          base_dataset=base_dataset,
#                          train=train,
#                          dev=dev,
#                          test=test,
#                          **kwargs)
#
#         self._task_labels = True
#         self.labels_mapping = labels_mapping
#
#     def set_task_labels(self):
#         self._task_labels = True
#
#     def set_dataset_labels(self):
#         self._task_labels = False
#
#     @property
#     def labels(self):
#         if self._task_labels:
#             return list(self.labels_mapping.values())
#         else:
#             return list(self.labels_mapping.keys())
#
#     def _map_labels(self, y):
#         if self.labels_mapping is None:
#             return y
#
#         if self._task_labels:
#             if not isinstance(y, list):
#                 return self.labels_mapping[y]
#             else:
#             # if self._task_labels:
#                 y = [self.labels_mapping[i] for i in y]
#
#         return y
#
#     @property
#     def data(self):
#         return self.x, self.y
#
#     def y(self, split: DatasetSplits = None):
#         return self._map_labels(self.base_dataset.y(split))
#
#     def x(self, split: DatasetSplits = None):
#         return self.base_dataset.x(split)
#
#     def __getitem__(self, item):
#         i, x, y = super().__getitem__(item)
#         y = self._map_labels(y)
#         return i, x, y
#
#
# class UnsupervisedTransformerTask(Task):
#     def __init__(self,
#                  *,
#                  base_dataset: UnsupervisedDataset,
#                  transformer: Callable,
#                  index: int,
#                  train: Union[list, np.ndarray],
#                  test: [list, np.ndarray] = None,
#                  dev: [list, np.ndarray] = None,
#                  **kwargs):
#         super().__init__(base_dataset=base_dataset,
#                          train=train,
#                          dev=dev,
#                          test=test,
#                          index=index,
#                          **kwargs)
#         self.transformer = transformer
#
#     def __getitem__(self, item):
#         print(isinstance(self.base_dataset, UnsupervisedDataset))
#         i, x = super().__getitem__(item)
#         if isinstance(i, list):
#             x = list(map(self.transformer, x))
#         else:
#             x = self.transformer(x)
#         return i, x
#
#
# class SupervisedTransformerTask(SupervisedTask):
#     def __init__(self,
#                  *,
#                  base_dataset: SupervisedDataset,
#                  transformer: Callable,
#                  index: int,
#                  labels_mapping: Union[dict, None],
#                  train: Union[list, np.ndarray],
#                  test: [list, np.ndarray] = None,
#                  dev: [list, np.ndarray] = None,
#                  **kwargs):
#
#         super().__init__(base_dataset=base_dataset,
#                          labels_mapping=labels_mapping,
#                          train=train,
#                          dev=dev,
#                          test=test,
#                          index=index,
#                          **kwargs)
#
#         self.transformer = transformer
#
#     def x(self, split: DatasetSplits = None):
#         return list(map(self.transformer, super().x(split)))
#
#     def __getitem__(self, item):
#         i, x, y = super().__getitem__(item)
#         if isinstance(i, list):
#             x = list(map(self.transformer, x))
#         else:
#             x = self.transformer(x)
#         return i, x, y
from typing import Union, Callable, Any

import numpy as np

from continual_learning.datasets.base import UnsupervisedDataset, \
    SupervisedDataset, DatasetSplits
from continual_learning.scenarios.base import Task


class UnsupervisedTransformerTask(Task):
    def __init__(self,
                 *,
                 base_dataset: UnsupervisedDataset,
                 transformer: Callable,
                 index: int,
                 train: Union[list, np.ndarray],
                 test: [list, np.ndarray] = None,
                 dev: [list, np.ndarray] = None,
                 **kwargs):

        super().__init__(base_dataset=base_dataset,
                         train=train,
                         dev=dev,
                         test=test,
                         index=index,
                         **kwargs)

        self.transformer = transformer

    @property
    def x(self, split: DatasetSplits = None):
        return list(map(self.transformer, super().x))

    def __getitem__(self, item):
        print(isinstance(self.base_dataset, UnsupervisedDataset))
        i, x = super().__getitem__(item)
        if isinstance(i, list):
            x = list(map(self.transformer, x))
        else:
            x = self.transformer(x)
        return i, x


class SupervisedTask(Task):
    def __init__(self,
                 *,
                 base_dataset: SupervisedDataset,
                 task_index: int,
                 labels_mapping: dict,
                 **kwargs):

        super().__init__(task_index=task_index,
                         base_dataset=base_dataset,
                         **kwargs)

        self._task_labels = True
        self.labels_mapping = labels_mapping

    def set_task_labels(self):
        self._task_labels = True

    def set_dataset_labels(self):
        self._task_labels = False

    def get_task_labels(self):
        return list(self.labels_mapping.values())

    def get_dataset_labels(self):
        return list(self.labels_mapping.keys())

    @property
    def labels(self):
        if self._task_labels:
            return self.get_task_labels()
        else:
            return self.get_dataset_labels()

    def _map_labels(self, y):
        if self.labels_mapping is None:
            return y

        if self._task_labels:
            if not isinstance(y, list):
                return self.labels_mapping[y]
            else:
                y = [self.labels_mapping[i] for i in y]

        return y

    @property
    def data(self):
        return self.x, self.y

    @property
    def y(self):
        return self._map_labels(self.base_dataset.y)

    @property
    def x(self):
        return self.base_dataset.x

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, item):
        i, x, y = self.base_dataset[item]
        y = self._map_labels(y)
        return i, x, y


class SupervisedTransformerTask(SupervisedTask):
    def __init__(self,
                 *,
                 base_dataset: SupervisedDataset,
                 transformer: Callable,
                 index: int,
                 labels_mapping: Union[dict, None],
                 train: Union[list, np.ndarray],
                 test: [list, np.ndarray] = None,
                 dev: [list, np.ndarray] = None,
                 **kwargs):

        super().__init__(base_dataset=base_dataset,
                         labels_mapping=labels_mapping,
                         train=train,
                         dev=dev,
                         test=test,
                         index=index,
                         **kwargs)

        self.transformer = transformer

    @property
    def x(self, split: DatasetSplits = None):
        return list(map(self.transformer, super().x))

    def __getitem__(self, item):
        i, x, y = super().__getitem__(item)
        if isinstance(i, list):
            x = list(map(self.transformer, x))
        else:
            x = self.transformer(x)
        return i, x, y


class TransformerTask(Task):
    def __init__(self,
                 *,
                 base_dataset: Union[UnsupervisedDataset, SupervisedDataset,
                                     SupervisedTask],
                 transformer: Callable[[Any], Any],
                 index: int,
                 **kwargs):

        train, dev, test = [base_dataset.get_indexes(DatasetSplits(v))
                            for v in ['train', 'dev', 'test']]

        super().__init__(base_dataset=base_dataset,
                         train=train,
                         dev=dev,
                         test=test,
                         index=index,
                         **kwargs)

        self.transformer = transformer

    def __getitem__(self, item):
        y = None

        a = super().__getitem__(item)
        if len(a) == 3:
            i, x, y = a
        else:
            i, x = a

        # if isinstance(self.base_dataset, UnsupervisedDataset):
        #     i, x = super().__getitem__(item)
        # else:
        #     i, x, y = super().__getitem__(item)

        if isinstance(i, list):
            x = list(map(self.transformer, x))
        else:
            x = self.transformer(x)

        if y is not None:
            return i, x, y

        return i, x
