__all__ = ['SupervisedTask',
           'SupervisedTransformerTask',
           'TransformerTask']

from typing import Union, Callable, Any

from continual_learning.datasets.base import UnsupervisedDataset, \
    SupervisedDataset, DatasetSplits
from continual_learning.scenarios.base import Task


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


class TransformerTask(Task):
    def __init__(self,
                 *,
                 base_dataset: Union[UnsupervisedDataset, SupervisedDataset,
                                     Task],
                 transformer: Callable[[Any], Any],
                 task_index: int,
                 **kwargs):

        train, dev, test = [base_dataset.get_indexes(DatasetSplits(v))
                            for v in ['train', 'dev', 'test']]

        super().__init__(base_dataset=base_dataset,
                         train=train,
                         dev=dev,
                         test=test,
                         task_index=task_index,
                         **kwargs)

        self.transformer = transformer

    def __len__(self):
        return len(self.base_dataset)

    @property
    def x(self):
        return list(map(self.transformer, self.base_dataset.x))

    def __getitem__(self, item):
        y = None

        a = super().__getitem__(item)

        if len(a) == 3:
            i, x, y = a
        else:
            i, x = a

        if isinstance(i, list):
            x = list(map(self.transformer, x))
        else:
            x = self.transformer(x)

        if y is not None:
            return i, x, y

        return i, x


class SupervisedTransformerTask(SupervisedTask):
    def __init__(self,
                 *,
                 base_dataset: SupervisedDataset,
                 transformer: Callable[[Any], Any],
                 task_index: int,
                 labels_mapping: Union[dict, None],
                 **kwargs):

        super().__init__(base_dataset=base_dataset,
                         task_index=task_index,
                         labels_mapping=labels_mapping,
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
