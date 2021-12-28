__all__ = ['Task',
           'SupervisedTransformerTask',
           'TransformerTask']

from typing import Union, Callable, Any
from continual_learning.datasets.base import AbstractDataset, DatasetType
from continual_learning.scenarios.base import AbstractTask


class Task(AbstractTask):
    def __init__(self,
                 *,
                 base_dataset: AbstractDataset,
                 task_index: int,
                 labels_mapping: Union[dict, Callable[[Any], Any]] = None,
                 **kwargs):

        super().__init__(task_index=task_index,
                         base_dataset=base_dataset,
                         **kwargs)

        self.is_supervised = base_dataset.dataset_type == DatasetType.SUPERVISED
        self._task_labels = True
        self.labels_mapping = labels_mapping

        self._task_classes = None
        self._dataset_classes = sorted(base_dataset.classes)

        if self._dataset_classes is not None:
            self._task_classes = sorted(self._map_labels(self._dataset_classes))

    def use_task_labels(self):
        self._task_labels = True

    def use_dataset_labels(self):
        self._task_labels = False

    @property
    def task_labels(self):
        return self._task_classes

    @property
    def dataset_labels(self):
        return self._dataset_classes

    @property
    def classes(self):
        if self._task_labels:
            return self.task_labels
        else:
            return self.dataset_labels

    def _map_labels(self, y):
        if self.labels_mapping is None:
            return y

        if isinstance(self.labels_mapping, dict):
            if self._task_labels:
                if not isinstance(y, list):
                    y = self.labels_mapping[y]
                else:
                    y = [self.labels_mapping[i] for i in y]
        else:
            y = list(map(self.labels_mapping, y))

        return y

    @property
    def targets(self):
        base_target = self.base_dataset.targets

        if base_target is not None:
            return self._map_labels(base_target)

        return None

    @property
    def values(self):
        return self.base_dataset.values

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, item):
        y = None

        a = self.base_dataset[item]

        if len(a) == 3:
            i, x, y = a
            y = self._map_labels(y)
        else:
            i, x = a

        if y is not None:
            return i, x, y

        return i, x


class TransformerTask(Task):
    def __init__(self,
                 *,
                 base_dataset: AbstractDataset,
                 transformer: Callable[[Any], Any],
                 task_index: int,
                 labels_mapping: Union[dict, Callable[[Any], Any]] = None,
                 **kwargs):

        super().__init__(task_index=task_index,
                         base_dataset=base_dataset,
                         labels_mapping=labels_mapping,
                         **kwargs)

        self.transformer = transformer

    # def __len__(self):
    #     return len(self.base_dataset)

    @property
    def values(self):
        return list(map(self.transformer, self.base_dataset.values))

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


class SupervisedTransformerTask(Task):
    def __init__(self,
                 *,
                 base_dataset: AbstractDataset,
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
    def values(self):
        return list(map(self.transformer, super().values))

    def __getitem__(self, item):
        i, x, y = super().__getitem__(item)

        if isinstance(i, list):
            x = list(map(self.transformer, x))
        else:
            x = self.transformer(x)

        y = self._map_labels(y)

        return i, x, y
