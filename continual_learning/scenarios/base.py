__all__ = ['StreamDataset',
           'AbstractTask',
           'TasksGenerator']

from abc import ABC, abstractmethod
from functools import wraps
from typing import Union, Tuple, Sequence, Any

import numpy as np

# from continual_learning.datasets.base import SupervisedDataset, \
#     UnsupervisedDataset, DatasetSplits, AbstractDataset, DatasetSplitsContainer, \
#     BaseDataset
from continual_learning.datasets.base import DatasetSplitsContainer, \
    AbstractDataset, BaseDataset, DatasetSplits

IndexesType = Union[list, np.ndarray]


class conditional_decorator(object):
    def __init__(self, instance):
        self.instance = instance

    def __call__(self, func):
        if not isinstance(self.instance, DatasetSplitsContainer):
            raise AttributeError(f'Trying to call the function {func} of a task'
                                 f' that does not contains a '
                                 f'DatasetSplitsContainer '
                                 f'({type(self.instance)}).')
        return func


def conditional_split_function(method):
    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        if not isinstance(self.base_dataset, DatasetSplitsContainer):
            raise AttributeError(f'Trying to call the function '
                                 f'{method.__name__} '
                                 f'of a task that does not contains a '
                                 f'DatasetSplitsContainer '
                                 f'({type(self.base_dataset).__name__}).')
        return method(self, *method_args, **method_kwargs)
    return _impl


def conditional_split_property(method):
    # @wraps(method)
    # def getter(self):
    #     if not isinstance(self.base_dataset, DatasetSplitsContainer):
    #         raise AttributeError(f'Trying to call a property '
    #                              f'{method.__name__} '
    #                              f'of a task that does not contains a '
    #                              f'DatasetSplitsContainer '
    #                              f'({type(self.base_dataset).__name__}).')
    #     """Enhance the property"""
    #     return method(self)
    #
    @property
    def wrapper(self, *args, **kwargs):
        if not isinstance(self.base_dataset, DatasetSplitsContainer):
            raise AttributeError(f'Trying to call a property '
                                 f'{method.__name__} '
                                 f'of a task that does not contains a '
                                 f'DatasetSplitsContainer '
                                 f'({type(self.base_dataset).__name__}).')
        return method(self, *args, **kwargs)

    return wrapper


class StreamDataset(ABC):
    def __init__(self):
        pass


class AbstractTask(ABC):

    def __init__(self,
                 *,
                 base_dataset: AbstractDataset,
                 task_index: int,
                 **kwargs):
        super().__init__(**kwargs)

        self._base_dataset = base_dataset
        self.task_index = task_index

    # @abstractmethod
    def __len__(self):
        return len(self.base_dataset)

    # @abstractmethod
    def __getitem__(self, item):
        return self.base_dataset[item]

    @property
    def base_dataset(self) -> Union[AbstractDataset, DatasetSplitsContainer]:
        return self._base_dataset

    @property
    def classes(self) -> Union[Sequence[int], None]:
        return self.base_dataset.classes

    @property
    def values(self) -> Sequence[Any]:
        return self.base_dataset.values

    @property
    def targets(self) -> Sequence[Any]:
        return self.base_dataset.targets

    @conditional_split_property
    def current_split(self) -> DatasetSplits:
        return self.base_dataset.current_split

    @current_split.setter
    def current_split(self, v: Union[DatasetSplits, int, str]) -> None:
        if isinstance(v, (str, int)):
            v = DatasetSplits(v)

        if v == DatasetSplits.TRAIN:
            self.base_dataset.train()
        elif v == DatasetSplits.TEST:
            self.base_dataset.test()
        elif v == DatasetSplits.DEV:
            self.base_dataset.dev()

    @property
    def current_dataset(self):
        return self.base_dataset.current_dataset

    @conditional_split_function
    def get_split(self, split: Union[DatasetSplits, str]):
        self.base_dataset.get_split(split)

    @conditional_split_function
    def train(self) -> None:
        self.base_dataset.train()

    @conditional_split_function
    def train_split(self) -> DatasetSplitsContainer:
        return self.base_dataset.train_split()

    @conditional_split_function
    def test(self) -> None:
        self.base_dataset.test()

    @conditional_split_function
    def test_split(self) -> DatasetSplitsContainer:
        return self.base_dataset.test_split()

    @conditional_split_function
    def dev(self) -> None:
        self.base_dataset.dev()

    @conditional_split_function
    def dev_split(self) -> DatasetSplitsContainer:
        return self.get_dataset(DatasetSplits.DEV)

    @conditional_split_function
    def dev_split(self) -> DatasetSplitsContainer:
        return self.get_dataset(DatasetSplits.DEV)

    def get_subset(self,
                   subset: IndexesType = None,
                   train_subset: IndexesType = None,
                   test_subset: IndexesType = None,
                   dev_subset: IndexesType = None,
                   as_splitted_dataset: bool = False,
                   **kwargs) -> Union[DatasetSplitsContainer,
                                      Tuple[AbstractDataset,
                                            AbstractDataset,
                                            AbstractDataset]]:

        return self.base_dataset.get_subset(subset=subset,
                                            train_subset=train_subset,
                                            test_subset=test_subset,
                                            dev_subset=dev_subset,
                                            as_splitted_dataset=
                                            as_splitted_dataset)

    @conditional_split_function
    def get_dataset(self, split: Union[DatasetSplits, str],
                    **kwargs) -> BaseDataset:

        return self.base_dataset.get_split(split)


class TaskSplitContainer(AbstractTask):

    def __init__(self,
                 *,
                 base_dataset: DatasetSplitsContainer,
                 task_index: int,
                 **kwargs):
        super().__init__(**kwargs)

        self._base_dataset: DatasetSplitsContainer = base_dataset
        self.task_index = task_index

    @property
    def current_split(self) -> BaseDataset:
        return self.base_dataset.current_split

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
    def base_dataset(self) -> DatasetSplitsContainer:
        return self._base_dataset

    @property
    def current_dataset(self):
        return self.base_dataset.current_dataset

    @property
    def classes(self):
        return self.base_dataset.classes

    @property
    def values(self):
        return self.base_dataset.values

    @property
    def targets(self):
        return self.base_dataset.targets

    def train(self) -> None:
        self.base_dataset.train()

    def train_split(self) -> AbstractDataset:
        return self.base_dataset.train_split()

    def test(self) -> None:
        self.base_dataset.test()

    def test_split(self) -> AbstractDataset:
        return self.base_dataset.test_split()

    def dev(self) -> None:
        self.base_dataset.dev()

    def dev_split(self) -> AbstractDataset:
        return self.get_dataset(DatasetSplits.DEV)

    def get_subset(self,
                   train_subset: IndexesType,
                   test_subset: IndexesType = None,
                   dev_subset: IndexesType = None,
                   as_splitted_dataset: bool = False,
                   **kwargs) -> Union[DatasetSplitsContainer,
                                      Tuple[AbstractDataset,
                                            AbstractDataset,
                                            AbstractDataset]]:

        return self.base_dataset.get_subset(train_subset,
                                            test_subset,
                                            dev_subset,
                                            as_splitted_dataset=
                                            as_splitted_dataset)

    def get_dataset(self, split: Union[DatasetSplits, str],
                    **kwargs) -> BaseDataset:

        return self.base_dataset.get_split(split)


class TasksGenerator(ABC):
    def __init__(self,
                 dataset: DatasetSplitsContainer,
                 random_state: Union[np.random.RandomState, int] = None,
                 **kwargs):

        super().__init__()

        if random_state is not None:
            if isinstance(random_state, int):
                random_state = np.random.RandomState(random_state)
        else:
            random_state = np.random.RandomState(None)

        self.dataset = dataset
        self.random_state = random_state

    @abstractmethod
    def generate_task(self, dataset: AbstractDataset,
                      random_state: Union[np.random.RandomState, int] = None,
                      **kwargs) \
            -> Union[AbstractTask, None]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, i: int):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError
