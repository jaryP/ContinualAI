from itertools import count
from typing import Union, List, Type, Any, Callable

import numpy as np

from continual_learning.datasets.base import UnsupervisedDataset, \
    SupervisedDataset
from continual_learning.scenarios.base import DomainIncremental
from continual_learning.scenarios.classification.new_instances.utils import \
    TransformerTask, ImageRotation, PixelsPermutation
from continual_learning.scenarios.tasks import Task


class GenericTransformingScenario(DomainIncremental):
    def __init__(self,
                 dataset: Union[SupervisedDataset, UnsupervisedDataset],
                 transform_factory: Callable[[Any], Callable],
                 tasks_n: int,
                 transformation_parameters: Union[List[any],
                                                  Callable[[Any], Any]],
                 infinite_stream: bool = False,
                 random_state: Union[np.random.RandomState, int] = None,
                 lazy_initialization: bool = True,
                 **kwargs):

        super().__init__(dataset,
                         random_state=random_state,
                         **kwargs)

        if infinite_stream and not callable(transformation_parameters):
            raise ValueError(f'Setting infinite_stream=True requires the '
                             f'transformation_parameters parameter '
                             f'as callable '
                             f'{type(transformation_parameters)}')
        else:
            if tasks_n < 0:
                raise ValueError(f'Argument tasks_n=True must be '
                                 f'greater than 0 '
                                 f'{type(tasks_n)}')

            if callable(transformation_parameters):
                transformation_parameters = [
                    transformation_parameters(task=task,
                                              random_state=
                                              self.random_state)
                    for task in range(tasks_n)]

        self.infinite_stream = infinite_stream or infinite_stream
        self.lazy_initialization = lazy_initialization

        self.parameters = transformation_parameters
        self.task_n = tasks_n
        self.transform_function = transform_factory

        self._t_counter = 0
        self._current_task = 0

        self._transform_functions = []
        self._tasks_generated = []

        if not lazy_initialization:
            for _ in range(tasks_n):
                t = self.generate_task()
                # self._tasks_generated.append(t)

    def __len__(self):
        if self.infinite_stream:
            return np.inf
        else:
            return self.task_n

    def __getitem__(self, i: int):
        if i > len(self._tasks_generated):
            if self.infinite_stream:
                raise ValueError(f'Attempting to get a non generated task from '
                                 f'an infinite stream of tasks '
                                 f'(generated tasks: '
                                 f'({len(self._tasks_generated) - 1})), '
                                 f'index: {i})')
            else:
                raise ValueError(f'Attempting to get a non generated task from '
                                 f'an lazy created stream of tasks (index: {i})'
                                 f'. Generate the task or set '
                                 f'lazy_initialization=False when '
                                 f'instantiating this class.')

        return self._tasks_generated[i]

    def generate_task(self, **kwargs) -> Union[Task, None]:

        counter = len(self._tasks_generated)

        if self.infinite_stream and callable(self.parameters):
            t_parameters = self.parameters(task=counter,
                                           random_state=self.random_state)
        else:
            if counter == self.task_n:
                return None

            t_parameters = self.parameters[counter]

        t = self.transform_function(t_parameters)

        task = TransformerTask(base_dataset=self.dataset, transformer=t,
                               index=counter)

        self._tasks_generated.append(task)

        return task

    def __next__(self):
        self._current_task = 0
        return self

    def __iter__(self):
        while True:
            if not self.infinite_stream:
                if self._current_task >= self.task_n or \
                        self._current_task >= len(self._tasks_generated):
                    return
            # if self._current_task <= self.task_n or self.infinite_stream:

            if len(self._tasks_generated) >= self._current_task:
                t = self._tasks_generated[self._current_task]
            else:
                t = self.generate_task()

                if t is None:
                    return

            self._current_task += 1
            yield t
            # else:
            #     raise StopIteration

    # def __iter__(self):
    #     i = 0
    #     while i < self.task_n or self.infinite_stream:
    #         task = self.generate_task()
    #         if task is None:
    #             return
    #         yield task


class ImageRotationScenario(GenericTransformingScenario):
    def __init__(self,
                 dataset: Union[SupervisedDataset, UnsupervisedDataset],
                 tasks_n: int,
                 transformation_parameters: Union[List[any],
                                                  Callable[[Any], Any]],
                 infinite_stream: bool = False,
                 random_state: Union[np.random.RandomState, int] = None,
                 lazy_initialization: bool = True,
                 **kwargs):

        transform_function = self.get_rotation

        super().__init__(dataset=dataset,
                         tasks_n=tasks_n,
                         transform_factory=transform_function,
                         transformation_parameters=transformation_parameters,
                         infinite_stream=infinite_stream,
                         random_state=random_state,
                         lazy_initialization=lazy_initialization,
                         **kwargs)

    def get_rotation(self, degree, **kwargs):
        return ImageRotation(degree)


class PixelsPermutationScenario(GenericTransformingScenario):
    def __init__(self,
                 dataset: Union[SupervisedDataset, UnsupervisedDataset],
                 tasks_n: int,
                 transformation_parameters: Union[List[any],
                                                  Callable[[Any], Any]],
                 infinite_stream: bool = False,
                 random_state: Union[np.random.RandomState, int] = None,
                 lazy_initialization: bool = True,
                 **kwargs):

        transform_factory = self.get_permutation

        super().__init__(dataset=dataset,
                         tasks_n=tasks_n,
                         transform_factory=transform_factory,
                         transformation_parameters=transformation_parameters,
                         infinite_stream=infinite_stream,
                         random_state=random_state,
                         lazy_initialization=lazy_initialization,
                         **kwargs)

    def get_permutation(self, permutation, **kwargs):
        return PixelsPermutation(permutation)
