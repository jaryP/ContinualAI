from collections import defaultdict
from typing import Union, List, Any, Callable, Dict

import numpy as np

from continual_learning.datasets.base import AbstractDataset, DatasetSplitsContainer

from continual_learning.scenarios.base import TasksGenerator
from continual_learning.scenarios.classification.new_classes import \
    NCTransformingScenario
from continual_learning.scenarios.classification.utils import \
    get_dataset_subset_using_labels
from continual_learning.scenarios.tasks import TransformerTask, Task
from continual_learning.scenarios.utils import ImageRotation, PixelsPermutation


class NCScenario(TasksGenerator):
    def __init__(self,
                 *,
                 tasks_n: int,
                 dataset: DatasetSplitsContainer,
                 # transform_factory: Callable[[Any], Callable],
                 # transformation_parameters: Union[List[any],
                 #                                  Callable[[Any], Any]],
                 random_state: Union[np.random.RandomState, int] = None,
                 lazy_initialization: bool = True,
                 labels_per_tasks: Dict[int, int] = None,
                 labels_task_mapping: Dict[int, int] = None,
                 shuffle_labels: bool = False,
                 remap_labels_across_task: bool = False,
                 remap_labels_in_task: bool = False,
                 **kwargs):

        super().__init__(dataset,
                         random_state=random_state,
                         **kwargs)

        dataset_labels = np.asarray(dataset.classes)
        assigned_labels = []

        if labels_task_mapping is None:
            labels_task_mapping = {}

        if labels_per_tasks is None:
            # if len(labels_task_mapping) == 0:
            if len(dataset_labels) % tasks_n != 0:
                raise ValueError(
                    f'Attempted to create labels_per_tasks dictionary, '
                    f'but the number of labels ({len(dataset_labels)}) '
                    f'cannot be distributed equally between the tasks '
                    f'({tasks_n}), '
                    f'because len(dataset_labels) % tasks_n != 0.')

            labels_per_tasks = {task: len(dataset_labels) // tasks_n
                                for task in range(tasks_n)}

        else:
            remaining_tasks = tasks_n - len(labels_per_tasks)

            if remaining_tasks > 0:
                assigned_tasks = sum(labels_per_tasks.values())
                remaining_labels = len(dataset_labels) - assigned_tasks
                labels_per_remaining_task = remaining_labels // remaining_tasks

                tasks_map = {i: labels_per_remaining_task
                             for i in range(tasks_n)
                             if i not in labels_per_tasks}

                labels_per_tasks.update(tasks_map)

                if any([v == 1 for v in labels_per_tasks.values()]):
                    raise ValueError('Due to the lack of tasks '
                                     'in labels_per_tasks, '
                                     'the dictionary has been populated, '
                                     'but some task has only '
                                     'one labels associated ot it. '
                                     'If intended, '
                                     'please force this behaviour by setting '
                                     f'labels_per_tasks = {labels_per_tasks}')

        if remap_labels_in_task and remap_labels_across_task:
            raise ValueError('Both remap_labels_in_task and '
                             'remap_labels_across_task are set to True '
                             'but are mutually exclusive. '
                             'Please set at least one to False.')

        if max(labels_per_tasks.keys()) >= tasks_n or min(
                labels_per_tasks.keys()) < 0:
            raise ValueError('Invalid key value in labels_per_tasks. '
                             f'The keys must be in  [0, {tasks_n - 1}] '
                             f'({labels_per_tasks.keys()})')

        if min(labels_per_tasks.values()) < 0:
            raise ValueError('Invalid value in labels_per_tasks. '
                             f'The values must be > 0'
                             f'({labels_per_tasks.keys()})')

        sm = sum(labels_per_tasks.values())
        if sm > len(dataset_labels):
            raise ValueError(f'The total number of classes in labels_per_tasks '
                             f'({sm}) exceeds the number of labels '
                             f'in the dataset ({len(dataset_labels)}).')

        if not all(label in dataset_labels
                   for label, task in labels_task_mapping.items()):
            raise ValueError(f'Some labels in labels_task_mapping are not '
                             f'present in the dataset. '
                             f'Dataset labels: {dataset_labels}, '
                             f'given labels: {labels_task_mapping}')

        if len(labels_task_mapping) > 0:
            if max(labels_task_mapping.keys()) > len(dataset_labels) - 1 \
                    or min(labels_task_mapping.keys()) < 0:
                raise ValueError('Invalid key value in labels_task_mapping. '
                                 f'The keys must be in  '
                                 f'[0, {len(dataset_labels) - 1}] '
                                 f'({labels_task_mapping.keys()})')

            if max(labels_task_mapping.values()) >= tasks_n \
                    or min(labels_per_tasks.values()) < 0:
                raise ValueError('Invalid value in labels_task_mapping. '
                                 f'The values must be in  [0, {tasks_n - 1}] '
                                 f'({labels_task_mapping.values()})')

        task_labels = {k: [] for k in range(tasks_n)}

        for label, task in labels_task_mapping.items():
            task_labels[task].append(label)
            assigned_labels.append(label)

        if any([len(v) > labels_per_tasks[t]
                for t, v in task_labels.items()]):
            s = {t: len(v) for t, v in task_labels.items()}
            raise ValueError(f'After populating the tasks '
                             f'using labels_task_mapping, some task has more '
                             f'assigned labels ({s}) than the limit '
                             f'imposed by labels_per_tasks '
                             f'({labels_per_tasks}).')

        if shuffle_labels:
            self.random_state.shuffle(dataset_labels)

        for label in [l for l in dataset_labels
                      if l not in assigned_labels]:
            eligible_tasks = [t for t, v in task_labels.items()
                              if len(v) < labels_per_tasks[t]]

            selected_task = eligible_tasks[0]
            task_labels[selected_task].append(label)

        labels_mapping = {}
        indexes = iter(range(len(dataset_labels)))

        for t, vals in task_labels.items():
            if remap_labels_across_task:
                map_dict = {v: next(indexes) for v in vals}
            elif remap_labels_in_task:
                map_dict = {v: i for i, v in enumerate(vals)}
            else:
                map_dict = {v: v for v in vals}

            labels_mapping[t] = map_dict

        self.tasks_n = tasks_n
        self.labels_mapping = labels_mapping
        self.lazy_initialization = lazy_initialization
        self.task_labels = task_labels

        self._tasks_generated = []

        # self.task_wise_labels = task_wise_labels
        # self.shuffle_labels = shuffle_labels
        #
        # self.parameters = transformation_parameters
        # self.task_n = tasks_n
        # self.transform_function = transform_factory
        #
        # self._t_counter = 0
        # self._current_task = 0
        #
        # self._transform_functions = []
        # self._tasks_generated = []

        if not lazy_initialization:
            for _ in range(tasks_n):
                self.generate_task()
        #         # self._tasks_generated.append(t)

    def __len__(self):
        return self.tasks_n

    def __getitem__(self, i: int):
        if i >= len(self._tasks_generated):
            raise ValueError(f'Attempting to get a non generated task from '
                             f'the lazy created stream of tasks (index: {i})'
                             f'. Generate the task or set '
                             f'lazy_initialization=False when '
                             f'instantiating this class.')

        return self._tasks_generated[i]

    def generate_task(self, **kwargs) -> Union[Task, None]:

        counter = len(self._tasks_generated)

        # if self.infinite_stream and callable(self.parameters):
        #     t_parameters = self.parameters(task=counter,
        #                                    random_state=self.random_state)
        # else:
        if counter == len(self):
            return None

        # t_parameters = self.parameters[counter]
        #
        # t = self.transform_function(t_parameters)

        labels = self.task_labels[counter]
        labels_map = self.labels_mapping[counter]

        dataset = get_dataset_subset_using_labels(self.dataset, labels=labels)
        task = Task(base_dataset=dataset,
                    labels_mapping=labels_map,
                    task_index=counter)

        # task = TransformerTask(base_dataset=self.dataset, transformer=t,
        #                        index=counter)

        self._tasks_generated.append(task)

        return task

    # def __next__(self):
    #     self._current_task = 0
    #     return self

    def __iter__(self):
        for i in range(self.tasks_n):
            if len(self._tasks_generated) > i:
                t = self._tasks_generated[i]
            else:
                t = self.generate_task()
                if t is None:
                    return

            yield t


class ImageRotationScenario(NCTransformingScenario):
    def __init__(self,
                 dataset: DatasetSplitsContainer,
                 tasks_n: int,
                 transformation_parameters: Union[List[any],
                                                  Callable[[Any], Any]],
                 # infinite_stream: bool = False,
                 random_state: Union[np.random.RandomState, int] = None,
                 lazy_initialization: bool = True,
                 labels_task_mapping: Dict[int, Union[int, list]] = None,
                 remap_labels_across_task: bool = False,

                 **kwargs):

        transform_function = self.get_rotation

        super().__init__(dataset=dataset,
                         tasks_n=tasks_n,
                         transform_factory=transform_function,
                         transformation_parameters=transformation_parameters,
                         # infinite_stream=infinite_stream,
                         random_state=random_state,
                         lazy_initialization=lazy_initialization,
                         labels_task_mapping=labels_task_mapping,
                         remap_labels_across_task=remap_labels_across_task,
                         **kwargs)

    def get_rotation(self, degree, **kwargs):
        return ImageRotation(degree)


class PixelsPermutationScenario(NCTransformingScenario):
    def __init__(self,
                 dataset: DatasetSplitsContainer,
                 tasks_n: int,
                 transformation_parameters: Union[List[any],
                                                  Callable[[Any], Any]],
                 # infinite_stream: bool = False,
                 random_state: Union[np.random.RandomState, int] = None,
                 lazy_initialization: bool = True,
                 labels_task_mapping: Dict[int, Union[int, list]] = None,
                 remap_labels_across_task: bool = False,

                 **kwargs):

        transform_factory = self.get_permutation

        super().__init__(dataset=dataset,
                         tasks_n=tasks_n,
                         transform_factory=transform_factory,
                         transformation_parameters=transformation_parameters,
                         # infinite_stream=infinite_stream,
                         random_state=random_state,
                         lazy_initialization=lazy_initialization,
                         labels_task_mapping=labels_task_mapping,
                         remap_labels_across_task=remap_labels_across_task,
                         **kwargs)

    def get_permutation(self, permutation, **kwargs):
        return PixelsPermutation(permutation)

