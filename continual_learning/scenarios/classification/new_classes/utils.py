from typing import Union, Callable, Any, List, Dict

import numpy as np

from continual_learning.datasets.base import SupervisedDataset, \
    UnsupervisedDataset
from continual_learning.scenarios.base import TasksGenerator, AbstractTask
from continual_learning.scenarios.classification.utils import \
    get_dataset_subset_using_labels
from continual_learning.scenarios.tasks import SupervisedTransformerTask


class NCTransformingScenario(TasksGenerator):
    # TODO: Implement infinite stream

    def __init__(self,
                 *,
                 tasks_n: int,
                 dataset: Union[SupervisedDataset, UnsupervisedDataset],

                 transform_factory: Callable[[Any], Callable],
                 transformation_parameters: Union[List[any],
                                                  Callable[[Any], Any]],

                 # infinite_stream: bool = False,

                 random_state: Union[np.random.RandomState, int] = None,
                 lazy_initialization: bool = True,

                 # labels_per_tasks: Dict[int, int] = None,
                 labels_task_mapping: Dict[int, Union[int, list]] = None,

                 remap_labels_across_task: bool = False,

                 **kwargs):

        super().__init__(dataset,
                         random_state=random_state,
                         **kwargs)

        self.dataset_labels, dataset_labels = np.asarray(dataset.labels)

        if tasks_n < 0:
            raise ValueError(f'Argument tasks_n must be '
                             f'greater than 0 '
                             f'{type(tasks_n)}')

        if callable(transformation_parameters):
            transformation_parameters = [
                transformation_parameters(task=task,
                                          random_state=
                                          self.random_state)
                for task in range(tasks_n)]

        if labels_task_mapping is not None:
            # TODO: compress this code
            for l in dataset_labels:
                if l in labels_task_mapping:
                    v = labels_task_mapping[l]
                    if isinstance(v, int):
                        labels_task_mapping[l] = [v]
                else:
                    labels_task_mapping[l] = list(range(tasks_n))
        else:
            labels_task_mapping = {l: list(range(tasks_n))
                                   for l in dataset_labels}

        # if labels_per_tasks is None:
        #     labels_per_tasks = {task: len(dataset_labels)
        #                         for task in range(tasks_n)}

        # if remap_labels_in_task and remap_labels_across_task:
        #     raise ValueError('Both remap_labels_in_task and '
        #                      'remap_labels_across_task are set to True '
        #                      'but are mutually exclusive. '
        #                      'Please set at least one to False.')
        #
        # if not remap_labels_in_task and not remap_labels_across_task:
        #     raise ValueError('Both remap_labels_in_task and '
        #                      'remap_labels_across_task are set to False. '
        #                      'Please specify how to set.')

        # if max(labels_per_tasks.keys()) >= tasks_n or min(
        #         labels_per_tasks.keys()) < 0:
        #     raise ValueError('Invalid key value in labels_per_tasks. '
        #                      f'The keys must be in  [0, {tasks_n - 1}] '
        #                      f'({labels_per_tasks.keys()})')
        #
        # if min(labels_per_tasks.values()) < 0:
        #     raise ValueError('Invalid value in labels_per_tasks. '
        #                      f'The values must be > 0'
        #                      f'({labels_per_tasks.keys()})')
        #
        # if max(labels_per_tasks.values()) > 0:
        #     raise ValueError('Invalid value in labels_per_tasks. '
        #                      f'The values must be <= {len(dataset_labels)}'
        #                      f'({labels_per_tasks.keys()})')

        if not all(label in dataset_labels
                   for label, task in labels_task_mapping.items()):
            raise ValueError(f'Some labels in labels_task_mapping are not '
                             f'present in the dataset. '
                             f'Dataset labels: {dataset_labels}, '
                             f'given labels: {labels_task_mapping}')

        if max(labels_task_mapping.keys()) > len(dataset_labels) - 1 \
                or min(labels_task_mapping.keys()) < 0:
            raise ValueError('Invalid key value in labels_task_mapping. '
                             f'The keys must be in  '
                             f'[0, {len(dataset_labels) - 1}] '
                             f'({labels_task_mapping.keys()})')

        flat_list = [item for sublist in labels_task_mapping.values()
                     for item in sublist]

        if max(flat_list) >= tasks_n or min(flat_list) < 0:
            raise ValueError('Invalid value in labels_task_mapping. '
                             f'The values must be in  [0, {tasks_n - 1}] '
                             f'({labels_task_mapping.values()})')

        task_labels = {k: [] for k in range(tasks_n)}

        for label, tasks in labels_task_mapping.items():
            for t in tasks:
                task_labels[t].extend(label)

        labels_mapping = {}
        indexes = iter(range(len(dataset_labels)))

        for t, vals in task_labels.items():
            if remap_labels_across_task:
                map_dict = {v: next(indexes) for v in vals}
            else:
                map_dict = {v: v for v in vals}

            #     map_dict = {v: i for i, v in enumerate(vals)}

            labels_mapping[t] = map_dict

        if not remap_labels_across_task:
            lens = {k: len(v) for k, v in task_labels.items()}
            if any([l == 1 for l in lens.values()]):
                raise ValueError('Some task has only one class but '
                                 'remap_labels_across_task=False. '
                                 'Please populate the task by setting'
                                 'labels_task_mapping or '
                                 'set remap_labels_across_task=True'
                                 f'({lens}).')

        # if any([len(v) != labels_per_tasks[t]
        #         for t, v in task_labels.items()]):
        #     s = {t: len(v) for t, v in task_labels.items()}
        #     raise ValueError(f'After populating the tasks '
        #                      f'using labels_task_mapping, some task has more '
        #                      f'assigned labels ({s}) than the limit '
        #                      f'imposed by labels_per_tasks '
        #                      f'({labels_per_tasks}).')

        self.labels_mapping = labels_mapping
        self.task_labels = task_labels

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

    def __len__(self):
        # if self.infinite_stream:
        #     return np.inf
        # else:
        return self.task_n

    def __getitem__(self, i: int) -> AbstractTask:
        if i > len(self._tasks_generated):
            # if self.infinite_stream:
            #     raise ValueError(f'Attempting to get a non generated task from '
            #                      f'an infinite stream of tasks '
            #                      f'(generated tasks: '
            #                      f'({len(self._tasks_generated) - 1})), '
            #                      f'index: {i})')
            # else:
            raise ValueError(f'Attempting to get a non generated task from '
                             f'an lazy created stream of tasks (index: {i})'
                             f'. Generate the task or set '
                             f'lazy_initialization=False when '
                             f'instantiating this class.')

        return self._tasks_generated[i]

    def generate_task(self, **kwargs) -> Union[AbstractTask, None]:

        counter = len(self._tasks_generated)

        # if self.infinite_stream and callable(self.parameters):
        #     t_parameters = self.parameters(task=counter,
        #                                    random_state=self.random_state)
        # else:
        if counter == self.task_n:
            return None

        t_parameters = self.parameters[counter]
        labels_map = self.labels_mapping[counter]
        labels = self.task_labels[counter]

        if len(labels) == len(self.dataset.labels):
            dataset = self.dataset
        else:
            dataset = get_dataset_subset_using_labels(self.dataset,
                                                      labels=labels)

        t = self.transform_function(t_parameters)

        task = SupervisedTransformerTask(base_dataset=dataset,
                                         transformer=t,
                                         task_index=counter,
                                         labels_mapping=labels_map)

        self._tasks_generated.append(task)

        return task

    def __next__(self):
        self._current_task = 0
        return self
    
    def __iter__(self):
        while True:
            # if not self.infinite_stream:
            if self._current_task >= self.task_n or \
                    self._current_task >= len(self._tasks_generated):
                return

            if len(self._tasks_generated) >= self._current_task:
                t = self._tasks_generated[self._current_task]
            else:
                t = self.generate_task()

                if t is None:
                    return

            self._current_task += 1
            yield t