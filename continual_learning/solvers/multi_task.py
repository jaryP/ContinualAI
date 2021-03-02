from functools import reduce
from operator import mul
from typing import Union, Callable

import torch
from torch import nn

from continual_learning.solvers.base import Solver


class MultiHeadsSolver(Solver):
    def __init__(self, input_dim: Union[int, tuple], topology: Callable[[int, int], nn.Module] = None):
        super().__init__()

        if topology is None:
            topology = self.base_topology

        if hasattr(input_dim, '__len__') and len(input_dim) > 1:
            input_dim = reduce(mul, input_dim, 1)
            self.flat_input = True
        else:
            self.flat_input = False

        self._tasks = nn.ModuleList()
        self.input_dim = input_dim

        self.classification_layer = None

        self.topology = topology
        self._task = 0

    def base_topology(self, ind, outd):
        return nn.Sequential(*[nn.Linear(ind, ind),
                               nn.Dropout(0.25),
                               nn.ReLU(),
                               nn.Linear(ind, ind // 4),
                               nn.Dropout(0.25),
                               nn.ReLU(),
                               nn.Linear(ind // 4, outd)])

    def add_task(self, output_size):
        self._tasks.append(self.topology(self.input_dim, output_size))

    @property
    def task(self):
        return self._task

    def get_parameters(self, task=None, recuse=True):
        if task is None:
            task = self.task
        th = self.heads[task]
        for param in th.parameters(recurse=recuse):
            yield param

    @property
    def heads(self):
        return self._tasks

    @task.setter
    def task(self, value):
        if value > len(self._tasks) or value < 0:
            raise ValueError()
        self._task = value

    def forward(self, x, task=None):

        if task is not None:
            _t = task
        else:
            _t = self.task

        if self.flat_input:
            x = torch.flatten(x, 1)

        x = self.heads[_t](x)

        return x
