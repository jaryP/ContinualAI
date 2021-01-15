from functools import reduce
from operator import mul
from typing import Union, Callable

import torch
from torch import nn

from solvers.base import Solver


class MultiTaskSolver(Solver):
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

    # def _get_head(self):
    #     def _weights_init(m):
    #         if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
    #             init.kaiming_normal_(m.weight)
    #
    #     # if isinstance(self.base_head, nn.Module):
    #     h = deepcopy(self.base_head)
    #     h.apply(_weights_init)
    #     # else:
    #     return h

    def base_topology(self, ind, outd):
        return nn.Sequential(*[nn.Linear(ind, ind),
                               nn.Dropout(0.25),
                               nn.ReLU(),
                               nn.Linear(ind, ind // 4),
                               nn.Dropout(0.25),
                               nn.ReLU(),
                               nn.Linear(ind // 4, outd)])

    def add_task(self, output_size):
        self._tasks.append(self.topology(self.input_di, output_size))

    @property
    def task(self):
        return self._task

    def task_parameters(self, t=None, recuse=True):
        if t is None:
            t = self.task
        th = self.heads[t]
        for param in th.parameters(recurse=recuse):
            yield param

    def parameters(self, t=None, recuse=True):
        for h in self.heads:
            for param in h.parameters(recurse=recuse):
                yield param

    @property
    def heads(self):
        return self._tasks

    @task.setter
    def task(self, value):
        if value > len(self._tasks) or 0 < value:
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