from abc import ABC, abstractmethod

from torch import nn


class Solver(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def add_task(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self, task=None, recuse=True):
        raise NotImplementedError

    @abstractmethod
    def forward(self, **kwargs):
        raise NotImplementedError


# class SingleIncrementalTaskSolver(Solver):
#     pass
