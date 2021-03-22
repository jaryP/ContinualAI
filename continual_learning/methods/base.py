from abc import ABC

import torch
from torch import nn

from continual_learning.solvers.base import Solver


class Container(object):
    def __init__(self):

        self.encoder = None
        self.solver = None
        self.other_models = torch.nn.ModuleDict()
        self.optimizer = None

        self.current_loss = None

        self.current_task = None
        self.current_batch = None
        self.current_epoch = None
        self.num_tasks = None

        self.others_parameters = dict()


class BaseMethod(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()

    def set_up(self):
        pass

    def preprocess_dataset(self, task, backbone: nn.Module, solver: Solver,  **kwargs):
        pass

    def get_parameters(self, task, backbone: nn.Module, solver: Solver,  **kwargs):
        parameters = []
        parameters.extend(backbone.parameters())
        return parameters

    def set_task(self, backbone, solver, task,  **kwargs):
        pass

    def before_evaluation(self, backbone, solver, task, *args, **kwargs):
        pass

    def on_epoch_starts(self, backbone, solver, task, *args, **kwargs):
        pass

    def on_epoch_ends(self, backbone, solver, task, *args, **kwargs):
        pass

    def on_task_starts(self, backbone, solver, task, *args, **kwargs):
        pass

    def on_task_ends(self, backbone, solver, task, *args, **kwargs):
        pass

    def on_batch_starts(self, backbone, solver, task, *args, **kwargs):
        pass

    def on_batch_ends(self, backbone, solver, task, *args, **kwargs):
        pass

    def after_optimization_step(self, backbone, solver, task, *args, **kwargs):
        pass

    def after_gradient_calculation(self, backbone, solver, task, *args, **kwargs):
        pass

    def before_gradient_calculation(self, backbone, solver, task,  *args, **kwargs):
        pass


class Naive(BaseMethod):
    def __init__(self):
        super().__init__()

    def set_task(self, **kwargs):
        pass
