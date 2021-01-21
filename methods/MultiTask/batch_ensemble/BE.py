from torch import nn

from methods import BaseMethod
from solvers.multi_task import MultiHeadsSolver
from .base import layer_to_masked, BElayer
from solvers.base import Solver


class BatchEnsemble(BaseMethod):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        layer_to_masked(backbone)
        self.model = backbone

    def get_parameters(self, current_task: int, network: nn.Module, solver: Solver):
        parameters = []

        if current_task == 0:
            parameters.extend(network.parameters())
        else:
            for n, m in network.named_modules():
                if isinstance(m, BElayer):
                    parameters.append(m.tasks_alpha[current_task])
                    parameters.append(m.tasks_gamma[current_task])

        if isinstance(solver, MultiHeadsSolver):
            parameters.extend(solver.heads[current_task].parameters())

        return parameters

    def set_task(self, t):
        for n, m in self.model.named_modules():
            if isinstance(m, BElayer):
                m.set_current_task(t)

    def on_task_starts(self, network: nn.Module, task_i: int, *args, **kwargs):
        for n, m in self.model.named_modules():
            if isinstance(m, BElayer):
                m.add_task()
                m.set_current_task(task_i)
                # parameters.append(m.tasks[current_task])
