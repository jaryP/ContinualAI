import torch
from torch import nn

from methods import BaseMethod
from methods.MultiTask.piggyback.base import layer_to_masked, PiggyBackLayer, ForwardHook
from settings.supervised import ClassificationTask
from solvers.base import Solver
from solvers.multi_task import MultiHeadsSolver


class PiggyBack(BaseMethod):
    def __init__(self, backbone: nn.Module, threshold=5e-3, device='cpu'):
        super().__init__()
        layer_to_masked(backbone)
        self.model = backbone
        self.threshold = threshold
        self.device = device

        self.hooks = []
        self.task_masks = {}


    def get_parameters(self, task: ClassificationTask, backbone: nn.Module, solver: Solver):
        parameters = []
        current_task = task.index

        for n, m in backbone.named_modules():
            if isinstance(m, PiggyBackLayer):
                parameters.append(m.mask)

        if isinstance(solver, MultiHeadsSolver):
            parameters.extend(solver.heads[current_task].parameters())

        return parameters

    def set_task(self, backbone: nn.Module, task: ClassificationTask, **kwargs):
        task_i = task.index
        if task_i == 0 or task_i not in self.task_masks:
            return

        for h in self.hooks:
            h.remove()
        self.hooks = []

        for n, m in backbone.named_modules():
            if isinstance(m, PiggyBackLayer):
                h = ForwardHook(m.layer, self.task_masks[task_i][n])
                self.hooks.append(h)

    def on_task_starts(self, backbone: nn.Module, task: ClassificationTask, *args, **kwargs):
        for n, m in backbone.named_modules():
            if isinstance(m, PiggyBackLayer):
                m.add_task()

    def on_task_ends(self, backbone, solver, task, *args, **kwargs):
        mask_dict = {}
        for name, m in backbone.named_modules():
            if isinstance(m, PiggyBackLayer):
                mask = m.mask.data
                mask = torch.ge(mask, self.threshold).float().to(self.device)
                mask_dict[name] = mask
                m.mask = None
        self.task_masks[task.index] = mask_dict
