import torch
from torch import nn

from continual_learning.methods.task_incremental.multi_task.gg\
    import BaseMultiTaskGGMethod
from continual_learning.methods.task_incremental.multi_task.gg.piggyback.base\
    import PiggyBackLayer, ForwardHook
from continual_learning.scenarios.tasks import SupervisedTask
from continual_learning.solvers.base import Solver


class PiggyBack(BaseMultiTaskGGMethod):
    def __init__(self, backbone: nn.Module, threshold=5e-3, device='cpu'):
        super().__init__()
        self.apply_wrapper_to_model(backbone)
        self.model = backbone
        self.threshold = threshold
        self.device = device

        self.hooks = []
        self.task_masks = {}

    def apply_wrapper_to_model(self, model):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                l = getattr(model, name)
                setattr(model, name, PiggyBackLayer(l))
        print(model)

    def get_parameters(self, task: SupervisedTask, backbone: nn.Module, solver: Solver):
        parameters = []

        for n, m in backbone.named_modules():
            if isinstance(m, PiggyBackLayer):
                parameters.append(m.mask)

        return parameters

    def set_task(self, backbone: nn.Module, task: SupervisedTask, **kwargs):
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

    def on_task_starts(self, backbone: nn.Module, task: SupervisedTask, *args, **kwargs):
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
