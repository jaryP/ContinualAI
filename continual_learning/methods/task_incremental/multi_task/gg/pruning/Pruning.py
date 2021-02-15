import functools

import numpy as np
import torch
from torch import nn

from continual_learning.methods.task_incremental.multi_task.gg\
    import BaseMultiTaskGGMethod
from continual_learning.methods.task_incremental.multi_task.gg.pruning.utils \
    import PrunedLayer, get_accuracy
from continual_learning.scenarios.tasks import SupervisedTask
from continual_learning.solvers.base import Solver


class Pruning(BaseMultiTaskGGMethod):
    """
    @misc{golkar2019continual,
        title={Continual Learning via Neural Pruning},
        author={Siavash Golkar and Michael Kagan and Kyunghyun Cho},
        year={2019},
        eprint={1903.04476},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }
    """

    def __init__(self, backbone: nn.Module, alpha=1e-5, tolerance=0.05, steps=100, device='cpu', **kwargs):
        super().__init__()
        self.apply_wrapper_to_model(backbone)
        self.alpha = alpha
        self.tolerance = tolerance
        self.steps = steps
        self.device = device

        self.tasks_masks = {}

    # def get_parameters(self, task, backbone: nn.Module, solver: Solver, **kwargs):
    #     current_task = task.index
    #
    #     if current_task > 0:
    #         for n, m in backbone.named_modules():
    #             if isinstance(m, BatchNorm2d):
    #                 m.track_running_stats = False
    #
    #     return super().get_parameters(task, backbone, solver)

    def apply_wrapper_to_model(self, model):
        # for name, module in model.named_modules():
        #     if isinstance(module, (nn.Linear, nn.Conv2d)):
        #         l = getattr(model, name)
        #         setattr(model, name, PrunedLayer(l))
        # print(model)
        for name, module in model.named_children():

            if isinstance(module, (nn.Linear, nn.Conv2d)):
                l = getattr(model, name)
                setattr(model, name, PrunedLayer(l))
            self.apply_wrapper_to_model(module)

    def set_task(self, backbone, solver, task: SupervisedTask,  **kwargs):
        if task.index in self.tasks_masks:
            for name, module in backbone.named_modules():
                if isinstance(module, PrunedLayer):
                    module.mask = self.tasks_masks[task.index][name]

    def _get_past_masks(self, backbone, i):
        masks = {}
        if i > 0:
            for name, module in backbone.named_modules():
                if isinstance(module, PrunedLayer):
                    all_m = [self.tasks_masks[j][name] for j in range(i)]
                    _m = functools.reduce(torch.logical_or, all_m)
                    masks[name] = _m
        return masks

    def before_gradient_calculation(self, backbone: nn.Module, loss: torch.Tensor, *args, **kwargs):
        l1 = 0
        for module in backbone.modules():
            if hasattr(module, 'weight'):
                w = module.weight
                norm = torch.norm(w, 1)
                l1 += norm
        l1 *= self.alpha

        loss += l1

    def after_gradient_calculation(self, backbone: nn.Module, task: SupervisedTask, solver: Solver, **kwargs):
        if task.index > 0:
            past_masks = self._get_past_masks(backbone, task.index)
            for name, module in backbone.named_modules():
                if isinstance(module, PrunedLayer):
                    module.weight.grad *= torch.logical_not(past_masks[name])

    def on_task_ends(self, backbone: nn.Module, task: SupervisedTask, solver: Solver, *args, **kwargs):
        task.dev()
        if len(task) == 0:
            task.train()

        final_accuracy = get_accuracy(backbone, solver, task, device=self.device)
        min_accuracy = final_accuracy - self.tolerance

        min_w = min(p.abs().min().item() for p in backbone.parameters() if p.requires_grad)
        max_w = min(p.abs().max().item() for p in backbone.parameters() if p.requires_grad)
        theres = np.linspace(min_w, max_w, self.steps)[1:-2]

        masks = {}
        past_masks = self._get_past_masks(backbone, task.index)

        for threshold in theres:
            _masks = {}
            for name, module in backbone.named_modules():
                if isinstance(module, PrunedLayer):
                    w = module.weight.abs()
                    m = torch.ge(w, threshold)
                    if task.index > 0:
                        m = torch.logical_or(m, torch.logical_not(past_masks[name]))
                    module.mask = m
                    _masks[name] = m

            accuracy = get_accuracy(backbone, solver, task, device=self.device)

            if accuracy > min_accuracy or len(masks) == 0:
                masks = _masks
            else:
                break

        self.tasks_masks[task.index] = masks


