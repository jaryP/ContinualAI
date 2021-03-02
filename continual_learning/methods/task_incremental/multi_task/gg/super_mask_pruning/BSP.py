import functools
from collections import defaultdict

import torch
from torch import nn

from continual_learning.methods.task_incremental.multi_task.gg\
    import BaseMultiTaskGGMethod
from continual_learning.methods.task_incremental.multi_task.gg.\
    super_mask_pruning.base.utils import \
    mask_training, get_masks_from_gradients, ForwardHook
from continual_learning.methods.task_incremental.multi_task.gg.\
    super_mask_pruning.base.layer import EnsembleMaskedWrapper
from continual_learning.scenarios.tasks import SupervisedTask
from continual_learning.solvers.multi_task import MultiHeadsSolver


class SuperMask(BaseMultiTaskGGMethod):
    def __init__(self,
                 mask_epochs=5,
                 global_pruning=False,
                 hard_pruning: bool = True,
                 mask_parameters: dict = None,
                 pruning_percentage=0.5,
                 device='cpu'):
        super().__init__()
        if mask_parameters is None:
            mask_parameters = {'name': 'weights',
                               'initialization': {
                                   'name': 'normal',
                                   'mu': 0,
                                   'std': 1}
                               }
        self.device = device
        self.pruning = pruning_percentage
        self.mask_parameters = mask_parameters
        self.global_pruning = global_pruning
        self.mask_epochs = mask_epochs
        self.hard_pruning = hard_pruning

        self.hooks = []
        self.tasks_masks = defaultdict(list)

    def apply_wrapper_to_model(self,
                               model,
                               mask_params=None):
        for name, module in model.named_children():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                l = getattr(model, name)
                setattr(model, name,
                        EnsembleMaskedWrapper(l, where='output',
                                              masks_params=mask_params))
            else:
                self.apply_wrapper_to_model(module, mask_params=mask_params)

    def remove_wrappers_from_model(self, model):
        for name, module in model.named_children():
            if isinstance(module, EnsembleMaskedWrapper):
                l = getattr(model, name)
                setattr(model, name, l.layer)
            else:
                self.remove_wrappers_from_model(module)

    def reset_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _get_mask_for_task(self,
                           name: str,
                           task_i: int,
                           invert_masks: bool = False):
        if name not in self.tasks_masks or \
                len(self.tasks_masks) == 0 or\
                task_i == 0:
            return None

        masks = self.tasks_masks[name]
        t_masks = masks[:task_i]

        if len(t_masks) == 1:
            m = t_masks[0]
        else:
            m = functools.reduce(torch.logical_or, t_masks)

        if invert_masks:
            m = torch.logical_not(m)

        return m.float()

    def set_task(self,
                 backbone: nn.Module,
                 solver: MultiHeadsSolver,
                 task: SupervisedTask,
                 invert_masks=False,
                 **kwargs):

        task_i = task.index
        self.reset_hooks()
        for name, module in backbone.named_modules():
            _m = self._get_mask_for_task(name=name, task_i=task_i + 1,
                                         invert_masks=invert_masks)
            if _m is not None:
                hook = ForwardHook(module=module, mask=_m)
                self.hooks.append(hook)

    def _get_task_mask(self, task):
        r = defaultdict(list)
        for t, ms in self.masks.items():
            if t < task:
                for name, m in ms.items():
                    r[name].append(m)

        final_masks = {}
        for name, ms in r:
            m = None
            for _m in ms:
                if m is None:
                    m = _m
                else:
                    m = torch.logical_or(m, _m)
            final_masks[name] = m

        return final_masks

    def on_task_starts(self,
                       backbone: nn.Module,
                       solver: MultiHeadsSolver,
                       task: SupervisedTask,
                       **kwargs):

        # self.set_task(task_i=task_i, network=network, invert_masks=True)
        task_i = task.index
        dataset = task.get_iterator(64, shuffle=True)

        self.reset_hooks()

        self.apply_wrapper_to_model(backbone, mask_params=self.mask_parameters)

        solver.task = task_i

        backbone.to(self.device)

        mask_training(model=backbone, epochs=self.mask_epochs,
                      dataset=dataset, solver=solver, device=self.device)
        grads = defaultdict(list)

        for i, x, y in dataset:
            x, y = x.to(self.device), y.to(self.device)

            pred = solver(backbone(x))

            loss = torch.nn.functional.cross_entropy(pred, y, reduction='mean')

            backbone.zero_grad()
            loss.backward(retain_graph=True)

            for name, module in backbone.named_modules():
                if isinstance(module, EnsembleMaskedWrapper):
                    g = torch.autograd.grad(loss, module.last_mask, retain_graph=True)[0]
                    grads[name].append(torch.abs(g).detach().cpu())
                    # grads[name].append(torch.abs(g))

        backbone.zero_grad()

        self.remove_wrappers_from_model(backbone)

        f = lambda x: torch.mean(x, 0)

        ens_grads = {}
        past_masks = {}

        for name, gs in grads.items():
            g = f(torch.stack(gs, 0))
            _masks = self._get_mask_for_task(task_i=task_i, name=name,
                                             invert_masks=True)
            if _masks is not None:
                past_masks[name] = _masks
            ens_grads[name] = g

        # ens_grads = {name: f(torch.stack(gs, 0)).detach().cpu() for name, gs in grads.items()}
        # TODO: valutare se metter a zero i gradienti relativi ai task passati
        # _masks = self._get_mask_for_task(task_i)

        masks = get_masks_from_gradients(gradients=ens_grads,
                                         prune_percentage=self.pruning,
                                         global_pruning=self.global_pruning,
                                         past_masks=past_masks,
                                         hard_pruning=self.hard_pruning,
                                         device=self.device)

        # if len(past_masks) > 0:
        #     for name, mask in masks.items():
        #         if name in past_masks:
        #             masks[name] = mask * past_masks[name]

        #     gs = gs * past_masks[name]

        for name, m in masks.items():
            # ms = self.tasks_masks.get(name, [])
            # ms.append(m)
            # self.tasks_masks[name] = ms
            self.tasks_masks[name].append(m)

        self.set_task(backbone=backbone, task=task, solver=solver)

    def after_gradient_calculation(self,
                                   backbone: nn.Module,
                                   task: SupervisedTask,
                                   **kwargs):
        current_task = task.index
        for name, module in backbone.named_modules():
            m = self._get_mask_for_task(name=name, task_i=current_task, invert_masks=True)

            if m is not None:
                if hasattr(module, 'weight'):
                    weight = module.weight
                    if weight.grad is not None:
                        grad = weight.grad
                        grad *= torch.transpose(m, 0, 1)
                if hasattr(module, 'bias'):
                    bias = module.bias
                    if bias is not None and bias.grad is not None:
                        grad = bias.grad
                        grad *= m.view(-1)
