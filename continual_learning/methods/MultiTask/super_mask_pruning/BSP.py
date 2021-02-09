import functools
from collections import defaultdict

import torch
from torch import nn

from continual_learning.methods.MultiTask.piggyback.base import ForwardHook
from continual_learning.methods.MultiTask.super_mask_pruning.base.base import mask_training, get_masks_from_gradients
from continual_learning.methods.MultiTask.super_mask_pruning.base.layer import EnsembleMaskedWrapper
from continual_learning.methods.base import BaseMethod
from continual_learning.scenarios.supervised import ClassificationTask
from continual_learning.solvers.multi_task import MultiHeadsSolver


class SuperMask(BaseMethod):
    def __init__(self, mask_epochs=5, global_pruning=False,
                 mask_parameters: dict = None, pruning_percentage=0.5, device='cpu'):
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
        # self.model = backbone

        self.hooks = []
        self.tasks_masks = defaultdict(list)

    def apply_wrapper_to_model(self, model, mask_params=None):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                l = getattr(model, name)
                setattr(model, name, EnsembleMaskedWrapper(l, where='output', masks_params=mask_params))
        print(model)

    def remove_wrappers_from_model(self, model):
        for name, module in model.named_modules():
            if isinstance(module, EnsembleMaskedWrapper):
                l = getattr(model, name)
                setattr(model, name, l.layer)
        print(model)

    def reset_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _get_mask_for_task(self, name: str, task: int, invert_masks: bool = False):
        masks = self.tasks_masks[name][:task]

        if len(masks) == 0:
            return None
        if len(masks) == 1:
            _m = masks[0]
        else:
            _m = functools.reduce(torch.logical_or, masks)

        if invert_masks:
            _m = torch.logical_not(_m)

        return _m.float()

    def set_task(self, backbone: nn.Module, solver: MultiHeadsSolver, task: ClassificationTask, invert_masks=False,
                 *args, **kwargs):
        task_i = task.index
        self.reset_hooks()
        for name, module in backbone.named_modules():
            if name in self.tasks_masks:
                _m = self._get_mask_for_task(name=name, task=task_i + 1, invert_masks=invert_masks)
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

    def on_task_starts(self, backbone: nn.Module, solver: MultiHeadsSolver, task: ClassificationTask, *args, **kwargs):
        # def backbone=backbone, solver=solver, task=t
        # self.set_task(task_i=task_i, network=network, invert_masks=True)
        task_i = task.index
        dataset = task.get_iterator(64, shuffle=True)

        self.reset_hooks()
        
        self.apply_wrapper_to_model(backbone, mask_params=self.mask_parameters)
        solver.task = task_i

        mask_training(model=backbone, epochs=self.mask_epochs, dataset=dataset, solver=solver, device=self.device)
        grads = defaultdict(list)
        is_conv = set()

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
                    if isinstance(module, nn.Conv2d):
                        is_conv.add(name)

        backbone.zero_grad()

        self.remove_wrappers_from_model(backbone)

        f = lambda x: torch.mean(x, 0)

        ens_grads = {}
        old_grads = {}

        for name, gs in grads.items():
            g = f(torch.stack(gs, 0))
            _masks = self._get_mask_for_task(task=task_i, name=name, invert_masks=True)
            if _masks is not None:
                _masks = _masks.unsqueeze(0)
                # if name in is_conv:
                _masks = _masks.unsqueeze(-1).unsqueeze(-1).cpu()
                # m = m.unsqueeze(1)
                # _masks = 1 - _masks.cpu()
                old_grads[name] = _masks
                # g *= _masks
            ens_grads[name] = g

        # ens_grads = {name: f(torch.stack(gs, 0)).detach().cpu() for name, gs in grads.items()}
        # TODO: valutare se metter a zero i gradienti relativi ai task passati
        # _masks = self._get_mask_for_task(task_i)

        masks = get_masks_from_gradients(gradients=ens_grads, prune_percentage=self.pruning,
                                         global_pruning=self.global_pruning, past_masks=old_grads,
                                         device=self.device)

        for name, m in masks.items():
            self.tasks_masks[name].append(m)

        self.set_task(backbone=backbone, task=task, solver=solver)

    def after_gradient_calculation(self, backbone: nn.Module, task: ClassificationTask, *args, **kwargs):
        current_task = task.index
        for name, module in backbone.named_modules():
            if hasattr(module, 'weight'):
                if module.weight.grad is not None:
                    grad = module.weight.grad
                    m = self._get_mask_for_task(name=name, task=current_task, invert_masks=True)
                    if m is not None:
                        m = m.unsqueeze(1)
                        if isinstance(module, nn.Conv2d):
                            m = m.unsqueeze(-1).unsqueeze(-1)
                        # m = m.unsqueeze(1)
                        # m = 1 - m
                        grad *= m
