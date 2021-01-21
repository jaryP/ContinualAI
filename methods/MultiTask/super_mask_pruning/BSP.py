from collections import defaultdict

import torch
from torch import nn

from methods import BaseMethod
from methods.MultiTask.super_mask_pruning.base.base import add_wrappers_to_model, mask_training, \
    remove_wrappers_from_model, get_masks_from_gradients, ForwardHook
from methods.MultiTask.super_mask_pruning.base.layer import EnsembleMaskedWrapper
from solvers.multi_task import MultiHeadsSolver
from solvers.base import Solver


class SuperMask(BaseMethod):
    def __init__(self, backbone: nn.Module, mask_parameters: dict = None, pruning_percentage=0.5, device='cpu'):
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

        self.model = backbone

        self.hooks = []
        self.tasks_masks = defaultdict(list)

    # def get_parameters(self, current_task: int, network: nn.Module, solver: Solver):
    #     parameters = []
    #
    #     if current_task == 0:
    #         parameters.extend(network.parameters())
    #     else:
    #         for n, m in network.named_modules():
    #             if isinstance(m, BElayer):
    #                 parameters.append(m.tasks_alpha[current_task])
    #                 parameters.append(m.tasks_gamma[current_task])
    #
    #     if isinstance(solver, MultiHeadsSolver):
    #         parameters.extend(solver.heads[current_task].parameters())
    #     return parameters
    #
    # def set_task(self, t):
    #     for n, m in self.model.named_modules():
    #         if isinstance(m, BElayer):
    #             m.set_current_task(t)

    def _get_mask_for_task(self, name: str, task: int):
        masks = self.tasks_masks[name][:task]
        if len(masks) == 0:
            return None
        _m = masks[0]
        for i in range(1, len(masks)):
            _m = torch.logical_or(_m, masks[i])
        return _m.float()

    def set_task(self, network: nn.Module, task_i:int):
        for h in self.hooks:
            h.remove()
        for name, module in network.named_modules():
            if name in self.tasks_masks:
                _m = self._get_mask_for_task(name=name, task=task_i+1)
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

    def on_task_starts(self, network: nn.Module, solver: MultiHeadsSolver, dataset, task_i: int, *args, **kwargs):
        add_wrappers_to_model(network, masks_params=self.mask_parameters)
        solver.task = task_i

        mask_training(model=network, epochs=10, dataset=dataset, solver=solver)
        grads = defaultdict(list)

        for i, x, y in dataset:
            x, y = x.to(self.device), y.to(self.device)

            pred = solver(network(x))

            loss = torch.nn.functional.cross_entropy(pred, y, reduction='mean')

            self.model.zero_grad()
            loss.backward(retain_graph=True)

            for name, module in self.model.named_modules():
                if isinstance(module, EnsembleMaskedWrapper):
                    g = torch.autograd.grad(loss, module.last_mask, retain_graph=True)[0]
                    grads[name].append(torch.abs(g).cpu())

        self.model.zero_grad()

        remove_wrappers_from_model(self.model)

        f = lambda x: torch.mean(x, 0)

        ens_grads = {name: f(torch.stack(gs, 0)).detach().cpu() for name, gs in grads.items()}
        # TODO: valutare se metter a zero i gradienti relativi ai task passati
        # _masks = self._get_mask_for_task(task_i)

        masks = get_masks_from_gradients(gradients=ens_grads, prune_percentage=self.pruning,
                                         global_pruning=True, device=self.device)

        for name, m in masks.items():
            self.tasks_masks[name].append(m)

        self.set_task(network, task_i)

    def after_back_propagation(self, network: nn.Module, current_task: int, *args, **kwargs):
        for name, module in network.named_modules():
            if hasattr(module, 'weight'):
                if module.weight.grad is not None:
                    grad = module.weight.grad
                    m = self._get_mask_for_task(name=name, task=current_task)
                    if m is not None:
                        m = m.unsqueeze(1)
                        m = 1 - m
                        grad *= m
