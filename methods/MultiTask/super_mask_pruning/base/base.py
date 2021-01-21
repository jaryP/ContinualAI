import numpy as np
import torch
from torch import nn

from backbone_networks.vgg import VGG
from methods.MultiTask.super_mask_pruning.base.layer import EnsembleMaskedWrapper
from solvers.multi_task import MultiHeadsSolver


class ForwardHook:
    def __init__(self, module: nn.Module, mask: torch.Tensor):
        self.mask = mask
        self.hook = module.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, module_in, module_out):
        return module_out * self.mask

    def remove(self):
        self.hook.remove()


def add_wrappers_to_model(module, masks_params=None):
    where = 'output'

    def apply_mask_sequential(s, skip_last):
        for i, l in enumerate(s):
            if isinstance(l, (nn.Linear, nn.Conv2d)):
                if skip_last and i == len(s) - 1:
                    continue
                s[i] = EnsembleMaskedWrapper(l, where=where, masks_params=masks_params)
            # elif isinstance(l, BasicBlock):
            #     s[i] = ResNetBlockWrapper(l, masks_params=masks_params,
            #                               ensemble=ensemble, batch_ensemble=batch_ensemble)

    if isinstance(module, nn.Sequential):
        apply_mask_sequential(module, skip_last=True)
    elif isinstance(module, VGG):
        apply_mask_sequential(module.features, skip_last=False)
        apply_mask_sequential(module.classifier, skip_last=True)
    # elif isinstance(module, ResNet):
    #     module.conv1 = EnsembleMaskedWrapper(module.conv1, masks_params=masks_params, where='output',
    #                            ensemble=ensemble, batch_ensemble=batch_ensemble)
    #     for i in range(1, 4):
    #         apply_mask_sequential(getattr(module, 'layer{}'.format(i)), skip_last=True)
    #
    #     module.fc = wrapper(module.fc, masks_params=masks_params, where='output',
    #                         ensemble=ensemble, batch_ensemble=batch_ensemble)
    else:
        assert False


def remove_wrappers_from_model(model):
    def remove_masked_layer(s):
        for i, l in enumerate(s):
            if isinstance(l, (EnsembleMaskedWrapper, EnsembleMaskedWrapper)):
                s[i] = l.layer
            # if isinstance(l, ResNetBlockWrapper):
            #     s[i] = l.block
            #     if isinstance(l.block.shortcut, nn.Sequential):
            #         if len(l.block.shortcut) > 0:
            #          # if l.block.downsample is not None:
            #             s[i].shortcut[0] = l.block.shortcut[0].layer

    if isinstance(model, nn.Sequential):
        remove_masked_layer(model)
    elif isinstance(model, VGG):
        remove_masked_layer(model.features)
        remove_masked_layer(model.classifier)
    # elif isinstance(model, ResNet):
    #     model.conv1 = model.conv1.layer
    #     if isinstance(model.fc, (EnsembleMaskedWrapper, BatchEnsembleMaskedWrapper)):
    #         model.fc = model.fc.layer
    #     for i in range(1, 4):
    #         remove_masked_layer(getattr(model, 'layer{}'.format(i)))
    else:
        assert False

    return model


def get_masks_from_gradients(gradients, prune_percentage, global_pruning, device='cpu'):
    if global_pruning:
        stacked_grads = np.concatenate([gs.view(-1).numpy() for name, gs in gradients.items()])
        grads_sum = np.sum(stacked_grads)
        stacked_grads = stacked_grads / grads_sum

        threshold = np.quantile(stacked_grads, q=prune_percentage)

        masks = {name: torch.ge(gs / grads_sum, threshold).float().to(device)
                 for name, gs in gradients.items()}
    else:
        masks = {name: torch.ge(gs, torch.quantile(gs, prune_percentage)).float()
                 for name, gs in gradients.items()}

    for name, mask in masks.items():
        mask = mask.squeeze()
        if mask.sum() == 0:
            max = torch.argmax(gradients[name])
            mask = torch.zeros_like(mask)
            mask[max] = 1.0
        masks[name] = mask

    return masks


def mask_training(model, solver, epochs, dataset, device='cpu', parameters=None):
    model.to(device)
    bar = range(epochs)

    if parameters is None:
        parameters = [param for name, param in model.named_parameters() if 'distributions' in name]

    optim = torch.optim.Adam(parameters, lr=0.001)

    for e in bar:
        losses = []
        model.train()

        for _, x, y in dataset:
            x, y = x.to(device), y.to(device)
            pred = solver(model(x))

            loss = torch.nn.functional.cross_entropy(pred, y, reduction='mean')
            losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()