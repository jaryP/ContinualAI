import numpy as np
import torch
from torch import nn


class ForwardHook:
    def __init__(self, module: nn.Module, mask: torch.Tensor):
        # mask = mask.unsqueeze(0)
        # if isinstance(module, nn.Conv2d):
        #     mask = mask.unsqueeze(-1).unsqueeze(-1)

        self.mask = mask
        self.hook = module.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, module_in, module_out):
        return module_out * self.mask

    def remove(self):
        self.hook.remove()

    def update_mask(self, mask):
        self.mask = mask


def get_masks_from_gradients(gradients,
                             prune_percentage,
                             global_pruning, past_masks=None,
                             hard_pruning: bool = True,
                             device='cpu'):
    masks = {}
    if past_masks is None:
        past_masks = {}

    if global_pruning:
        grads = []
        for name, g in gradients.items():
            if name in past_masks and hard_pruning:
                # g = g.view(-1).numpy()
                g = torch.masked_select(g.view(-1),
                                        past_masks[name].view(-1).bool())
                # m = past_masks[name]
                # gradients[name] *= m
                # m = m.view(-1).numpy()
                # g = g[m.astype(np.bool)]
                # grads.append(g.numpy())
            # else:
            if len(g) > 0:
                grads.append(g.view(-1).numpy())

        # stacked_grads = np.concatenate([gs.view(-1).numpy() for name, gs in gradients.items()])
        stacked_grads = np.concatenate(grads)
        grads_sum = np.sum(stacked_grads)
        stacked_grads_normalized = stacked_grads / grads_sum
        threshold = np.quantile(stacked_grads_normalized, q=prune_percentage)

        # masks = {name: torch.ge(gs / grads_sum, threshold).float().to(device)
        #          for name, gs in gradients.items()}
        for name, gs in gradients.items():
            # if hard_pruning and name in past_masks:
            #     gs = gs * past_masks[name]
            mask = torch.ge(gs / grads_sum, threshold).float().to(device)
            masks[name] = mask

    else:

        for name, gs in gradients.items():
            if name in past_masks and hard_pruning:
                masked = torch.masked_select(gs.view(-1),
                                             past_masks[name].view(-1).bool())
                if len(masked) == 0:
                    thres = 0
                else:
                    thres = torch.quantile(masked, prune_percentage)

            else:
                thres = torch.quantile(gs.view(-1), prune_percentage)
                # print(thres, gs.mean(), gs.std(), '\n', gs.view(-1))

            # if hard_pruning and name in past_masks:
            #     gs = gs * past_masks[name]
            mask = torch.ge(gs, thres).float().to(device)
            masks[name] = mask

        # masks = {name: torch.ge(gs, torch.quantile(gs, prune_percentage)).float().to(device)
        #          for name, gs in gradients.items()}

    # for name, mask in masks.items():
    #     mask = mask.squeeze()
    #     if mask.sum() == 0:
    #         max = torch.argmax(gradients[name])
    #         mask = torch.zeros_like(mask)
    #         mask[max] = 1.0
    #     masks[name] = mask

    if len(past_masks) > 0:
        for name, mask in masks.items():
            if name in past_masks:
                masks[name] = mask * past_masks[name]

    return masks


def mask_training(model, solver, epochs, dataset, device='cpu', parameters=None):
    model.to(device)
    bar = range(epochs)

    if parameters is None:
        parameters = [param for name, param in model.named_parameters() if 'distributions' in name]

    optim = torch.optim.Adam(parameters, lr=0.001)

    model.train()
    solver.train()

    for e in bar:
        losses = []
        for _, x, y in dataset:
            x, y = x.to(device), y.to(device)
            pred = solver(model(x))

            loss = torch.nn.functional.cross_entropy(pred, y, reduction='mean')
            losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
