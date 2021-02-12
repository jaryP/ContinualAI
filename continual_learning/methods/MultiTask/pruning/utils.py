from typing import Union

import numpy as np
import torch
from torch import nn

from continual_learning.scenarios.tasks import SupervisedTask


def get_accuracy(encoder: torch.nn.Module, solver: torch.nn.Module,
                 task: SupervisedTask, batch_size=64, device='cpu'):
    with torch.no_grad():
        encoder.eval()
        solver.eval()

        true_labels = []
        predicted_labels = []

        for j, x, y in task.get_iterator(batch_size):
            x = x.to(device)
            true_labels.extend(y.tolist())
            emb = encoder(x)
            a = solver(emb, task=task.index).cpu()
            predicted_labels.extend(a.max(dim=1)[1].tolist())
    true_labels, predicted_labels = np.asarray(true_labels), np.asarray(predicted_labels)
    eq = predicted_labels == true_labels
    accuracy = eq.sum() / len(eq)

    return accuracy


class PrunedLayer(nn.Module):
    def __init__(self, layer: Union[nn.Linear, nn.Conv2d]):

        super().__init__()

        self._use_mask = True
        self._eval_mask = None

        self._mask = None
        self.steps = 0

        self.last_mask = None
        self.layer = layer

        self.is_conv = isinstance(layer, nn.Conv2d)

    @property
    def weight(self):
        return self.layer.weight

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        self._mask = m

    def forward(self, x):

        mask = self.mask

        w = self.layer.weight
        if mask is not None:
            w = w * mask

        if self.is_conv:
            o = nn.functional.conv2d(x, w, None, stride=self.layer.stride, padding=self.layer.padding,
                                     dilation=self.layer.dilation, groups=self.layer.groups)
        else:
            o = nn.functional.linear(x, w, None)

        return o


class ForwardHook:
    def __init__(self, module: nn.Module, mask: torch.Tensor):
        mask = mask.unsqueeze(0)
        self.mask = mask
        self.hook = module.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, module_in, module_out):

        return module_out * self.mask

    def remove(self):
        self.hook.remove()

    def update_mask(self, mask):
        self.mask = mask


