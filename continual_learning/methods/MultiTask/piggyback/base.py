from typing import Union

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


class PiggyBackLayer(nn.Module):
    def __init__(self, layer: Union[nn.Linear, nn.Conv2d]):

        super().__init__()

        self.mask = None

        self.last_mask = None
        self.layer = layer

        # mask_dim = layer.weight.shape
        self.is_conv = isinstance(layer, nn.Conv2d)

    def add_task(self):
        t = torch.full(self.layer.weight.shape, 0.1)
        self.mask = nn.Parameter(t)

    def forward(self, x):

        w = self.layer.weight
        if self.mask is not None:
            w = w * self.mask

        if self.is_conv:
            o = nn.functional.conv2d(x, w, self.layer.bias, stride=self.layer.stride, padding=self.layer.padding,
                                     dilation=self.layer.dilation, groups=self.layer.groups)
        else:
            o = nn.functional.linear(x, w, self.layer.bias)

        return o

    @property
    def __repr__(self):
        return 'Supermask {} layer with distribution {}. ' \
               'Original layer: {} '.format('structured' if self.where != 'weights' else 'unstructured',
                                            self.task_distributions, self.layer.__repr__)
