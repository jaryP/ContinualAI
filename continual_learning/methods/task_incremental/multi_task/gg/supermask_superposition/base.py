import math
from typing import Union

import torch
from torch import autograd, nn


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, pruning_percentage):
        threshold = torch.quantile(scores.abs(), q=pruning_percentage)
        mask = torch.ge(scores, threshold).float()
        return mask

    @staticmethod
    def backward(ctx, g):
        return g, None


class SupSupMaskWrapper(nn.Module):
    def __init__(self,
                 layer: Union[nn.Linear, nn.Conv2d],
                 pruning_percentage: float,
                 **kwargs):

        super().__init__()

        self.layer = layer
        self.pruning_percentage = pruning_percentage
        self.scores = nn.ParameterList()
        self.current_task = 0

        self.layer.weight.requires_grad = False

    def add_task(self):
        scores = torch.Tensor(self.layer.weight.size())
        nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
        self.scores.append(nn.Parameter(scores))

    def set_current_task(self, t):
        self.current_task = t

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores[self.current_task],
                                 self.pruning_percentage)
        w = self.layer.weight * subnet

        if isinstance(self.layer, nn.Linear):
            x = torch.nn.functional.linear(x,
                                           w,
                                           None)
        else:
            x = nn.functional.conv2d(x,
                                     w,
                                     None,
                                     stride=self.layer.stride,
                                     padding=self.layer.padding,
                                     dilation=self.layer.dilation,
                                     groups=self.layer.groups)
        return x

    def __repr__(self):
        return f"MultitaskMaskLinear({self.in_dims}, {self.out_dims})"
