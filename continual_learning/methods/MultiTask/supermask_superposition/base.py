import math
from typing import Union

import torch
from torch import autograd, nn


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        return (scores >= 0).float()

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g


class SupSupMaskWrapper(nn.Module):
    def __init__(self, layer: Union[nn.Linear, nn.Conv2d], **kwargs):
        super().__init__()

        self.layer = layer

        self.scores = nn.ParameterList()
        self.current_task = 0

        # self.num_tasks = num_tasks
        # self.scores = nn.ParameterList(
        #     [
        #         nn.Parameter(mask_init(self))
        #         for _ in range(num_tasks)
        #     ]
        # )

        # Keep weights untrained
        self.layer.weight.requires_grad = False
        # signed_constant(self)

    # @torch.no_grad()
    # def cache_masks(self):
    #     self.register_buffer(
    #         "stacked",
    #         torch.stack(
    #             [
    #                 GetSubnet.apply(self.scores[j])
    #                 for j in range(self.num_tasks)
    #             ]
    #         ),
    #     )

    def add_task(self):
        scores = torch.Tensor(self.layer.weight.size())
        nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
        self.scores.append(nn.Parameter(scores))

    def set_current_task(self, t):
        self.current_task = t

    def forward(self, x):
        # if self.task < 0:
        #     # Superimposed forward pass
        #     alpha_weights = self.alphas[: self.num_tasks_learned]
        #     idxs = (alpha_weights > 0).squeeze().view(self.num_tasks_learned)
        #     if len(idxs.shape) == 0:
        #         idxs = idxs.view(1)
        #     subnet = (
        #             alpha_weights[idxs]
        #             * self.stacked[: self.num_tasks_learned][idxs]
        #     ).sum(dim=0)
        # else:
        # Subnet forward pass (given task info in self.task)

        subnet = GetSubnet.apply(self.scores[self.current_task])
        w = self.layer.weight * subnet
        if isinstance(self.layer, nn.Linear):
            x = torch.nn.functional.linear(x, w, None)
        else:
            x = nn.functional.conv2d(x, w, None, stride=self.layer.stride, padding=self.layer.padding,
                                     dilation=self.layer.dilation, groups=self.layer.groups)
        return x

    def __repr__(self):
        return f"MultitaskMaskLinear({self.in_dims}, {self.out_dims})"
