from typing import Union

import torch
from torch import nn

from continual_learning.methods.task_incremental.multi_task.gg.\
    super_mask_pruning.base import \
    TrainableBeta, TrainableLaplace, TrainableNormal, TrainableWeights, \
    TrainableExponential, TrainableGamma


class EnsembleMaskedWrapper(nn.Module):
    def __init__(self, layer: Union[nn.Linear, nn.Conv2d], where: str, masks_params: dict, t: int = 1):

        super().__init__()

        self._use_mask = True
        self._eval_mask = None

        self.where = where.lower()
        self.masks = []
        self.steps = 0

        self.last_mask = None
        self.layer = layer

        mask_dim = layer.weight.shape
        self.is_conv = isinstance(layer, nn.Conv2d)

        where = where.lower()
        # if not where == 'weights':

        if where == 'output':
            mask_dim = (1, mask_dim[0])
        elif where == 'inuput':
            mask_dim = (1, mask_dim[1])
        else:
            assert False, 'The following types are allowed: output, input and weights. {} given'.format(where)

        if self.is_conv:
            mask_dim = mask_dim + (1, 1)

        self.task_masks = []

        self.task_distributions = None

        self.t = t
        self.masks_params = masks_params
        self.mask_dim = mask_dim

        if masks_params['name'] == 'beta':
            distribution = TrainableBeta(mask_dim, t=t, initialization=masks_params['initialization'])
        elif masks_params['name'] == 'laplace':
            distribution = TrainableLaplace(mask_dim, t=t, initialization=masks_params['initialization'])
        elif masks_params['name'] == 'normal':
            distribution = TrainableNormal(mask_dim, t=t, initialization=masks_params['initialization'])
        elif masks_params['name'] == 'weights':
            distribution = TrainableWeights(mask_dim, initialization=masks_params['initialization'])
        elif masks_params['name'] == 'exponential':
            distribution = TrainableExponential(mask_dim, t=t, initialization=masks_params['initialization'])
        elif masks_params['name'] == 'gamma':
            distribution = TrainableGamma(mask_dim, t=t, initialization=masks_params['initialization'])
        else:
            assert False

        distribution.to(layer.weight.device)
        self.task_distributions = distribution

    def set_task(self, task):
        pass

    def add_task(self):
        if self.masks_params['name'] == 'beta':
            distribution = TrainableBeta(self.mask_dim, t=self.t, initialization=self.masks_params['initialization'])
        elif self.masks_params['name'] == 'laplace':
            distribution = TrainableLaplace(self.mask_dim, t=self.t, initialization=self.masks_params['initialization'])
        elif self.masks_params['name'] == 'normal':
            distribution = TrainableNormal(self.mask_dim, t=self.t, initialization=self.masks_params['initialization'])
        elif self.masks_params['name'] == 'weights':
            distribution = TrainableWeights(self.mask_dim, initialization=self.masks_params['initialization'])
        elif self.masks_params['name'] == 'exponential':
            distribution = TrainableExponential(self.mask_dim, t=self.t, initialization=self.masks_params['initialization'])
        elif self.masks_params['name'] == 'gamma':
            distribution = TrainableGamma(self.mask_dim, t=self.t, initialization=self.masks_params['initialization'])
        else:
            assert False

        self.task_distributions.append(distribution)

    def set_distribution(self, v):

        if self.single_distribution:
            self._current_distribution = 0
        else:
            assert v <= len(self.task_distributions) or v == 'all'
            if v == 'all':
                v = -1
            self._current_distribution = v

    @property
    def apply_mask(self):
        return self._use_mask

    @apply_mask.setter
    def apply_mask(self, v: bool):
        self._use_mask = v

    def posterior(self):
        return self.distribution.posterior

    @property
    def mask(self):

        if not self.apply_mask:
            return 1

        if self._current_distribution < 0:
            masks = [d(reduce=True) for d in self.task_distributions]
            m = torch.mean(torch.stack(masks), 0)
        else:
            m = self.task_distributions[self._current_distribution](reduce=True)

        return m

    def eval(self):
        self._eval_mask = self.mask
        return self.train(False)

    def train(self, mode=True):
        self._eval_mask = None
        return super().train(mode)

    def forward(self, x):

        mask = self.task_distributions(reduce=True)
        if self.where == 'input':
            # mask = self.mask
            x = mask * x

        # if self.where == 'weights':
        #     w = self.mask * self.layer.weight
        # else:

        w = self.layer.weight

        if self.is_conv:
            o = nn.functional.conv2d(x, w, self.layer.bias, stride=self.layer.stride, padding=self.layer.padding,
                                     dilation=self.layer.dilation, groups=self.layer.groups)
        else:
            o = nn.functional.linear(x, w, self.layer.bias)

        if self.where == 'output':
            # mask = self.mask
            o = o * mask

        # mask.requires_grad = True
        # mask = Parameter(mask, requires_grad=True)
        # mask.retain_grad()
        # # save last mask to retrieve the gradient
        self.last_mask = mask

        return o

    def __repr__(self):
        return 'Supermask {} layer with distribution {}. ' \
               'Original layer: {} '.format('structured' if self.where != 'weights' else 'unstructured',
                                            self.task_distributions, self.layer.__repr__())

