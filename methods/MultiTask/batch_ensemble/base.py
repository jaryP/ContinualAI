import torch
from torch import nn

from backbone_networks.resnet import ResNet
from backbone_networks.vgg import VGG


class BElayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

        self.is_conv = False
        if isinstance(layer, nn.Linear):
            self.input_dim = layer.in_features
            self.out_dim = layer.out_features
        else:
            self.is_conv = True
            self.input_dim = layer.in_channels
            self.out_dim = layer.out_channels

        self.tasks_alpha = nn.ParameterList()
        self.tasks_gamma = nn.ParameterList()
        self.current_task = 0

    def add_task(self):
        a = torch.ones(self.input_dim)
        b = torch.ones(self.out_dim)
        self.tasks_alpha.append(nn.Parameter(a, requires_grad=True))
        self.tasks_gamma.append(nn.Parameter(b, requires_grad=True))

    def set_current_task(self, t):
        self.current_task = t

    def forward(self, x, task=None):
        if task is None:
            task = self.current_task
        # batch_size = x.size(0)
        # rest = batch_size % self.ensemble
        # m = batch_size // self.ensemble
        # alpha = self.alpha.repeat(1, m).view(-1,  self.input_dim)
        # gamma = self.gamma.repeat(1, m).view(-1,  self.out_dim)
        # print(self.tasks[task].shape)
        alpha, gamma = self.tasks_alpha[task], self.tasks_gamma[task]
        # print(alpha.shape)
        # alpha = alpha.repeat(1, m).view(-1,  self.input_dim)
        # gamma = gamma.repeat(1, m).view(-1,  self.input_dim)
        # if rest > 0:
        #     alpha = torch.cat([alpha, alpha[:rest]], dim=0)
        #     gamma = torch.cat([gamma, gamma[:rest]], dim=0)

        if self.is_conv:
            alpha = alpha.unsqueeze(-1).unsqueeze(-1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)

        x = x * alpha
        output = self.layer(x) * gamma
        return output

# class BELinear(nn.Module):
#     def __init__(self, layer: nn.Linear, ensemble: int):
#         super().__init__()
#
#         self.ensemble = ensemble
#         alpha = torch.ones((ensemble, layer.in_features))
#         self.alpha = nn.Parameter(alpha, requires_grad=True)
#
#         gamma = torch.ones((ensemble, layer.out_features))
#         self.gamma = nn.Parameter(gamma, requires_grad=True)
#
#         self.input_dim = layer.in_features
#         self.out_dim = layer.out_features
#         self.layer = layer
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         rest = batch_size % self.ensemble
#         m = batch_size // self.ensemble
#
#         alpha = self.alpha.repeat(1, m).view(-1,  self.input_dim)
#         gamma = self.gamma.repeat(1, m).view(-1,  self.out_dim)
#
#         if rest > 0:
#             alpha = torch.cat([alpha, alpha[:rest]], dim=0)
#             gamma = torch.cat([gamma, gamma[:rest]], dim=0)
#
#         x = x * alpha
#         output = self.layer(x) * gamma
#         return output
#
#     def add_task(self):
#         pass
#
#     def __repr__(self):
#         return 'Batch ensemble. Original layer: {} '.format(self.layer.__repr__())
#
#
# class BEConv2D(nn.Module):
#     def __init__(self, layer: nn.Conv2d, ensemble: int):
#         super().__init__()
#
#         self.ensemble = ensemble
#         alpha = torch.ones((ensemble, layer.in_channels))
#         self.alpha = nn.Parameter(alpha, requires_grad=True)
#
#         gamma = torch.ones((ensemble, layer.out_channels))
#         self.gamma = nn.Parameter(gamma, requires_grad=True)
#
#         self.input_dim = layer.in_channels
#         self.out_dim = layer.out_channels
#
#         self.layer = layer
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         m = batch_size // self.ensemble
#         rest = batch_size % self.ensemble
#
#         alpha = self.alpha.repeat(1, m).view(-1, self.input_dim)
#         gamma = self.gamma.repeat(1, m).view(-1, self.out_dim)
#
#         if rest > 0:
#             alpha = torch.cat([alpha, alpha[:rest]], dim=0)
#             gamma = torch.cat([gamma, gamma[:rest]], dim=0)
#
#         alpha = alpha.unsqueeze(-1).unsqueeze(-1)
#         gamma = gamma.unsqueeze(-1).unsqueeze(-1)
#
#         x = x * alpha
#         output = self.layer(x) * gamma
#
#         return output
#
#     def __repr__(self):
#         return 'Batch ensemble. Original layer: {} '.format(self.layer.__repr__())


def layer_to_masked(module, ensemble=1):
    def apply_mask_sequential(s):
        for i, l in enumerate(s):
            if isinstance(l, (nn.Linear, nn.Conv2d)):
                s[i] = BElayer(l)

    if isinstance(module, nn.Sequential):
        apply_mask_sequential(module)
    elif isinstance(module, VGG):
        apply_mask_sequential(module.features)
        # apply_mask_sequential(module.classifier)
    elif isinstance(module, ResNet):
        module.conv1 = BElayer(module.conv1)
        # module.fc = BELinear(module.fc, ensemble=ensemble)
        for i in range(1, 4):
            apply_mask_sequential(getattr(module, 'layer{}'.format(i)))
    else:
        assert False