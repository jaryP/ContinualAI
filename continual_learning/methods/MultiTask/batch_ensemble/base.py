import torch
from torch import nn

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
        # rest = batch_size % self.single_task
        # m = batch_size // self.single_task
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
