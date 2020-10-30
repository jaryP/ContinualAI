import torch


class Container(object):
    def __init__(self):

        self.encoder = None
        self.solver = None
        self.other_models = torch.nn.ModuleDict()
        self.optimizer = None

        self.current_loss = None

        self.current_task = None
        self.current_batch = None
        self.current_epoch = None
        self.num_tasks = None

        self.others_parameters = dict()


class BaseMethod(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def on_epoch_starts(self, *args, **kwargs):
        pass

    def on_epoch_ends(self,  *args, **kwargs):
        pass

    def on_task_starts(self, *args, **kwargs):
        pass

    def on_task_ends(self, *args, **kwargs):
        pass

    def on_batch_starts(self, *args, **kwargs):
        pass

    def after_optimization_step(self,  *args, **kwargs):
        pass

    def after_back_propagation(self, *args, **kwargs):
        pass

    def before_gradient_calculation(self,  *args, **kwargs):
        pass

