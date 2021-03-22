from copy import deepcopy

from torch import nn

from continual_learning.methods.task_incremental.multi_task.gg\
    import BaseMultiTaskGGMethod
from continual_learning.scenarios.tasks import SupervisedTask


class SingleTask(BaseMultiTaskGGMethod):
    def __init__(self, **kwargs):
        super().__init__()
        self.tasks_dict = {}

    def on_task_starts(self, backbone: nn.Module, task: SupervisedTask, *args, **kwargs):
        def reset(module):
            for layer in module.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
                elif len(list(layer.children())) > 0:
                    reset(layer)
        reset(backbone)

    def on_task_ends(self, backbone: nn.Module, task: SupervisedTask, *args, **kwargs):
        task_i = task.index
        sd = deepcopy(backbone.state_dict())
        self.tasks_dict[task_i] = sd
        # self.tasks_dict[task.index] = backbone.state_dict().copy()

    def set_task(self, backbone: nn.Module, task: SupervisedTask,  **kwargs):
        task_i = task.index
        if task_i in self.tasks_dict:
            backbone.load_state_dict(self.tasks_dict[task_i])
