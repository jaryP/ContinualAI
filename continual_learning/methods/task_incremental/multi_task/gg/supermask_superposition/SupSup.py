from torch import nn

from continual_learning.methods.task_incremental.multi_task.gg\
    import BaseMultiTaskGGMethod
from continual_learning.methods.task_incremental.multi_task.gg.\
    supermask_superposition.base import SupSupMaskWrapper
from continual_learning.scenarios.tasks import Task


class SupermaskSuperposition(BaseMultiTaskGGMethod):
    """
    @misc{wen2020batchensemble,
    title={BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning},
    author={Yeming Wen and Dustin Tran and Jimmy Ba},
    year={2020},
    eprint={2002.06715},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
    }
    """
    def __init__(self, backbone: nn.Module,
                 pruning_percentage: float,
                 **kwargs):
        super().__init__()
        self.pruning_percentage = pruning_percentage
        self.apply_wrapper_to_model(model=backbone)

    def apply_wrapper_to_model(self, model):
        for name, module in model.named_children():

            if isinstance(module, (nn.Linear, nn.Conv2d)):
                l = getattr(model, name)
                setattr(model, name, SupSupMaskWrapper(l,
                                                       self.pruning_percentage))
            self.apply_wrapper_to_model(module)

    def set_task(self, backbone: nn.Module, task: Task, **kwargs):
        task_i = task.index
        for n, m in backbone.named_modules():
            if isinstance(m, SupSupMaskWrapper):
                m.set_current_task(task_i)

    def on_task_starts(self, backbone: nn.Module, task: Task, *args, **kwargs):
        task_i = task.index
        for n, m in backbone.named_modules():
            if isinstance(m, SupSupMaskWrapper):
                m.add_task()
                m.set_current_task(task_i)
