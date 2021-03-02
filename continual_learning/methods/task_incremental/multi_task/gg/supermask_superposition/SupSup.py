from torch import nn

from continual_learning.methods.task_incremental.multi_task.gg\
    import BaseMultiTaskGGMethod
from continual_learning.methods.task_incremental.multi_task.gg.\
    supermask_superposition.base import SupSupMaskWrapper
from continual_learning.scenarios.tasks import SupervisedTask


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
    def __init__(self, backbone: nn.Module, pruning_percentage: float):
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

        # for name, module in model.named_modules():
        #     if isinstance(module, (nn.Linear, nn.Conv2d)):
        #         l = getattr(model, name)
        #         setattr(model, name, BElayer(l))

    # def get_parameters(self, task: SupervisedTask, backbone: nn.Module, solver: Solver):
    #     parameters = []
    #     current_task = task.index
    #
    #     if current_task == 0:
    #         parameters.extend(backbone.parameters())
    #     else:
    #         for n, m in backbone.named_modules():
    #             if isinstance(m, BElayer):
    #                 parameters.append(m.tasks_alpha[current_task])
    #                 parameters.append(m.tasks_gamma[current_task])
    #             elif isinstance(m, BatchNorm2d):
    #                 m.track_running_stats = False
    #                 # parameters.append(m.parameters())
    #
    #     if isinstance(solver, MultiHeadsSolver):
    #         parameters.extend(solver.heads[current_task].parameters())
    #
    #     return parameters

    def set_task(self, backbone: nn.Module, task: SupervisedTask, **kwargs):
        task_i = task.index
        for n, m in backbone.named_modules():
            if isinstance(m, SupSupMaskWrapper):
                m.set_current_task(task_i)

    def on_task_starts(self, backbone: nn.Module, task: SupervisedTask, *args, **kwargs):
        task_i = task.index
        for n, m in backbone.named_modules():
            if isinstance(m, SupSupMaskWrapper):
                m.add_task()
                m.set_current_task(task_i)
                # parameters.append(m.tasks[current_task])
