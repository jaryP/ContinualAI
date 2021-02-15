from torch import nn
from torch.nn import BatchNorm2d

from continual_learning.methods.task_incremental.multi_task.gg import \
    BaseMultiTaskGGMethod
from continual_learning.methods.task_incremental.multi_task.gg.\
    batch_ensemble.base import BElayer
from continual_learning.scenarios.tasks import SupervisedTask
from continual_learning.solvers.base import Solver


class BatchEnsemble(BaseMultiTaskGGMethod):
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
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.apply_wrapper_to_model(model=backbone)
        # self.model = backbone
        # print(backbone)

    def apply_wrapper_to_model(self, model):
        for name, module in model.named_children():

            if isinstance(module, (nn.Linear, nn.Conv2d)):
                l = getattr(model, name)
                setattr(model, name, BElayer(l))
            self.apply_wrapper_to_model(module)

        # for name, module in model.named_modules():
        #     if isinstance(module, (nn.Linear, nn.Conv2d)):
        #         l = getattr(model, name)
        #         setattr(model, name, BElayer(l))

    def get_parameters(self,
                       task: SupervisedTask,
                       backbone: nn.Module,
                       solver: Solver,
                       **kwargs):
        parameters = []
        current_task = task.index

        if current_task == 0:
            parameters.extend(backbone.parameters())
        else:
            for n, m in backbone.named_modules():
                if isinstance(m, BElayer):
                    parameters.append(m.tasks_alpha[current_task])
                    parameters.append(m.tasks_gamma[current_task])
                elif isinstance(m, BatchNorm2d):
                    m.track_running_stats = False
                    # parameters.append(m.parameters())

        return parameters

    def set_task(self,
                 backbone: nn.Module,
                 task: SupervisedTask,
                 **kwargs):
        task_i = task.index
        for n, m in backbone.named_modules():
            if isinstance(m, BElayer):
                m.set_current_task(task_i)

    def on_task_starts(self,
                       backbone: nn.Module,
                       task: SupervisedTask,
                       **kwargs):
        task_i = task.index
        for n, m in backbone.named_modules():
            if isinstance(m, BElayer):
                m.add_task()
                m.set_current_task(task_i)
                # parameters.append(m.tasks[current_task])
