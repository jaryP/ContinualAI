from torch import nn

from continual_learning.methods.MultiTask.batch_ensemble.base import BElayer
from continual_learning.methods.base import BaseMethod
from continual_learning.scenarios.supervised import ClassificationTask
from continual_learning.solvers.base import Solver
from continual_learning.solvers.multi_task import MultiHeadsSolver


class BatchEnsemble(BaseMethod):
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
        self.model = backbone

    def apply_wrapper_to_model(self, model):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                l = getattr(model, name)
                setattr(model, name, BElayer(l))

    def get_parameters(self, task: ClassificationTask, backbone: nn.Module, solver: Solver):
        parameters = []
        current_task = task.index

        if current_task == 0:
            parameters.extend(backbone.parameters())
        else:
            for n, m in backbone.named_modules():
                if isinstance(m, BElayer):
                    parameters.append(m.tasks_alpha[current_task])
                    parameters.append(m.tasks_gamma[current_task])

        if isinstance(solver, MultiHeadsSolver):
            parameters.extend(solver.heads[current_task].parameters())

        return parameters

    def set_task(self, backbone: nn.Module, task: ClassificationTask, **kwargs):
        task_i = task.index
        for n, m in backbone.named_modules():
            if isinstance(m, BElayer):
                m.set_current_task(task_i)

    def on_task_starts(self, backbone: nn.Module, task: ClassificationTask, *args, **kwargs):
        task_i = task.index
        for n, m in backbone.named_modules():
            if isinstance(m, BElayer):
                m.add_task()
                m.set_current_task(task_i)
                # parameters.append(m.tasks[current_task])
