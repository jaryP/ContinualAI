from abc import abstractmethod, ABC
from typing import Callable

import torch
from torch import nn

from eval import Evaluator
from settings.supervised import MultiTask, ClassificationTask
from solvers.base import Solver
from solvers.multi_task import MultiHeadsSolver


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


class BaseMethod(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()

    def get_parameters(self, task, backbone: nn.Module, solver: Solver):
        task_i = task.index
        parameters = []
        parameters.extend(backbone.parameters())
        if isinstance(solver, MultiHeadsSolver):
            parameters.extend(solver.heads[task_i].parameters())
        return parameters

    def set_task(self, backbone, solver, task,  **kwargs):
        pass

    def before_evaluation(self, backbone, solver, task, *args, **kwargs):
        pass

    def on_epoch_starts(self, backbone, solver, task, *args, **kwargs):
        pass

    def on_epoch_ends(self, backbone, solver, task, *args, **kwargs):
        pass

    def on_task_starts(self, backbone, solver, task, *args, **kwargs):
        pass

    def on_task_ends(self, backbone, solver, task, *args, **kwargs):
        pass

    def on_batch_starts(self, backbone, solver, task, *args, **kwargs):
        pass

    def after_optimization_step(self, backbone, solver, task, *args, **kwargs):
        pass

    def after_gradient_calculation(self, backbone, solver, task, *args, **kwargs):
        pass

    def before_gradient_calculation(self, backbone, solver, task,  *args, **kwargs):
        pass


class Naive(BaseMethod):
    def __init__(self):
        super().__init__()

    def set_task(self, **kwargs):
        pass


def standard_trainer(method: BaseMethod, tasks: ClassificationTask, model: torch.nn.Module, solver: Solver,
                     epochs: int, optimizer: 'str',
                     solver_fn: Callable = None, optimizer_parameters: dict = None, device: str = 'cpu'):

    assert optimizer in ['Adam', 'SGD']
    if optimizer_parameters is None:
        if optimizer == 'Adam':
            optimizer_parameters = {'lr': 0.001, 'l1': 0}
        elif optimizer == 'SGD':
            optimizer_parameters = {'lr': 0.1, 'l1': 0, 'momentum': 0.5}

    test_results = Evaluator(classification_metrics=[Accuracy()],
                             cl_metrics=[BackwardTransfer(), TotalAccuracy(), FinalAccuracy(), LastBackwardTransfer()],
                             other_metrics=TimeMetric())

    train = Evaluator(classification_metrics=[Accuracy()],
                             cl_metrics=[BackwardTransfer(), TotalAccuracy(), FinalAccuracy(), LastBackwardTransfer()],
                             other_metrics=TimeMetric())

    for ti, t in enumerate(tasks):

        for e in range(epochs):
            print(e)
            t.train()
            method.on_epoch_starts(None)
            method.set_task(ti)

            for _, img, y in dataset:
                method.on_batch_starts(batch=(img, y))
                solver.task = ti

                backbone.train()
                solver.train()

                img, y = img.to(cuda), y.to(cuda)

                emb = backbone(img)
                pred = solver(emb)
                loss = torch.nn.functional.cross_entropy(pred, y)

                method.before_gradient_calculation(current_loss=loss, encoder=backbone)

                optimizer.zero_grad()
                method.before_gradient_calculation(None)
                loss.backward()
                method.after_gradient_calculation(encoder=backbone, solver=solver)
                optimizer.step()

                method.after_optimization_step(None)

            method.on_epoch_ends()

            t.test()
            y_true, y_pred = get_predictions(backbone, solver, dataset, evaluate_task_index=ti, device=cuda)
            test_results.evaluate(y_true, y_pred, current_task=ti, evaluated_task=ti)

            for i in range(ti):
                method.set_task(i)
                test_task = mt[i]
                test_task.test()
                solver.task = i

                test_task = DataLoader(test_task, batch_size=64)
                y_true, y_pred = get_predictions(backbone, solver, test_task, evaluate_task_index=i, device=cuda)
                test_results.evaluate(y_true, y_pred, current_task=ti, evaluated_task=i)

