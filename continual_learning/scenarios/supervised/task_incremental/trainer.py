import itertools
from collections import defaultdict
from typing import Callable, Union, List

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from continual_learning.eval import Evaluator
from continual_learning.eval.metrics import Metric
from continual_learning.eval.metrics.classification import Accuracy

from continual_learning.methods.task_incremental.multi_task.gn \
    import BaseMultiTaskGNMethod

from continual_learning.methods.task_incremental.multi_task.gg \
    import BaseMultiTaskGGMethod

from continual_learning.scenarios.base import AbstractTrainer
from continual_learning.scenarios.supervised.task_incremental import MultiTask
from continual_learning.scenarios.tasks import SupervisedTask
from continual_learning.solvers.multi_task import MultiHeadsSolver
from continual_learning.benchmarks import DatasetSplits


@torch.no_grad()
def predict_batch(x: torch.Tensor,
                  backbone: nn.Module,
                  solver: MultiHeadsSolver,
                  task_i: int):
    emb = backbone(x)
    a = solver(emb, task=task_i).cpu()
    return a.max(dim=1)[1]


@torch.no_grad()
def get_predictions(backbone: torch.nn.Module,
                    solver: MultiHeadsSolver,
                    task: SupervisedTask,
                    evaluate_task_index: int = None,
                    batch_size: int = 1,
                    device='cpu'):
    if evaluate_task_index is None:
        evaluate_task_index = task.index

    backbone.eval()
    solver.eval()

    true_labels = []
    predicted_labels = []

    for j, x, y in DataLoader(task, batch_size=batch_size):
        x = x.to(device)
        true_labels.extend(y.tolist())
        pred = predict_batch(x=x,
                             backbone=backbone,
                             solver=solver,
                             task_i=evaluate_task_index)
        predicted_labels.extend(pred.tolist())

    return np.asarray(true_labels), np.asarray(predicted_labels)


class GgTrainer(AbstractTrainer):
    def __init__(self, *,
                 backbone: nn.Module,
                 solver: MultiHeadsSolver,
                 method: BaseMultiTaskGGMethod,
                 tasks: MultiTask,
                 optimizer: Union[Callable, torch.optim.Optimizer],
                 task_epochs: int,
                 batch_size: int, criterion: Callable,
                 device: Union[str, torch.device],
                 metrics: List[Metric]):

        self.backbone = backbone
        self.solver = solver
        self.method = method
        self.tasks = tasks
        self.current_task = 0
        self.optimizer = optimizer
        self.epochs = task_epochs
        self.device = device
        self.batch_size = batch_size
        self.criterion = criterion

        self.evaluator = Evaluator(classification_metrics=metrics)

    def evaluate_on_split(self, task: SupervisedTask, split: DatasetSplits,
                          batch_size: int,
                          current_task_index: int = None):

        scores, accuracy = None, None
        evaluated_task_index = task.index

        cs = task.current_split
        task.current_split = split

        if current_task_index is None:
            current_task_index = evaluated_task_index

        if len(task) > 0:
            y_true, y_pred = get_predictions(self.backbone, self.solver, task,
                                             evaluate_task_index=evaluated_task_index,
                                             batch_size=batch_size,
                                             device=self.device)

            accuracy = (y_true == y_pred).sum() / len(y_true)

            scores = self.evaluator.evaluate(y_true, y_pred,
                                             current_task=current_task_index,
                                             evaluated_task=evaluated_task_index,
                                             evaluated_split=split)
        task.current_split = cs

        return scores, accuracy

    def train_epoch(self, task: SupervisedTask,
                    optimizer: torch.optim.Optimizer, batch_size: int = None):

        if batch_size is None:
            batch_size = self.batch_size

        task_index = task.index
        self.solver.task = task_index

        task.train()

        self.method.on_epoch_starts(backbone=self.backbone,
                                    solver=self.solver,
                                    task=task)

        self.method.set_task(backbone=self.backbone,
                             solver=self.solver,
                             task=task)

        self.backbone.to(self.device)

        modified_task = self.method.preprocess_dataset(backbone=self.backbone,
                                                       solver=self.solver,
                                                       task=task)

        if modified_task is not None:
            task = modified_task

        losses = []

        for idx, img, y in DataLoader(task, batch_size=batch_size):
            self.evaluator.on_batch_starts()

            self.method.on_batch_starts(batch=(idx, img, y),
                                        backbone=self.backbone,
                                        solver=self.solver, task=task)

            self.backbone.train()
            self.solver.train()

            img, y = img.to(self.device), y.to(self.device)

            emb = self.backbone(img)
            pred = self.solver(emb, task=task_index)
            loss = self.criterion(pred, y)

            losses.append(loss.item())

            self.method.before_gradient_calculation(current_loss=loss,
                                                    backbone=self.backbone,
                                                    solver=self.solver,
                                                    task=task, loss=loss)

            self.backbone.zero_grad()
            self.solver.zero_grad()

            loss.backward()

            self.method.after_gradient_calculation(backbone=self.backbone,
                                                   solver=self.solver,
                                                   task=task)

            optimizer.step()

            self.method.after_optimization_step(backbone=self.backbone,
                                                solver=self.solver,
                                                task=task)

            self.evaluator.on_batch_ends()

        self.method.on_epoch_ends(backbone=self.backbone,
                                  solver=self.solver,
                                  task=task)

        return np.mean(losses), \
               self.backbone.state_dict(), \
               self.solver.state_dict()

    def train_task(self, task: SupervisedTask, epochs: int):
        task.train()
        self.solver.add_task(len(task.labels))

        self.method.to(self.device)
        self.backbone.to(self.device)
        self.solver.to(self.device)

        self.method.on_task_starts(backbone=self.backbone, solver=self.solver,
                                   task=task)

        parameters = itertools.chain(
            self.method.get_parameters(task=task, backbone=self.backbone,
                                       solver=self.solver),
            self.solver.get_parameters(task=task.index))

        if callable(self.optimizer):
            optimizer = self.optimizer(parameters)
        else:
            self.change_optimizer_parameters(self.optimizer,
                                             parameters=parameters)
            self.optimizer.state = defaultdict(dict)

            optimizer = self.optimizer

        best_model = (None, None)
        best_score = -1

        for e in range(self.epochs):
            print(e)

            self.evaluator.on_epoch_starts()

            task.train()

            losses, model_state_dict, solver_state_dict = \
                self.train_epoch(task,
                                 optimizer=optimizer,
                                 batch_size=self.batch_size)

            self.evaluator.on_epoch_ends()

            train_scores, train_accuracy = \
                self.evaluate_on_split(task=task,
                                       batch_size=self.batch_size * 2,
                                       current_task_index=task.index,
                                       split=DatasetSplits.TRAIN)

            dev_scores, dev_accuracy = \
                self.evaluate_on_split(task=task,
                                       batch_size=self.batch_size * 2,
                                       current_task_index=task.index,
                                       split=DatasetSplits.DEV)

            if dev_scores is not None:
                score_to_compare = dev_accuracy
            else:
                score_to_compare = train_accuracy

            if score_to_compare > best_score:
                best_score = dev_accuracy
                best_model = (self.backbone.state_dict(),
                              self.solver.state_dict())

            # test_scores = self.evaluate_on_split(task=task,
            #                                      batch_size=self.batch_size * 2,
            #                                      current_task_index=task.index,
            #                                      split=DatasetSplits.TEST)

            print(train_scores, dev_scores)

        if best_score != -1:
            self.backbone.load_state_dict(best_model[0])
            self.solver.load_state_dict(best_model[1])

        self.method.on_task_ends(backbone=self.backbone, solver=self.solver,
                                 task=task)

    def train_full(self):
        for i, task in enumerate(self.tasks):
            self.evaluator.on_task_starts()

            self.train_task(task, epochs=self.epochs)

            self.evaluator.on_task_ends()

            for j in range(i + 1):
                evaluated_task = self.tasks[j]

                self.method.set_task(backbone=self.backbone,
                                     solver=self.solver,
                                     task=evaluated_task)

                self.evaluate_on_split(task=evaluated_task,
                                       batch_size=self.batch_size * 2,
                                       current_task_index=task.index,
                                       split=DatasetSplits.TRAIN)

                self.evaluate_on_split(task=evaluated_task,
                                       batch_size=self.batch_size * 2,
                                       current_task_index=task.index,
                                       split=DatasetSplits.DEV)

                self.evaluate_on_split(task=evaluated_task,
                                       batch_size=self.batch_size * 2,
                                       current_task_index=task.index,
                                       split=DatasetSplits.TEST)


class GnTrainer(AbstractTrainer):
    def __init__(self, *,
                 backbone: nn.Module,
                 solver: MultiHeadsSolver,
                 method: BaseMultiTaskGNMethod,
                 tasks: MultiTask,
                 optimizer: Union[Callable, torch.optim.Optimizer],
                 task_epochs: int,
                 batch_size: int, criterion: Callable,
                 device: Union[str, torch.device],
                 metrics: List[Metric],
                 task_inference_batch_size: int = 1):

        self.backbone = backbone
        self.solver = solver
        self.method = method
        self.tasks = tasks
        self.current_task = 0
        self.optimizer = optimizer
        self.epochs = task_epochs
        self.device = device
        self.batch_size = batch_size
        self.criterion = criterion
        self.task_inference_batch_size = task_inference_batch_size

        self.evaluator = Evaluator(classification_metrics=metrics)
        self.task_inference_evaluator = \
            Evaluator(classification_metrics=Accuracy())

    def evaluate_gn(self,
                    task: SupervisedTask,
                    split: DatasetSplits,
                    batch_size: int,
                    current_task_index: int = None):

        scores, accuracy = None, None
        correct_task_prediction = 0

        cs = task.current_split
        task.current_split = split

        if len(task) > 0:
            y_true, y_pred = [], []
            # task_prediction = []
            task_true, task_pred = [], []

            for _, x, y in DataLoader(task, batch_size=batch_size):
                task_i = self.method.infer_task(x=x.to(self.device),
                                                backbone=self.backbone,
                                                solver=self.solver)
                task_true.append(task.index)
                # y_true.append(y.item())
                # pred = -1

                if task_i is not None:
                    task_pred.append(task_i)
                    # if task_i == task.index:
                    #     correct_task_prediction += 1

                    # self.method.set_task(backbone=self.backbone,
                    #                      solver=self.solver,
                    #                      task_index=task_i)
                    #
                    # pred = predict_batch(x=x,
                    #                      backbone=self.backbone,
                    #                      solver=self.solver,
                    #                      task_prediction=task_i)
                    # pred = pred[0].item()
                else:
                    task_pred.append(-1)

                # y_pred.append(pred)

            # y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
            #
            # accuracy = (y_true == y_pred).sum() / len(y_true)

            scores = \
                self.task_inference_evaluator.evaluate(task_pred,
                                                    task_true,
                                                    current_task=current_task_index,
                                                    evaluated_task=task.index,
                                                    evaluated_split=split)

        task.current_split = cs

        return scores

    def evaluate_on_split(self, task: SupervisedTask, split: DatasetSplits,
                          batch_size: int,
                          current_task_index: int = None):

        scores, accuracy = None, None
        evaluated_task_index = task.index

        cs = task.current_split
        task.current_split = split

        if current_task_index is None:
            current_task_index = evaluated_task_index

        if len(task) > 0:
            y_true, y_pred = get_predictions(self.backbone, self.solver, task,
                                             evaluate_task_index=evaluated_task_index,
                                             batch_size=batch_size,
                                             device=self.device)

            accuracy = (y_true == y_pred).sum() / len(y_true)

            scores = self.evaluator.evaluate(y_true, y_pred,
                                             current_task=current_task_index,
                                             evaluated_task=evaluated_task_index,
                                             evaluated_split=split)
        task.current_split = cs

        return scores, accuracy

    def train_epoch(self,
                    task: SupervisedTask,
                    optimizer: torch.optim.Optimizer,
                    batch_size: int = None):

        if batch_size is None:
            batch_size = self.batch_size

        task_index = task.index
        self.solver.task = task_index

        task.train()

        self.method.on_epoch_starts(backbone=self.backbone, solver=self.solver,
                                    task=task)

        self.method.set_task(backbone=self.backbone, solver=self.solver,
                             task=task)

        self.backbone.to(self.device)

        modified_task = self.method.preprocess_dataset(backbone=self.backbone,
                                                       solver=self.solver,
                                                       task=task)

        if modified_task is not None:
            task = modified_task

        losses = []

        for idx, img, y in DataLoader(task, batch_size=batch_size):
            self.evaluator.on_batch_starts()

            self.method.on_batch_starts(batch=(idx, img, y),
                                        backbone=self.backbone,
                                        solver=self.solver, task=task)

            self.backbone.train()
            self.solver.train()

            img, y = img.to(self.device), y.to(self.device)

            emb = self.backbone(img)
            pred = self.solver(emb, task=task_index)
            loss = self.criterion(pred, y)

            losses.append(loss.item())

            self.method.before_gradient_calculation(current_loss=loss,
                                                    backbone=self.backbone,
                                                    solver=self.solver,
                                                    task=task, loss=loss)

            self.backbone.zero_grad()
            self.solver.zero_grad()

            loss.backward()

            self.method.after_gradient_calculation(backbone=self.backbone,
                                                   solver=self.solver,
                                                   task=task)

            optimizer.step()

            self.method.after_optimization_step(backbone=self.backbone,
                                                solver=self.solver,
                                                task=task)

            self.evaluator.on_batch_ends()

        self.method.on_epoch_ends(backbone=self.backbone,
                                  solver=self.solver,
                                  task=task)

        return np.mean(losses), \
               self.backbone.state_dict(), \
               self.solver.state_dict()

    def train_task(self,
                   task: SupervisedTask,
                   epochs: int):
        # TODO: aggiungere dev score e salvataggio modello migliore

        task.train()
        self.solver.add_task(len(task.labels))

        self.method.to(self.device)
        self.backbone.to(self.device)
        self.solver.to(self.device)

        self.method.on_task_starts(backbone=self.backbone, solver=self.solver,
                                   task=task)

        parameters = itertools.chain(
            self.method.get_parameters(task=task, backbone=self.backbone,
                                       solver=self.solver),
            self.solver.get_parameters(task=task.index))

        if callable(self.optimizer):
            optimizer = self.optimizer(parameters)
        else:
            self.change_optimizer_parameters(self.optimizer,
                                             parameters=parameters)
            self.optimizer.state = defaultdict(dict)

            optimizer = self.optimizer

        best_model = (None, None)
        best_score = -1

        for e in range(self.epochs):
            print(e)

            self.evaluator.on_epoch_starts()

            task.train()

            losses, model_state_dict, solver_state_dict = \
                self.train_epoch(task,
                                 optimizer=optimizer,
                                 batch_size=self.batch_size)

            self.evaluator.on_epoch_ends()

            train_scores, train_accuracy = self.evaluate_on_split(task=task,
                                                                  batch_size=self.batch_size * 2,
                                                                  current_task_index=task.index,
                                                                  split=DatasetSplits.TRAIN)

            dev_scores, dev_accuracy = self.evaluate_on_split(task=task,
                                                              batch_size=self.batch_size * 2,
                                                              current_task_index=task.index,
                                                              split=DatasetSplits.DEV)

            if dev_scores is not None:
                score_to_compare = dev_accuracy
            else:
                score_to_compare = train_accuracy

            if score_to_compare > best_score:
                best_score = dev_accuracy
                best_model = (self.backbone.state_dict(),
                              self.solver.state_dict())

            # test_scores = self.evaluate_on_split(task=task,
            #                                      batch_size=self.batch_size * 2,
            #                                      current_task_index=task.index,
            #                                      split=DatasetSplits.TEST)

            print(train_scores, dev_scores)

        if best_score != -1:
            self.backbone.load_state_dict(best_model[0])
            self.solver.load_state_dict(best_model[1])

        self.method.on_task_ends(backbone=self.backbone, solver=self.solver,
                                 task=task)

    def train_full(self):
        for i, task in enumerate(self.tasks):
            self.evaluator.on_task_starts()

            self.train_task(task, epochs=self.epochs)

            self.evaluator.on_task_ends()

            for j in range(i + 1):
                evaluated_task = self.tasks[j]

                self.method.set_task(backbone=self.backbone,
                                     solver=self.solver,
                                     task=evaluated_task)

                self.evaluate_on_split(task=evaluated_task,
                                       batch_size=self.batch_size * 2,
                                       current_task_index=task.index,
                                       split=DatasetSplits.TRAIN)

                self.evaluate_on_split(task=evaluated_task,
                                       batch_size=self.batch_size * 2,
                                       current_task_index=task.index,
                                       split=DatasetSplits.DEV)

                self.evaluate_on_split(task=evaluated_task,
                                       batch_size=self.batch_size * 2,
                                       current_task_index=task.index,
                                       split=DatasetSplits.TEST)

                scores = self.evaluate_gn(task=evaluated_task,
                                          batch_size=self.
                                          task_inference_batch_size,
                                          current_task_index=task.index,
                                          split=DatasetSplits.TEST)
                print(i, j, scores)