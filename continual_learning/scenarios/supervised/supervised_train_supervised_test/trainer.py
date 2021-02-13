import itertools
from collections import defaultdict
from typing import Callable, Union, List

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from continual_learning.eval import Evaluator
from continual_learning.eval.metrics import Metric
from continual_learning.methods.MultiTask.base import BaseMultiTaskMethod
from continual_learning.scenarios.base import AbstractTrainer
from continual_learning.scenarios.supervised.supervised_train_supervised_test import MultiTask
from continual_learning.scenarios.tasks import SupervisedTask
from continual_learning.solvers.multi_task import MultiHeadsSolver
from continual_learning.banchmarks import DatasetSplits


@torch.no_grad()
def get_predictions(backbone: torch.nn.Module, solver: MultiHeadsSolver,
                    task: SupervisedTask, evaluate_task_index: int = None, batch_size: int = 1,
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
        emb = backbone(x)
        a = solver(emb, task=evaluate_task_index).cpu()
        predicted_labels.extend(a.max(dim=1)[1].tolist())

    return np.asarray(true_labels), np.asarray(predicted_labels)


class Trainer(AbstractTrainer):
    def __init__(self, *, backbone: nn.Module, solver: MultiHeadsSolver, method: BaseMultiTaskMethod,
                 tasks: MultiTask,
                 optimizer: Union[Callable, torch.optim.Optimizer], task_epochs: int,
                 batch_size: int, criterion: Callable,
                 device: Union[str, torch.device], metrics: List[Metric]):

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

    def evaluate_on_split(self, task: SupervisedTask, split: DatasetSplits, batch_size: int,
                          current_task_index: int = None):

        evaluated_task_index = task.index
        task.current_split = split
        if current_task_index is None:
            current_task_index = evaluated_task_index

        if len(task) > 0:
            y_true, y_pred = get_predictions(self.backbone, self.solver, task, evaluate_task_index=evaluated_task_index,
                                             batch_size=batch_size, device=self.device)

            scores = self.evaluator.evaluate(y_true, y_pred,
                                             current_task=current_task_index,
                                             evaluated_task=evaluated_task_index,
                                             evaluated_split=split)
            return scores
        else:
            return None

    def train_epoch(self, task: SupervisedTask, optimizer: torch.optim.Optimizer, batch_size: int = None):

        if batch_size is None:
            batch_size = self.batch_size

        task_index = task.index
        self.solver.task = task_index

        task.train()

        self.method.on_epoch_starts(backbone=self.backbone, solver=self.solver, task=task)

        self.method.set_task(backbone=self.backbone, solver=self.solver, task=task)

        self.backbone.to(self.device)

        modified_task = self.method.preprocess_dataset(backbone=self.backbone, solver=self.solver, task=task)

        if modified_task is not None:
            task = modified_task

        losses = []

        for idx, img, y in DataLoader(task, batch_size=batch_size):
            self.evaluator.on_batch_starts()

            self.method.on_batch_starts(batch=(idx, img, y), backbone=self.backbone, solver=self.solver, task=task)

            self.backbone.train()
            self.solver.train()

            img, y = img.to(self.device), y.to(self.device)

            emb = self.backbone(img)
            pred = self.solver(emb, task=task_index)
            loss = self.criterion(pred, y)

            losses.append(loss.item())

            self.method.before_gradient_calculation(current_loss=loss, backbone=self.backbone, solver=self.solver,
                                                    task=task, loss=loss)

            self.backbone.zero_grad()
            self.solver.zero_grad()

            loss.backward()

            self.method.after_gradient_calculation(backbone=self.backbone, solver=self.solver, task=task)

            optimizer.step()

            self.method.after_optimization_step(backbone=self.backbone, solver=self.solver, task=task)

            self.evaluator.on_batch_ends()

        self.method.on_epoch_ends(backbone=self.backbone, solver=self.solver, task=task)

        return np.mean(losses), self.backbone.state_dict(), self.solver.state_dict()

        # print(scores)
        # for i in range(ti):
        #     test_task = mt[i]
        #     test_task.dev()
        #     solver.task = i
        #
        #     print(i, ti)
        #     # method.set_task(task=i, backbone=backbone)
        #     print(backbone.state_dict()[list(backbone.state_dict().keys())[0]][0][0])
        #     method.set_task(backbone=backbone, solver=solver, task=test_task)
        #     print(backbone.state_dict()[list(backbone.state_dict().keys())[0]][0][0])
        #
        #     # test_task = DataLoader(test_task, batch_size=64)
        #     y_true, y_pred = get_predictions(backbone, solver, dataset, evaluate_task_index=i, device=cuda)
        #     dev_results.evaluate(y_true, y_pred, current_task=ti, evaluated_task=i)
        # score = scores['Accuracy']
        # if score > best_score:
        #     best_score = score
        #     best_model = (solver.state_dict(), backbone.state_dict())

    def train_task(self, task: SupervisedTask, epochs: int):
        # TODO: aggiungere dev score e salvataggio modello migliore

        best_model = (None, None)
        best_score = 0

        task.train()
        self.solver.add_task(len(task.labels))

        self.method.to(self.device)
        self.backbone.to(self.device)
        self.solver.to(self.device)

        self.method.on_task_starts(backbone=self.backbone, solver=self.solver, task=task)

        parameters = itertools.chain(
            self.method.get_parameters(task=task, backbone=self.backbone, solver=self.solver),
            self.solver.get_parameters(task=task.index))

        if callable(self.optimizer):
            optimizer = self.optimizer(parameters)
        else:
            self.change_optimizer_parameters(self.optimizer, parameters=parameters)
            self.optimizer.state = defaultdict(dict)

            optimizer = self.optimizer

        for e in range(self.epochs):
            print(e)

            self.evaluator.on_epoch_starts()

            task.train()

            losses, model_state_dict, solver_state_dict = self.train_epoch(task,
                                                                           optimizer=optimizer,
                                                                           batch_size=self.batch_size)
            self.evaluator.on_epoch_ends()

            train_scores = self.evaluate_on_split(task=task, batch_size=self.batch_size * 2,
                                                  current_task_index=task.index,
                                                  split=DatasetSplits.TRAIN)

            dev_scores = self.evaluate_on_split(task=task, batch_size=self.batch_size * 2,
                                                current_task_index=task.index,
                                                split=DatasetSplits.DEV)

            print(train_scores, dev_scores)

        self.method.on_task_ends(backbone=self.backbone, solver=self.solver, task=task)

    def train_full(self):
        for i, task in enumerate(self.tasks):
            self.evaluator.on_task_starts()

            self.train_task(task, epochs=self.epochs)

            self.evaluator.on_task_ends()

            for j in range(i + 1):
                evaluated_task = self.tasks[j]
                self.method.set_task(backbone=self.backbone, solver=self.solver, task=evaluated_task)

                self.evaluate_on_split(task=evaluated_task, batch_size=self.batch_size * 2,
                                       current_task_index=task.index,
                                       split=DatasetSplits.TRAIN)

                self.evaluate_on_split(task=evaluated_task, batch_size=self.batch_size * 2,
                                       current_task_index=task.index,
                                       split=DatasetSplits.DEV)

                self.evaluate_on_split(task=evaluated_task, batch_size=self.batch_size * 2,
                                       current_task_index=task.index,
                                       split=DatasetSplits.TEST)

# if __name__ == '__main__':
#     import numpy as np
#     import torch
#     from torch import nn
#
#     # from backbone_networks.lenet import LeNet
#     # from continual_learning.banchmarks.cifar import CIFAR10
#     # from eval import Evaluator, Accuracy, TotalAccuracy, BackwardTransfer, TimeMetric, FinalAccuracy, LastBackwardTransfer
#     # from methods.MultiTask import EmbeddingRegularization
#     # from methods.MultiTask.super_mask_pruning.BSP import SuperMask
#     # from scenarios._supervised import MultiTask, ClassificationTask
#     # from solvers.multi_task import MultiHeadsSolver
#     # from continual_learning.backbone_networks import resnet20
#     # from continual_learning.backbone_networks.lenet import LeNet
#     # from continual_learning.banchmarks.cifar import CIFAR10
#     # from continual_learning.banchmarks.core50 import Core50_128
#     from torchvision.transforms import transforms
#
#     from continual_learning.eval.metrics.cl import BackwardTransfer
#     from continual_learning.eval.metrics.cl import FinalAccuracy
#     from continual_learning.eval.metrics.classification import Accuracy
#
#     from continual_learning.backbone_networks import LeNet
#     from continual_learning.banchmarks import MNIST, DatasetSplits
#     # from continual_learning.eval import Evaluator, Accuracy, BackwardTransfer, TotalAccuracy, FinalAccuracy, \
#     #     LastBackwardTransfer, TimeMetric, Metric
#     # from continual_learning.scenarios._supervised import ClassificationTask, MultiTask
#     # from continual_learning.methods.base import Naive
#     from continual_learning.scenarios.tasks import SupervisedTask
#     from continual_learning.scenarios.supervised.supervised_train_supervised_test.multi_task import MultiTask
#     from continual_learning.solvers.multi_task import MultiHeadsSolver
#
#     t = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(0.1307, 0.3081),
#         transforms.Resize(32),
#         # torch.nn.Flatten(0)
#     ])
#
#     tt = t
#     d = MNIST(download_if_missing=True, transformer=t, test_transformer=tt,
#               data_folder='/media/jary/Data/progetti/CL/cl_framework/continual_learning/tests/training/mnist')
#     mt = MultiTask(dataset=d, labels_per_task=2)
#
#     backbone = LeNet(input_size=1)
#
#     _, img, _ = next(iter(mt[0].get_iterator(batch_size=1)))
#     output_dim = backbone(img).shape[1:]
#     if len(output_dim) == 1:
#         output_dim = output_dim[0]
#
#
#     def solver_fn(input, output):
#         return nn.Sequential(*[nn.Linear(input, input // 2),
#                                nn.Dropout(0.25),
#                                nn.ReLU(),
#                                nn.Linear(input // 2, input // 4),
#                                nn.Dropout(0.25),
#                                nn.ReLU(),
#                                nn.Linear(input // 4, output)])
#
#
#     solver = MultiHeadsSolver(input_dim=output_dim, topology=solver_fn)
#     method = Naive()
#
#
#     def opt(params):
#         return torch.optim.Adam(params, lr=0.001)
#
#
#     trainer = Trainer(batch_size=32, backbone=backbone, task_epochs=1, solver=solver, tasks=mt,
#                       optimizer=opt, criterion=torch.nn.CrossEntropyLoss(), device='cpu', method=method,
#                       metrics=[Accuracy(), BackwardTransfer(), FinalAccuracy()])
#
#     trainer.train_full()
#
#     print(trainer.evaluator.cl_results())
