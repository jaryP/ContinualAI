import logging

import itertools
from copy import deepcopy

import torch

from continual_learning.methods.task_incremental.multi_task.gg \
    import BaseMultiTaskGGMethod
from continual_learning.scenarios.tasks import SupervisedTask


class ElasticWeightConsolidation(BaseMultiTaskGGMethod):
    """
    @article{kirkpatrick2017overcoming,
      title={Overcoming catastrophic forgetting in neural networks},
      author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume
              and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago
              and Grabska-Barwinska, Agnieszka and others},
      journal={Proceedings of the national academy of sciences},
      volume={114},
      number={13},
      pages={3521--3526},
      year={2017},
      publisher={National Acad Sciences}
    }
    """

    def __init__(self, sample_size: int = 200, penalty_importance: float = 1, logger: logging.Logger = None, **kwargs):
        super().__init__()

        self.sample_size = sample_size
        self.importance = penalty_importance

        if logger is not None:
            logger.info('EWC parameters:')
            logger.info(F'\tTask Sample size: {self.sample_size}')
            logger.info(F'\tPenalty importance: {self.importance}')

        self.memory = list()
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def on_task_ends(self, task: SupervisedTask, encoder: torch.nn.Module, solver, *args, **kwargs):

        final_w = {n: deepcopy(p.data) for n, p in itertools.chain(encoder.named_parameters())
                   if p.requires_grad and p.grad is not None}

        encoder.train()
        solver.train()
        task.train()

        encoder.zero_grad()
        solver.zero_grad()

        _s = 0
        cumloss = torch.zeros(1)

        for i, (_, image, label) in enumerate(task):
            emb = encoder(image)
            o = solver(emb, task=task.index)
            cumloss += self.loss(o, label)
            _s += image.size(0)

        cumloss = cumloss / _s
        cumloss.backward()

        f_matrix = {}
        for n, p in itertools.chain(encoder.named_parameters()):
            if p.requires_grad and p.grad is not None:
                f_matrix[n] = (deepcopy(p.grad.data) ** 2) / _s

        self.memory.append((final_w, f_matrix))

    def before_gradient_calculation(self, current_loss: torch.Tensor, encoder: torch.nn.Module, *args, **kwargs):

        if len(self.memory) > 0:

            penalty = 0
            p = {n: deepcopy(p.data) for n, p in itertools.chain(encoder.named_parameters())
                 if p.requires_grad and p.grad is not None}
            
            for w, f in self.memory:
                for n in w.keys():
                    _loss = f[n] * (p[n] - w[n]) ** 2
                    penalty += _loss.sum()

            current_loss += penalty * self.importance


# class OnlineElasticWeightConsolidation(NaiveMethod):
#     # TODO: Da provare
#     def __init__(self, config: ExperimentConfig):
#         super().__init__()
#         self.config = config.cl_technique_config
#
#         self.task_size = self.config.get('task_size', 200)
#         self.importance = self.config.get('penalty_importance', 1e3)
#         self.batch_size = self.config.get('batch_size', config.train_config['batch_size'])
#         # self.num_batches = self.config.get('num_batches', config.train_config['batch_size'])
#
#         # self.model = model
#         self.memory = list()
#         self.loss = torch.nn.CrossEntropyLoss(reduction='sum')
#
#         self.f = {}
#         self.final_w = {}
#
#     def on_task_ends(self, container: Container, *args, **kwargs):
#
#         task = container.current_task
#         # task = kwargs['task']
#
#         task.train()
#         # task_m = task.sample(size=self.sample_size, as_memory=True)
#
#         self.final_w = {n: deepcopy(p.data) for n, p in itertools.chain(container.encoder.named_parameters())
#                         if p.requires_grad and p.grad is not None}
#
#         container.encoder.train()
#         container.solver.train()
#
#         container.encoder.zero_grad()
#         container.solver.zero_grad()
#
#         _s = 0
#         cumloss = 0
#
#         for i, (image, label) in enumerate(task):
#             # _, images, labels = task.sample(size=self.task_siz)
#             emb = container.encoder(image)
#             o = container.solver(emb, task=task.index)
#             cumloss += self.loss(o, label)
#             _s += image.size(0)
#         # self.loss(o, labels).backward()
#         cumloss = cumloss / _s
#         cumloss.backward()
#
#         # for i, (image, label) in enumerate(task):
#         #     emb = container.encoder(image)
#         #     o = container.solver(emb, task=task.index)
#         #     self.loss(o, label).backward()
#         #
#         #     _s += label.shape[0]
#         #     if _s >= self.task_size:
#         #         break
#
#         # f_matrix = {}
#         # for n, p in itertools.chain(container.encoder.named_parameters()):
#         #     if p.requires_grad and p.grad is not None:
#         #         f_matrix[n] = (deepcopy(p.grad.data) ** 2) / _s
#
#         for n, p in self.model.named_parameters():
#             if p.requires_grad and p.grad is not None:
#                 self.f[n] = self.f.get(n, 0) + (deepcopy(p.grad.data) ** 2) / _s
#
#         # self.memory.append((final_w, f_matrix))
#
#     def before_gradient_calculation(self, container: Container, *args, **kwargs):
#
#         if container.current_task.index > 0:
#
#             penalty = 0
#
#             p = {n: deepcopy(p.data) for n, p in itertools.chain(container.encoder.named_parameters())
#                  if p.requires_grad and p.grad is not None}
#
#             for n in p.keys():
#                 _loss = self.f[n] * (p[n] - self.final_w[n]) ** 2
#                 penalty += _loss.sum()
#
#             container.current_loss += penalty * self.importance


# ewc = ElasticWeightConsolidation(10, 10)
# loss = torch.zeros(1)
# print(type(loss))
# ewc.before_gradient_calculation(loss, None)
# print(loss.item())
