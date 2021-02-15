from typing import Union

import itertools
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from continual_learning.methods.task_incremental.multi_task.gg\
    import BaseMultiTaskGGMethod
from continual_learning.methods.task_incremental.multi_task.gg.gem.utils\
    import qp
from continual_learning.scenarios.tasks import SupervisedTask


class GradientEpisodicMemory(BaseMultiTaskGGMethod):
    """
    @inproceedings{lopez2017gradient,
      title={Gradient episodic memory for continual learning},
      author={Lopez-Paz, David and Ranzato, Marc'Aurelio},
      booktitle={Advances in Neural Information Processing Systems},
      pages={6467--6476},
      year={2017}
    }
    """

    def __init__(self,  task_memory_size: int, margin: float = 0.5,
                 random_state: Union[np.random.RandomState, int] = None,
                 **kwargs):

        super().__init__()

        self.margin = margin
        self.task_memory_size = task_memory_size

        if random_state is None or isinstance(random_state, int):
            self.RandomState = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.RandomState = random_state
        else:
            raise ValueError("random_state can be None, Int or numpy RandomState object, an {} was give"
                             .format(type(random_state)))

        self.task_memory = []
        self.loss_f = nn.CrossEntropyLoss(reduction='mean')

    def on_task_ends(self, task: SupervisedTask, encoder: torch.nn.Module, *args, **kwargs):

        task.train()

        idxs = np.arange(len(task))
        idxs = self.RandomState.choice(idxs, self.task_memory_size, replace=False)

        images, labels = [], []
        encoder.eval()

        with torch.no_grad():
            for i in idxs:
                _, im, l = task[i]
                im = im.cpu()
                images.append(im)
                labels.append(l)

        images = torch.stack(images, 0)
        labels = torch.tensor(labels)

        self.task_memory.append((task.index, images.detach(), labels))

    def after_gradient_calculation(self, encoder: torch.nn.Module, solver, *args, **kwargs):

        if len(self.task_memory) > 0:
            named_parameters = dict(itertools.chain(encoder.named_parameters(),))

            current_gradients = {}

            for n, p in named_parameters.items():
                if p.requires_grad and p.grad is not None:
                    current_gradients[n] = deepcopy(p.grad.data.view(-1).cpu())

            tasks_gradients = {}

            for i, t in enumerate(self.task_memory):

                encoder.train()
                solver.eval()

                encoder.zero_grad()
                solver.zero_grad()

                index, image, label = t
                emb = encoder(image)
                o = solver(emb, task=i)

                loss = self.loss_f(o, label)
                loss.backward()

                gradients = {}
                for n, p in named_parameters.items():
                    if p.requires_grad and p.grad is not None:
                        gradients[n] = p.grad.data.view(-1).cpu()

                tasks_gradients[i] = deepcopy(gradients)

            encoder.zero_grad()
            solver.zero_grad()
            done = False

            for n, cg in current_gradients.items():
                tg = []
                for t, tgs in tasks_gradients.items():
                    tg.append(tgs[n])

                tg = torch.stack(tg, 1).cpu()
                a = torch.mm(cg.unsqueeze(0), tg)

                if (a < 0).sum() != 0:
                    done = True
                    cg_np = cg.unsqueeze(1).cpu().contiguous().numpy().astype(np.double)
                    tg = tg.numpy().transpose().astype(np.double)

                    try:
                        v = qp(tg, cg_np, self.margin)

                        cg_np += np.expand_dims(np.dot(v, tg), 1)

                        del tg

                        p = named_parameters[n]
                        p.grad.data.copy_(torch.from_numpy(cg_np).view(p.size()))

                    except Exception as e:
                        print(e)

            if not done:
                for n, p in named_parameters.items():
                    if p.requires_grad and p.grad is not None:
                        p.grad.copy_(current_gradients[n].view(p.grad.data.size()).cpu())


# class _GradientEpisodicMemory(NaiveMethod):
#     def __init__(self, model: torch.nn.Module, margin: float = 0.5, image_per_task: int = 300, sample_size: int = None,
#                  batch_size: int = 10, **kwargs):
#
#         super().__init__()
#
#         self.model = model
#         self.device = self.model.device
#
#         # self.margin = kwargs.get('margin', 0.5)
#         # self.memorized_task_size = kwargs.get('task_memory', 300)
#         # self.sample_size = kwargs.get('sample_size', self.memorized_task_size)
#         # self.batch_size = kwargs.get('batch_size', 10)
#         if margin < 0:
#             raise AttributeError('margin < 0')
#
#         if image_per_task <= 0:
#             raise AttributeError('image_per_task <= 0')
#
#         if sample_size is None:
#             sample_size = image_per_task
#         elif sample_size <= 0:
#             raise AttributeError('sample_size <= 0')
#         elif sample_size > image_per_task:
#             sample_size = image_per_task
#
#         if batch_size <= 0:
#             raise AttributeError('batch_size <= 0')
#         elif batch_size >= sample_size:
#             batch_size = sample_size
#
#         self.margin = margin
#         self.image_per_task = image_per_task
#         self.sample_size = sample_size
#         self.batch_size = batch_size
#
#         self.task_memory = []
#         self.loss_f = nn.CrossEntropyLoss()
#
#     def on_task_ends(self, *args, **kwargs):
#
#         task = kwargs['task']
#
#         task.train()
#         images, labels = task.sample(size=self.image_per_task)
#
#         # m = Memory(batch_size=self.batch_size, preprocessing=lambda x, y: (torch.tensor(x, device=self.model.device),
#         #                                                                    torch.tensor(y, device=self.model.device,
#         #                                                                                 dtype=torch.long)))
#
#         m = Memory(batch_size=self.batch_size, preprocessing=task.preprocessing)
#
#         m.extend(list(zip(images, labels)))
#
#         self.task_memory.append(m)
#
#     def before_optimization_step(self, *args, **kwargs):
#         # self.model.eval()
#
#         current_gradients = {}
#         if len(self.task_memory) > 0:
#             for n, p in self.model.named_parameters():
#                 if p.requires_grad and p.grad is not None:
#                     current_gradients[n] = deepcopy(p.grad.data.view(-1).cpu())
#
#             tasks_gradients = {}
#
#             for i, t in enumerate(self.task_memory):
#
#                 self.model.zero_grad()
#
#                 # if not self.model.SIT:
#                 #     self.model.task = i
#
#                 m = t.sample(self.sample_size, as_memory=True)
#
#                 for image, label in m(shuffle=True):
#                     # image = torch.stack(image, 0).to(self.device)
#                     #
#                     # label = torch.stack(label, dim=0).long().to(self.device)
#
#                     self.loss_f(self.model.evaluate_on_task(image, task=i), label).backward()
#
#                 gradients = {}
#                 for n, p in self.model.named_parameters():
#                     if p.requires_grad and p.grad is not None:
#                         gradients[n] = p.grad.data.view(-1).cpu()
#
#                 tasks_gradients[i] = deepcopy(gradients)
#
#             # self.model.train()
#             done = False
#
#             for n, cg in current_gradients.items():
#                 tg = []
#                 for t, tgs in tasks_gradients.items():
#                     tg.append(tgs[n])
#
#                 tg = torch.stack(tg, 1).cpu()
#                 a = torch.mm(cg.unsqueeze(0), tg)
#
#                 if (a < 0).sum() != 0:
#                     done = True
#                     cg_np = cg.unsqueeze(1).cpu().contiguous().numpy().astype(np.double)  # .astype(np.float16)
#                     tg = tg.numpy().transpose().astype(np.double)  # .astype(np.float16)
#
#                     try:
#                         v = self._qp(tg, cg_np)
#
#                         cg_np += np.expand_dims(np.dot(v, tg), 1)
#
#                         del tg
#
#                         p = dict(self.model.named_parameters())[n]
#                         # for name, p in :
#                         #     if name == n:
#                         p.grad.data.copy_(torch.from_numpy(cg_np).view(p.size()))
#                     except Exception as e:
#                         pass
#                         # print(e)
#
#             if not done:
#                 for n, p in self.model.named_parameters():
#                     if p.requires_grad and p.grad is not None:
#                         p.grad.copy_(current_gradients[n].view(p.grad.data.size()).cpu())
#
#     def _qp(self, past_tasks_gradient, current_gradient):
#         t = past_tasks_gradient.shape[0]
#         P = np.dot(past_tasks_gradient, past_tasks_gradient.transpose())
#         P = 0.5 * (P + P.transpose()) + np.eye(t) * 1e-3
#         q = np.dot(past_tasks_gradient, current_gradient) * -1
#         q = np.squeeze(q, 1)
#         h = np.zeros(t) + self.margin
#         G = np.eye(t)
#         v = quadprog.solve_qp(P, q, G, h)[0]
#         return v
#
#
# class _AveragedGradientEpisodicMemory(NaiveMethod):
#     # TODO: adattare e provare
#     def __init__(self, model: torch.nn.Module, *args, **kwargs):
#         super().__init__()
#
#         self.model = model
#
#         if not self.model.SIT:
#             raise ValueError('The model should have SIT equals to True.')
#
#         self.loss_f = nn.CrossEntropyLoss()
#
#         self.margin = kwargs.get('margin', 0.5)
#         self.memorized_task_size = kwargs.get('task_memory', 300)
#         self.sample_size = kwargs.get('sample_size', self.memorized_task_size // 10)
#         self.batch_size = kwargs.get('batch_size', 10)
#
#         self.device = self.model.device
#         self.task_memory = Memory(batch_processing=
#                                   lambda x, y: (torch.Tensor(x).to(self.model.device),
#                                                 torch.LongTensor(y).to(self.model.device)))
#
#     def on_task_ends(self, *args, **kwargs):
#
#         task = kwargs['task']
#
#         task.train()
#
#         images, labels = task.sample(size=self.memorized_task_size)
#
#         self.task_memory.add(*zip(images, labels))
#
#     def before_optimization_step(self, *args, **kwargs):
#
#         if len(self.task_memory) > 0:
#             current_gradients = {}
#             for n, p in self.model.named_parameters():
#                 if p.requires_grad and p.grad is not None:
#                     current_gradients[n] = deepcopy(p.grad.data.view(-1).cpu())
#
#             self.model.zero_grad()
#             self.model.train()
#
#             for images, labels in self.task_memory.sample(self.sample_size)(self.batch_size):
#                 self.model.zero_grad()
#                 self.loss_f(self.model(images), labels).backward()
#
#             gradients = {}
#             for n, p in self.model.named_parameters():
#                 if p.requires_grad and p.grad is not None:
#                     gradients[n] = p.grad.data.view(-1).cpu()
#
#             for n, cg in current_gradients.items():
#                 g = gradients[n]
#                 a = torch.dot(cg, g)
#
#                 if (a < 0).sum() != 0:
#                     g_new = cg - (torch.dot(cg, g)) / (torch.dot(g, g)) * g
#
#                     p = dict(self.model.named_parameters())[n]
#                     p.grad.data.copy_(g_new.view(p.size()))
