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

