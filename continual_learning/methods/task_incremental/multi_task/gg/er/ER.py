import numpy as np

from typing import Union

import torch

import torch.nn.functional as F

from continual_learning.methods.task_incremental.multi_task.gg \
    import BaseMultiTaskGGMethod
from continual_learning.methods.base import BaseMethod
from continual_learning.scenarios.tasks import SupervisedTask


class EmbeddingRegularization(BaseMultiTaskGGMethod):
    """
    @article{POMPONI2020,
    title = "Efficient continual learning in neural networks with embedding regularization",
    journal = "Neurocomputing",
    year = "2020",
    issn = "0925-2312",
    doi = "https://doi.org/10.1016/j.neucom.2020.01.093",
    url = "http://www.sciencedirect.com/science/article/pii/S092523122030151X",
    author = "Jary Pomponi and Simone Scardapane and Vincenzo Lomonaco and Aurelio Uncini",
    keywords = "Continual learning, Catastrophic forgetting, Embedding, Regularization, Trainable activation functions",
    }
    """

    # def get_parameters(self, current_task, network: nn.Module, solver: Solver):
    #     network.parameters()
    #     solver.parameters()
    #

    def __init__(self, task_memory_size: int, importance: float = 1, sample_size: int = None,
                 distance: str = 'cosine',
                 random_state: Union[np.random.RandomState, int] = None, **kwargs):

        BaseMethod.__init__(self)

        self.memorized_task_size = task_memory_size
        if sample_size is None:
            sample_size = task_memory_size

        self.sample_size = min(sample_size, self.memorized_task_size)

        self.importance = importance

        # TODO: messaggio
        assert distance in ['cosine', 'euclidean']
        self.distance = distance

        # self._supervised = config.get('_supervised', True)
        # self.normalize = config.get('normalize', False)
        # self.batch_size = config.get('batch_size', 25)

        if random_state is None or isinstance(random_state, int):
            self.RandomState = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.RandomState = random_state
        else:
            raise ValueError("random_state can be None, Int or numpy RandomState object, an {} was give"
                             .format(type(random_state)))

        self.task_memory = []

    def on_task_ends(self, task: SupervisedTask, backbone: torch.nn.Module, *args, **kwargs):

        task.train()

        idxs = np.arange(len(task))
        idxs = self.RandomState.choice(idxs, self.sample_size, replace=False)

        img, embs = [], []
        backbone.eval()

        with torch.no_grad():
            for i in idxs:
                _, im, _ = task[i]
                emb = backbone(im)
                im = im.cpu()
                emb = emb.cpu()
                img.append(im)
                embs.append(emb)

        img = torch.stack(img, 0)
        embs = torch.stack(embs, 0)

        # dataset = DataLoader(t, batch_size=64)
        #
        # _, images, _ = RandomSampler(task)[:self.sample_size]
        # _, images, _ = task[idxs]
        #
        #
        # embs = encoder(images)

        self.task_memory.append((task.index, img, embs))

    def before_gradient_calculation(self, loss: torch.Tensor, backbone: torch.nn.Module, *args, **kwargs):

        if len(self.task_memory) > 0:

            to_back = []
            for _, images, embs in self.task_memory:

                # idxs = np.arange(len(images))
                # idxs = self.RandomState.choice(idxs, self.sample_size, replace=False)
                # idxs = torch.tensor(idxs)

                # images, embs = images[idxs], embs[idxs]

                new_embedding = backbone(images)

                # if self.normalize:
                #     new_embedding = F.normalize(new_embedding, p=2, dim=1)

                if self.distance == 'euclidean':
                    dist = (embs - new_embedding).norm(p=None, dim=1)
                elif self.distance == 'cosine':
                    cosine = torch.nn.functional.cosine_similarity(embs, new_embedding)
                    dist = torch.sub(1, cosine)
                else:
                    assert False

                to_back.append(dist)

            to_back = torch.cat(to_back)

            loss += torch.mul(to_back.mean(), self.importance)
