import logging
import numpy as np

from typing import Union

import torch

import torch.nn.functional as F
# from continual_ai.continual_learning_strategies.base import NaiveMethod, Container
# from continual_ai.base import ExperimentConfig
# from continual_ai.utils import Sampler
# from continual_ai.cl_strategies import NaiveMethod, Container
# from continual_ai.iterators import Sampler
# from continual_ai.utils import ExperimentConfig
from methods import NaiveMethod
from settings.supervised import ClassificationTask


class EmbeddingRegularization(NaiveMethod):
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

    def __init__(self, task_memory_size: int, importance: float = 1, sample_size: int = None,
                 distance: str = 'cosine',
                 random_state: Union[np.random.RandomState, int] = None, **kwargs):

        NaiveMethod.__init__(self)

        self.memorized_task_size = task_memory_size
        if sample_size is None:
            sample_size = task_memory_size

        self.sample_size = min(sample_size, self.memorized_task_size)

        self.importance = importance

        # TODO: messaggio
        assert distance in ['cosine', 'euclidean']
        self.distance = distance

        # self.supervised = config.get('supervised', True)
        # self.normalize = config.get('normalize', False)
        # self.batch_size = config.get('batch_size', 25)

        if random_state is None or isinstance(random_state, int):
            self.RandomState = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.RandomState = random_state
        else:
            raise ValueError("random_state can be None, Int or numpy RandomState object, an {} was give"
                             .format(type(random_state)))

        # if logger is not None:
        #     logger.info('ER parameters:')
        #     logger.info(F'\tMemorized task size: {self.memorized_task_size}')
        #     logger.info(F'\tSample size: {self.sample_size}')
        #     logger.info(F'\tPenalty importance: {self.importance}')
        #     logger.info(F'\tDistance: {self.distance}')
        #     logger.info(F'\tNormalize: {self.normalize}')

        self.task_memory = []

    def on_task_ends(self, task: ClassificationTask, encoder: torch.nn.Module, *args, **kwargs):

        task.train()

        idxs = np.arange(len(task))
        idxs = self.RandomState.choice(idxs, self.task_memory_size, replace=False)

        _, images, _ = task[idxs]

        encoder.eval()

        embs = encoder(images)

        self.task_memory.append((task.index, images.detach(), embs.detach()))

    def before_gradient_calculation(self, current_loss: torch.Tensor, encoder: torch.nn.Module, *args, **kwargs):

        if len(self.task_memory) > 0:

            to_back = []
            for _, images, embs in self.task_memory:

                idxs = np.arange(len(images))
                idxs = self.RandomState.choice(idxs, self.task_memory_size, replace=False)
                idxs = torch.tensor(idxs)

                images, embs = images[idxs], embs[idxs]

                new_embedding = encoder(images)

                if self.normalize:
                    new_embedding = F.normalize(new_embedding, p=2, dim=1)

                if self.distance == 'euclidean':
                    dist = (embs - new_embedding).norm(p=None, dim=1)
                elif self.distance == 'cosine':
                    cosine = torch.nn.functional.cosine_similarity(embs, new_embedding)
                    dist = torch.sub(1, cosine)
                else:
                    assert False

                to_back.append(dist)

            to_back = torch.cat(to_back)

            current_loss += torch.mul(to_back.mean(), self.importance)
