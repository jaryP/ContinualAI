from typing import Union, List, Any, Callable

import numpy as np

from continual_learning.datasets.base import AbstractDataset
from . import NITransformingScenario

from continual_learning.scenarios.utils import \
    ImageRotation, PixelsPermutation


class ImageRotationScenario(NITransformingScenario):
    def __init__(self,
                 dataset: AbstractDataset,
                 tasks_n: int,
                 transformation_parameters: Union[List[any],
                                                  Callable[[Any], Any]],
                 infinite_stream: bool = False,
                 random_state: Union[np.random.RandomState, int] = None,
                 lazy_initialization: bool = True,
                 **kwargs):

        transform_function = self.get_rotation

        super().__init__(dataset=dataset,
                         tasks_n=tasks_n,
                         transform_factory=transform_function,
                         transformation_parameters=transformation_parameters,
                         infinite_stream=infinite_stream,
                         random_state=random_state,
                         lazy_initialization=lazy_initialization,
                         **kwargs)

    def get_rotation(self, degree, **kwargs):
        return ImageRotation(degree)


class PixelsPermutationScenario(NITransformingScenario):
    def __init__(self,
                 dataset: AbstractDataset,
                 tasks_n: int,
                 transformation_parameters: Union[List[any],
                                                  Callable[[Any], Any]],
                 infinite_stream: bool = False,
                 random_state: Union[np.random.RandomState, int] = None,
                 lazy_initialization: bool = True,
                 **kwargs):

        transform_factory = self.get_permutation

        super().__init__(dataset=dataset,
                         tasks_n=tasks_n,
                         transform_factory=transform_factory,
                         transformation_parameters=transformation_parameters,
                         infinite_stream=infinite_stream,
                         random_state=random_state,
                         lazy_initialization=lazy_initialization,
                         **kwargs)

    def get_permutation(self, permutation, **kwargs):
        return PixelsPermutation(permutation)
