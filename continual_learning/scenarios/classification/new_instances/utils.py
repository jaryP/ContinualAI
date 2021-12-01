from typing import Sequence, Union, Callable, Any

import numpy as np
from scipy.ndimage.interpolation import rotate as np_rotate
from PIL.Image import Image
from torch import Tensor, tensor
from torchvision.transforms.functional import rotate

from continual_learning.datasets.base import UnsupervisedDataset, \
    SupervisedDataset, DatasetSplits
from continual_learning.scenarios.tasks import Task


class ImageRotation(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img: Union[Image, Tensor, np.ndarray]):
        if isinstance(img, np.ndarray):
            img = np_rotate(img, angle=self.degree, reshape=False)
        elif isinstance(img, Image):
            img = img.rotate(self.degree)
        elif isinstance(img, Tensor):
            img = rotate(img, angle=self.degree)
        else:
            raise ValueError(f'Accepted types are: '
                             f'[ndarray, PIL Image, Tensor] {type(img)}')
        return img


class PixelsPermutation(object):
    def __init__(self, index_permutation: Sequence[int]):
        self.permutation = index_permutation

    def __call__(self, img: Union[Image, Tensor, np.ndarray]):
        if isinstance(img, np.ndarray):
            img = img.reshape(-1)[self.permutation].reshape(*img.shape)
        elif isinstance(img, Image):
            img = img.getdata()
            img = img.reshape(-1)[self.permutation].reshape(*img.shape)
            img = Image.fromarray(img)
        elif isinstance(img, Tensor):
            img = img.numpy()
            img = img.reshape(-1)[self.permutation].reshape(*img.shape)
            img = tensor(img)
        else:
            raise ValueError(f'Accepted types are: '
                             f'[ndarray, PIL Image, Tensor] {type(img)}')

        return img


class TransformerTask(Task):
    def __init__(self,
                 *,
                 base_dataset: Union[UnsupervisedDataset, SupervisedDataset],
                 transformer: Callable[[Any], Any],
                 index: int,
                 **kwargs):

        train, dev, test = [base_dataset.get_indexes(DatasetSplits(v))
                            for v in ['train', 'dev', 'test']]

        super().__init__(base_dataset=base_dataset,
                         train=train,
                         dev=dev,
                         test=test,
                         index=index,
                         **kwargs)

        self.transformer = transformer

    def __getitem__(self, item):
        y = None

        a = super().__getitem__(item)
        if len(a) == 3:
            i, x, y = a
        else:
            i, x = a

        # if isinstance(self.base_dataset, UnsupervisedDataset):
        #     i, x = super().__getitem__(item)
        # else:
        #     i, x, y = super().__getitem__(item)

        if isinstance(i, list):
            x = list(map(self.transformer, x))
        else:
            x = self.transformer(x)

        if y is not None:
            return i, x, y

        return i, x
