from typing import Sequence, Union

import numpy as np
from scipy.ndimage.interpolation import rotate as np_rotate
from PIL.Image import Image
from torch import Tensor, tensor
from torchvision.transforms.functional import rotate


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


