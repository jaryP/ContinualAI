from typing import Union

from continual_learning.methods import BaseMethod


class BaseMultiTaskGNMethod(BaseMethod):
    def infer_task(self, x, backbone, solver, **kwargs) -> int:
        pass

    def set_task(self, backbone, solver, **kwargs):
        pass


class Naive(BaseMultiTaskGNMethod):
    def __init__(self):
        super().__init__()

    def infer_task(self, x, backbone, solver, **kwargs) -> Union[None, int]:
        return None


