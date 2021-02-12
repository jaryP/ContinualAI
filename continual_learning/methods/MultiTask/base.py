from continual_learning.methods import BaseMethod


class BaseMultiTaskMethod(BaseMethod):
    pass


class Naive(BaseMultiTaskMethod):
    def __init__(self):
        super().__init__()

    def set_task(self, **kwargs):
        pass
