from continual_learning.methods import BaseMethod


class BaseMultiTaskGGMethod(BaseMethod):
    pass


class Naive(BaseMultiTaskGGMethod):
    def __init__(self):
        super().__init__()

    def set_task(self, **kwargs):
        pass
