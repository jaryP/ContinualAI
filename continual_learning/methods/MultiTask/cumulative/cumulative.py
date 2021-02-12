from continual_learning.banchmarks.utils import ConcatDataset
from continual_learning.methods.MultiTask.base import BaseMultiTaskMethod


class Cumulative(BaseMultiTaskMethod):
    def __init__(self):
        super().__init__()
        self.tasks = []

    def on_task_starts(self, backbone, solver, task, *args, **kwargs):
        self.tasks.append(task)

    def preprocess_dataset(self,  backbone, solver, task, *args, **kwargs):
        print(len(ConcatDataset(self.tasks)))
        return ConcatDataset(self.tasks)

    def set_task(self, **kwargs):
        pass
