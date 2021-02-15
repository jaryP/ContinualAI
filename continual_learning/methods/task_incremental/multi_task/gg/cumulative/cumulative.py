from continual_learning.benchmarks.utils import ConcatDataset
from continual_learning.methods.task_incremental.multi_task.gg\
    import BaseMultiTaskGGMethod


class Cumulative(BaseMultiTaskGGMethod):
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
