from .base import Metric
from time import time


class TimeMetric(Metric):
    def __init__(self):
        self.current_time = None
        self.total_time = 0

    def on_task_starts(self, *args, **kwargs):
        self.current_time = time()

    def on_task_ends(self, *args, **kwargs):
        self.total_time += time() - self.current_time
        self.current_time = None

    def __call__(self, *args, **kwargs):
        return self.total_time
