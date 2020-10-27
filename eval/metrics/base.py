from abc import abstractmethod


class Metric:
    def on_epoch_starts(self, *args, **kwargs):
        pass

    def on_epoch_ends(self, *args, **kwargs):
        pass

    def on_task_starts(self, *args, **kwargs):
        pass

    def on_task_ends(self, *args, **kwargs):
        pass

    def on_batch_starts(self, *args, **kwargs):
        pass

    def on_batch_ends(self, *args, **kwargs):
        pass

    # def after_optimization_step(self, *args, **kwargs):
    #     pass
    #
    # def after_back_propagation(self, *args, **kwargs):
    #     pass
    #
    # def before_gradient_calculation(self, *args, **kwargs):
    #     pass

    def __call__(self, *arg, **kwargs) -> int:
        pass


class ContinualLearningMetric(Metric):
    def __init__(self):
        super(ContinualLearningMetric, self).__init__()

    @abstractmethod
    def __call__(self, *args, **kwargs) -> int:
        raise NotImplementedError


class ClassificationMetric(Metric):
    def __init__(self):
        super(ClassificationMetric, self).__init__()

    @abstractmethod
    def __call__(self, y_true, y_pred, *args, **kwargs) -> int:
        raise NotImplementedError

