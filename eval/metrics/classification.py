import numpy as np
from .base import ClassificationMetric


# class F1(ClassificationMetric):
#     def __init__(self, average='macro', weights=None):
#         super().__init__()
#         self.average = average
#         self.weights = weights
#
#     def __call__(self, y_true, y_pred, *args, **kwargs):
#         raise NotImplementedError
#         # return f1_score(y_true, y_pred, sample_weight=self.weights, average=self.average)


class Accuracy(ClassificationMetric):
    def __call__(self, y_true, y_pred, *args, **kwargs):
        eq = np.asarray(y_true) == np.asarray(y_pred)
        accuracy = eq.sum()/len(y_true)
        return accuracy
