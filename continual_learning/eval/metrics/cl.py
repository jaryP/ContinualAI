import numpy as np

from .base import ContinualLearningMetric


class BackwardTransfer(ContinualLearningMetric):

    def __call__(self, r: np.ndarray, *args, **kwargs):
        n = r.shape[0]
        v = 0

        if n == 1:
            return 1

        det = (n * (n - 1)) / 2

        for i in range(1, n):
            for j in range(i):
                v += r[i][j] - r[j][j]

        v = v / det
        return v


class LastBackwardTransfer(ContinualLearningMetric):
    def __call__(self, r: np.ndarray, *args, **kwargs):
        T = r.shape[0]
        if T == 1:
            return 1

        v = 0
        last = r[-1, :]

        for i in range(T - 1):
            v += (last[i] - r[i, i])
        v = v / (T - 1)

        return v


class FinalAccuracy(ContinualLearningMetric):
    def __call__(self, r: np.ndarray, *args, **kwargs):
        return r[-1, :].mean()


class TotalAccuracy(ContinualLearningMetric):
    def __call__(self, r: np.ndarray, *args, **kwargs):
        n = r.shape[0]
        v = np.tril(r, 0).sum()
        v = v / ((n * (n + 1)) / 2)
        return v