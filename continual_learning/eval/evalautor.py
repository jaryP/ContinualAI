__all__ = ['Evaluator']

from collections import defaultdict
from itertools import chain

import numpy as np
from typing import List, Union, Any

from continual_learning.benchmarks import DatasetSplits
from continual_learning.eval.metrics import Metric, ClassificationMetric, ContinualLearningMetric


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


class Evaluator:
    # def __init__(self, classification_metrics: Union[List[ClassificationMetric], ClassificationMetric] = None,
    #              cl_metrics: Union[List[ContinualLearningMetric], ContinualLearningMetric] = None,
    #              other_metrics: Union[List[Metric], Metric] = None):
    def __init__(self, classification_metrics: Union[List[Metric], Metric] = None):

        self._classification_metrics = []
        self._cl_metrics = []
        self._others_metrics = []
        self._custom_metrics = []

        self._scores = {split: {} for split in DatasetSplits
                        if split != DatasetSplits.ALL}

        self._labels = {}

        self._r = {split: {} for split in DatasetSplits
                   if split != DatasetSplits.ALL}

        self._custom_metrics = {split: {} for split in DatasetSplits
                        if split != DatasetSplits.ALL}

        if isinstance(classification_metrics, Metric):
            classification_metrics = [classification_metrics]

        for metric in classification_metrics:
            if isinstance(metric, ClassificationMetric):
                self._classification_metrics.append((metric.__class__.__name__, metric))
            elif isinstance(metric, ContinualLearningMetric):
                self._cl_metrics.append((metric.__class__.__name__, metric))
            elif isinstance(metric, Metric):
                self._others_metrics.append((metric.__class__.__name__, metric))

        # if classification_metrics is not None:
        #     if isinstance(classification_metrics, ClassificationMetric):
        #         classification_metrics = [classification_metrics]
        #
        #     self._classification_metrics = [(i.__class__.__name__, i) for i in classification_metrics]
        #
        # if cl_metrics is not None:
        #     if isinstance(cl_metrics, ContinualLearningMetric):
        #         cl_metrics = [cl_metrics]
        #
        #     self._cl_metrics = [(i.__class__.__name__, i) for i in cl_metrics]
        #
        # if other_metrics is not None:
        #     if isinstance(other_metrics, Metric):
        #         other_metrics = [other_metrics]

        # self._others_metrics = [(i.__class__.__name__, i) for i in other_metrics]

    @property
    def task_matrix(self) -> dict:
        return self._r
        # r = {}
        # for name, results in self._scores.items():
        #     k = sorted(results.keys(), reverse=False)
        #     _r = np.zeros((len(k), len(k)))
        #     for i in k:
        #         for j in range(i, len(k)):
        #             res = results[i][j]
        #             # res = res[1:]  # remove the first score, which is calculated before the training
        #             # mx = np.argmax(res)
        #             # mx = len(res) - 1
        #             # print(i, j, k)
        #             _r[i, j] = res[-1]
        #             # for j in range(i + 1, len(k)):
        #             #     ek = k[j]
        #             #     res = results[ek][ck]
        #             #     mx = len(res) - 1
        #             # _r[ck, ek] = res[-1]
        #
        #     r[name] = _r
        #
        # return r

    def classification_results(self) -> dict:
        return default_to_regular(self._scores)

    def cl_results(self) -> dict:
        _r = self.task_matrix
        res = {s: {} for s in _r}
        for split in _r:
            for name, m in self._classification_metrics:
                res[split][name] = {n: m(_r[split][name])
                                    for n, m in self._cl_metrics}
        return res

    def others_metrics_results(self) -> dict:
        res = {}
        for name, m in self._others_metrics:
            res[name] = m()
        return res

    def custom_metrics_results(self) -> dict:
        res = {}
        for name, m in self._others_metrics:
            res[name] = m()
        return res

    @property
    def classification_metrics(self) -> List[str]:
        return [name for name, _ in self._classification_metrics]

    @property
    def cl_metrics(self) -> List[str]:
        return [name for name, _ in self._cl_metrics]

    @property
    def others_metrics(self) -> List[str]:
        return [name for name, _ in self._others_metrics]

    @property
    def custom_metrics(self) -> List[str]:
        return [name for name, _ in self._custom_metrics]

    def evaluate(self,
                 y_true: Union[list, np.ndarray],
                 y_pred: Union[list, np.ndarray],
                 current_task: int,
                 evaluated_task: int,
                 evaluated_split: DatasetSplits):

        if current_task not in self._labels:
            self._labels[evaluated_task] = set(y_true)
        mx = max(current_task, evaluated_task) + 1

        scores = {}

        for name, m in self._classification_metrics:

            _m = m(y_true, y_pred, evaluator=self)

            split_scores = self._scores.get(evaluated_split, {})
            _s = split_scores.get(name, defaultdict(lambda: defaultdict(list)))

            _s[evaluated_task][current_task].append(_m)

            self._scores[evaluated_split][name] = _s
            scores[name] = _m

            r = self._r.get(evaluated_split, {})
            r = r.get(name, None)

            if r is None:
                r = np.zeros((mx, mx), dtype=float)
            elif r.shape[0] < mx:
                com = np.zeros((mx, mx), dtype=r.dtype)
                com[:r.shape[0], :r.shape[1]] = r
                r = com

            r[current_task, evaluated_task] = _m
            self._r[evaluated_split][name] = r

        return scores

    def final_score(self, y_true: Union[list, np.ndarray], y_pred: Union[list, np.ndarray],
                 current_task: int, evaluated_task: int):
        pass

    def add_custom_metric(self,
                          name: str,
                          value: Any,
                          evaluated_split: DatasetSplits):

        metrics = self._custom_metrics.get(evaluated_split, {})


    def add_cl_metric(self, metric: ContinualLearningMetric):
        self._cl_metrics.append((metric.__class__.__name__, metric))

    def add_metric(self, metric: ClassificationMetric):
        self._classification_metrics.append((metric.__class__.__name__, metric))

    def on_epoch_starts(self, *args, **kwargs):
        for n, m in chain(self._classification_metrics, self._cl_metrics, self._others_metrics):
            m.on_epoch_starts(*args, **kwargs)

    def on_epoch_ends(self, *args, **kwargs):
        for n, m in chain(self._classification_metrics, self._cl_metrics, self._others_metrics):
            m.on_epoch_ends(*args, **kwargs)

    def on_task_starts(self, *args, **kwargs):
        for n, m in chain(self._classification_metrics, self._cl_metrics, self._others_metrics):
            m.on_task_starts(*args, **kwargs)

    def on_task_ends(self, *args, **kwargs):
        for n, m in chain(self._classification_metrics, self._cl_metrics, self._others_metrics):
            m.on_task_ends(*args, **kwargs)

    def on_batch_starts(self, *args, **kwargs):
        for n, m in chain(self._classification_metrics, self._cl_metrics, self._others_metrics):
            m.on_batch_starts(*args, **kwargs)

    def on_batch_ends(self, *args, **kwargs):
        for n, m in chain(self._classification_metrics, self._cl_metrics, self._others_metrics):
            m.on_batch_ends(*args, **kwargs)

    # def task_starts(self, t):
    #     if t not in self._task_times:
    #         self._task_times[t].append(time.time())
    #     else:
    #         self._task_times[t][0] = time.time()
    #
    # def task_ends(self, t):
    #     v = self._task_times[t]
    #     end = time.time()
    #     self._task_times[t].append(end - v[0])
    #
    # def times(self):
    #     d = {}
    #     for t, v in self._task_times.items():
    #         d[t] = np.sum(v[1:])
    #     return d


class ExperimentsContainer:

    def __init__(self):
        self._experiments_results = []

    def add_evaluator(self, evaluator: Evaluator):
        self._experiments_results.append(evaluator)

    def get_mean_std(self):
        n = len(self._experiments_results)

        # cl_results = [i.cl_results for i in self._experiments_results]
        task_results = [i.classification_results for i in self._experiments_results]
        # task_r = [i.task_matrix for i in self._experiments_results]

        tasks_scores = defaultdict(lambda: defaultdict(list))

        for exp_n in range(len(task_results)):
            _tasks = defaultdict(list)
            for metric, results in task_results[exp_n].items():
                for task, scores in results.items():
                    tasks_scores[metric][task].append(scores)

    def matrix(self):
        task_results = [i.task_matrix for i in self._experiments_results]

        tasks_scores = defaultdict(list)

        for exp_n in range(len(task_results)):
            for metric, results in task_results[exp_n].items():
                tasks_scores[metric].append(results)

        scores = {}
        for metric, v in tasks_scores.items():
            # print(metric)
            # for i, v in t.items():
            scores[metric] = {'mean': np.asarray(v).mean(0), 'std': np.asarray(v).std(0)}

        return scores

    def task_scores(self):
        task_results = [i.classification_results() for i in self._experiments_results]

        tasks_scores = defaultdict(lambda: defaultdict(list))

        for exp_n in range(len(task_results)):
            for metric, results in task_results[exp_n].items():
                for task, scores in results.items():
                    tasks_scores[metric][task].append(scores)

        scores = defaultdict(dict)
        for metric, t in tasks_scores.items():
            for i, v in t.items():
                scores[metric][i] = {'mean': np.asarray(v).mean(0), 'std': np.asarray(v).std(0)}

        return scores

    def cl_metrics(self):
        cl_results = [i.cl_results() for i in self._experiments_results]

        res = defaultdict(lambda: defaultdict(list))

        for exp_n in range(len(cl_results)):
            for metric, v in cl_results[exp_n].items():
                for cl_metric, r in v.items():
                    res[metric][cl_metric].append(r)

        metrics = defaultdict(dict)
        for metric, t in res.items():
            for i, v in t.items():
                metrics[metric][i] = {'mean': np.asarray(v).mean(0), 'std': np.asarray(v).std(0)}

        return metrics

    def others_metrics(self):
        cl_results = [i.others_metrics_results() for i in self._experiments_results]

        res = defaultdict(list)

        for exp_n in range(len(cl_results)):
            for metric, v in cl_results[exp_n].items():
                # for cl_metric, r in v.items():
                res[metric].append(v)

        metrics = dict()
        for metric, v in res.items():
            # for i, v in t.items():
            metrics[metric] = {'mean': np.asarray(v).mean(0), 'std': np.asarray(v).std(0)}

        return metrics
