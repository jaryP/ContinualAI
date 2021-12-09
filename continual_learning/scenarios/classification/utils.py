from typing import Union, Sequence

import numpy as np

from continual_learning.datasets.base import DatasetSplitsContainer


def get_dataset_subset_using_labels(dataset: DatasetSplitsContainer,
                                    labels: Union[int, Sequence]):
    if isinstance(labels, int):
        labels = [labels]

    keys = ['train', 'test', 'dev']
    splits = {k: [] for k in keys}

    for v in keys:
        subset = dataset.get_split(v)
        if len(subset) == 0:
            continue
        ys = subset.targets
        # w = np.where(np.in1d(ys, labels))[0]
        # ss = subset.base_dataset_indexes
        ss = subset.base_dataset_indexes[np.in1d(ys, labels)]

        splits[v] = ss

    return dataset.get_subset(train_subset=splits['train'],
                              test_subset=splits['test'],
                              dev_subset=splits['dev'],
                              as_splitted_dataset=True)



