from typing import Union, Sequence

from continual_learning.datasets.base import SupervisedDataset, \
    DatasetSplits, DatasetSplitContexView


def get_dataset_subset_using_labels(dataset: SupervisedDataset,
                                    labels: Union[int, Sequence]):
    if isinstance(labels, int):
        labels = [labels]

    keys = ['train', 'test', 'dev']
    splits = {k: [] for k in keys}

    for v in keys:
        ss = dataset.get_indexes(DatasetSplits(v))
        if len(ss) == 0:
            continue
        with DatasetSplitContexView(dataset, DatasetSplits(v)) as d:
            for i, (j, x, y) in enumerate(d):
                if y in labels:
                    splits[v].append(i)

    return dataset.get_subset(train_subset=splits['train'],
                              test_subset=splits['test'],
                              dev_subset=splits['dev'])



