from typing import Union, List

import numpy as np


def get_labels_set(labels: Union[tuple, list, np.ndarray],
                   labels_per_set: int,
                   shuffle_labels: bool = False,
                   random_state: Union[np.random.RandomState, int] = None
                   ) -> List[List[int]]:

    if shuffle_labels:
        if random_state is not None:
            if isinstance(random_state, int):
                random_state = np.random.RandomState(random_state)
            random_state.shuffle(labels)
        else:
            np.random.shuffle(labels)

    labels_sets = [list(labels[i:i + labels_per_set])
                   for i in range(0, len(labels), labels_per_set)]

    return labels_sets
