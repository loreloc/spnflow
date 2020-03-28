import numpy as np
from spnflow.structure.node import Sum, assign_ids
from spnflow.learning.structure import learn_structure


def learn_classifier(data, distributions, domains, class_idx, **kwargs):
    assert data is not None
    assert distributions is not None
    assert class_idx >= 0

    n_samples, n_features = data.shape
    assert len(distributions) == n_features, "Each feature must have a distribution"

    weights = []
    children = []
    classes = data[:, class_idx]
    unique_classes = np.unique(classes)
    for c in unique_classes:
        local_data = data[classes == c, :]
        n_local_samples, _ = local_data.shape
        weight = n_local_samples / n_samples
        local_spn = learn_structure(local_data, distributions, domains, **kwargs)
        weights.append(weight)
        children.append(local_spn)

    root = Sum(weights, children)
    return assign_ids(root)


