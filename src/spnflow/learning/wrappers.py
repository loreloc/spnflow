import joblib
import numpy as np
from spnflow.structure.node import Sum, assign_ids
from spnflow.learning.structure import learn_structure


def learn_classifier(data, distributions, domains, class_idx, n_jobs=1, **kwargs):
    assert data is not None
    assert distributions is not None
    assert class_idx >= 0

    n_samples, n_features = data.shape
    classes = data[:, class_idx]
    unique_classes = np.unique(classes)

    def learn_branch(local_data):
        n_local_samples, _ = local_data.shape
        weight = n_local_samples / n_samples
        local_spn = learn_structure(local_data, distributions, domains, **kwargs)
        return weight, local_spn

    branches = joblib.Parallel(n_jobs)(
        joblib.delayed(learn_branch)(data[classes == c]) for c in unique_classes
    )

    weights = [b[0] for b in branches]
    children = [b[1] for b in branches]
    root = Sum(weights, children)
    return assign_ids(root)

