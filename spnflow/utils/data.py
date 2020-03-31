import numpy as np
from spnflow.structure.leaf import LeafType


def get_data_domains(data, distributions):
    """
    Compute the domains based on the training data and the features distributions.

    :param data: The training data.
    :param distributions: A list of distribution classes.
    :return: A list of domains.
    """
    assert data is not None
    assert distributions is not None

    domains = []
    for i, d in enumerate(distributions):
        col = data[:, i]
        min = np.min(col)
        max = np.max(col)
        if d.LEAF_TYPE == LeafType.DISCRETE:
            domains.append(list(range(max.astype(int) + 1)))
        elif d.LEAF_TYPE == LeafType.CONTINUOUS:
            domains.append([min, max])
        else:
            raise NotImplementedError("Domain for leaf type " + d.LEAF_TYPE.__name__ + " not implemented")
    return domains


def ohe_data(data, domain):
    """
    One-Hot-Encoding
    :param data: The data to encode.
    :param domain: The domain to use.
    :return: The One Hot encoded data
    """
    n_samples, _ = len(data)
    ohe = np.zeros((n_samples, len(domain)))
    ohe[data == domain] = 1
    return ohe
