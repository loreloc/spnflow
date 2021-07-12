import numpy as np

from collections import deque
from deeprob.spn.structure.leaf import LeafType


def gvs_cols(data, distributions, domains, random_state, p=5.0):
    """
    Greedy Variable Splitting (GVS) independence test.

    :param data: The data.
    :param distributions: The distributions.
    :param domains: The domains.
    :param random_state: The random state.
    :param p: The threshold for the G-Test.
    :return: A partitioning of features.
    """
    n_samples, n_features = data.shape
    rand_init = random_state.randint(0, n_features)
    features_set = set(filter(lambda x: x != rand_init, range(n_features)))
    dependent_features_set = {rand_init}

    features_queue = deque()
    features_queue.append(rand_init)

    while features_queue:
        feature = features_queue.popleft()
        features_remove = set()

        for other_feature in features_set:
            if not gtest(data, feature, other_feature, distributions, domains, p):
                features_remove.add(other_feature)
                dependent_features_set.add(other_feature)
                features_queue.append(other_feature)
        features_set = features_set.difference(features_remove)

    partition = np.zeros(n_features, dtype=np.int32)
    partition[list(dependent_features_set)] = 1
    return partition


def gtest(data, i, j, distributions, domains, p):
    """
    The G-Test independence test between two features.

    :param data: The data.
    :param i: The index of the first feature.
    :param j: The index of the second feature.
    :param distributions: The distributions.
    :param domains: The domains.
    :param p: The threshold for the G-Test.
    :return: False if the features are assumed to be dependent, True otherwise.
    """
    n_samples = len(data)
    x1, x2 = data[:, i], data[:, j]

    if distributions[i].LEAF_TYPE == LeafType.DISCRETE and distributions[j].LEAF_TYPE == LeafType.DISCRETE:
        b1 = domains[i] + [len(domains[i])]
        b2 = domains[j] + [len(domains[j])]
        hist, _, _ = np.histogram2d(x1, x2, bins=[b1, b2])
    elif distributions[i].LEAF_TYPE == LeafType.CONTINUOUS and distributions[j].LEAF_TYPE == LeafType.CONTINUOUS:
        bins = np.ceil(np.cbrt(n_samples)).astype(np.int)
        hist, _, _ = np.histogram2d(x1, x2, bins=bins)
    else:
        raise NotImplementedError('Leaves distributions must be either discrete or continuous')

    m1, m2 = np.sum(hist, axis=1), np.sum(hist, axis=0)
    f1, f2 = np.count_nonzero(m1), np.count_nonzero(m2)
    dof = (f1 - 1) * (f2 - 1)

    g = 0.0
    for i, c1 in enumerate(m1):
        for j, c2 in enumerate(m2):
            c = hist[i, j]
            if c != 0:
                e = (c1 * c2) / n_samples
                g += c * np.log(c / e)
    g_val = 2.0 * g
    p_thresh = 2.0 * dof * p
    return g_val < p_thresh
