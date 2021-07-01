import numpy as np
from tqdm import tqdm

from deeprob.spn.structure.leaf import LeafType
from deeprob.spn.structure.node import Sum, assign_ids
from deeprob.spn.structure.pruning import prune
from deeprob.spn.learning.structure import learn_structure


def learn_estimator(data, distributions, domains=None, **kwargs):
    """
    Learn a SPN density estimator given some training data, the features distributions and domains.

    :param data: The training data.
    :param distributions: A list of distribution classes (one for each feature).
    :param domains: A list of domains (one for each feature).
    :param kwargs: Other parameters for structure learning.
    :return: A learned valid and optimized SPN.
    """
    assert data is not None
    assert distributions is not None

    if domains is None:
        domains = get_data_domains(data, distributions)

    root = learn_structure(data, distributions, domains, **kwargs)
    return prune(root)


def learn_classifier(data, distributions, domains=None, class_idx=-1, verbose=True, **kwargs):
    """
    Learn a SPN classifier given some training data, the features distributions and domains and
    the class index in the training data.

    :param data: The training data.
    :param distributions: A list of distribution classes (one for each feature).
    :param domains: A list of domains (one for each feature).
    :param class_idx: The index of the class feature in the training data.
    :param verbose: Whether to enable verbose mode.
    :param kwargs: Other parameters for structure learning.
    :return: A learned valid and optimized SPN.
    """
    assert data is not None
    assert distributions is not None

    if domains is None:
        domains = get_data_domains(data, distributions)

    n_samples, n_features = data.shape
    classes = data[:, class_idx]

    # Initialize the tqdm wrapped unique classes array, if verbose is enabled
    if verbose:
        tk = tqdm(np.unique(classes), bar_format='{l_bar}{bar:32}{r_bar}')
    else:
        tk = np.unique(classes)

    # Learn each sub-spn's structure individually
    weights = []
    children = []
    for c in tk:
        local_data = data[classes == c]
        n_local_samples, _ = local_data.shape
        weight = n_local_samples / n_samples
        branch = learn_structure(local_data, distributions, domains, verbose=verbose, **kwargs)
        weights.append(weight)
        children.append(prune(branch))

    root = Sum(weights, children)
    return assign_ids(root)


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
        vmin = np.min(col)
        vmax = np.max(col)
        if d.LEAF_TYPE == LeafType.DISCRETE:
            domains.append(list(range(vmax.astype(int) + 1)))
        elif d.LEAF_TYPE == LeafType.CONTINUOUS:
            domains.append([vmin, vmax])
        else:
            raise NotImplementedError("Domain for leaf type " + d.LEAF_TYPE.__name__ + " not implemented")
    return domains
