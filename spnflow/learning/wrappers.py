import numpy as np
from tqdm import tqdm

from spnflow.structure.node import Sum, assign_ids
from spnflow.learning.structure import learn_structure
from spnflow.utils.data import get_data_domains
from spnflow.optimization.pruning import prune


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


def learn_classifier(data, distributions, domains=None, class_idx=-1, **kwargs):
    """
    Learn a SPN classifier given some training data, the features distributions and domains and
    the class index in the training data.

    :param data: The training data.
    :param distributions: A list of distribution classes (one for each feature).
    :param domains: A list of domains (one for each feature).
    :param class_idx: The index of the class feature in the training data.
    :param kwargs: Other parameters for structure learning.
    :return: A learned valid and optimized SPN.
    """
    assert data is not None
    assert distributions is not None

    if domains is None:
        domains = get_data_domains(data, distributions)

    n_samples, n_features = data.shape
    classes = data[:, class_idx]

    def learn_branch(local_data, **kwargs):
        n_local_samples, _ = local_data.shape
        weight = n_local_samples / n_samples
        local_spn = learn_structure(local_data, distributions, domains, **kwargs)
        return weight, local_spn

    weights = []
    children = []
    tk = tqdm(np.unique(classes), bar_format='{l_bar}{bar:32}{r_bar}')
    for c in tk:
        weight, branch = learn_branch(data[classes == c], **kwargs, id_bar=1)
        weights.append(weight)
        children.append(prune(branch))

    root = Sum(weights, children)
    return assign_ids(root)
