import numpy as np
import scipy.stats as stats

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
    One-Hot-Encoding function.
    :param data: The 1D data to encode.
    :param domain: The domain to use.
    :return: The One Hot encoded data.
    """
    n_samples = len(data)
    ohe = np.zeros((n_samples, len(domain)))
    ohe[np.equal.outer(data, domain)] = 1
    return ohe


def ecdf_data(data):
    """
    Empirical Cumulative Distribution Function (ECDF).
    :param data: The data.
    :return: The result of the ECDF on data.
    """
    return stats.rankdata(data, method='max') / len(data)


def compute_mean_quantiles(data, n_quantiles):
    """
    Compute the mean quantiles of a dataset (Poon-Domingos).

    :param data: The data.
    :param n_quantiles: The number of quantiles.
    :return: The mean quantiles.
    """
    # Split the dataset in quantiles regions
    data = np.sort(data, axis=0)
    values_per_quantile = np.array_split(data, n_quantiles, axis=0)

    # Compute the mean quantiles
    mean_per_quantiles = [np.mean(x, axis=0) for x in values_per_quantile]
    return np.stack(mean_per_quantiles, axis=0)
