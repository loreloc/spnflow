import warnings
import numpy as np
from sklearn import mixture, cluster
from sklearn.exceptions import ConvergenceWarning

from spnflow.structure.leaf import LeafType
from spnflow.utils.data import ohe_data


def mixed_ohe_data(data, distributions, domains):
    """
    One-Hot-Encoding function, applied on mixed data (both continuous and discrete).

    :param data: The 2D data to encode.
    :param distributions: The given distributions.
    :param domains: The domains to use.
    :return: The One Hot encoded data.
    """
    n_samples, n_features = data.shape
    ohe = []
    for i in range(n_features):
        if distributions[i].LEAF_TYPE == LeafType.DISCRETE:
            ohe.append(ohe_data(data[:, i], domains[i]))
        else:
            ohe.append(data[:, i])
    return np.column_stack(ohe)


def gmm(data, distributions, domains, n=2):
    """
    Execute GMM clustering on some data.

    :param data: The data.
    :param distributions: The data distributions.
    :param domains: The data domains.
    :param n: The number of clusters.
    :return: An array where each element is the cluster where the corresponding data belong.
    """
    # Convert the data using One Hot Encoding, if needed
    if any([d.LEAF_TYPE == LeafType.DISCRETE for d in distributions]):
        data = mixed_ohe_data(data, distributions, domains)

    # Apply GMM
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)  # Ignore convergence warnings for GMM
        return mixture.GaussianMixture(n_components=n, n_init=3).fit_predict(data)


def kmeans(data, distributions, domains, n=2):
    """
    Execute KMeans clustering on some data.

    :param data: The data.
    :param distributions: The data distributions.
    :param domains: The data domains.
    :param n: The number of clusters.
    :return: An array where each element is the cluster where the corresponding data belong.
    """
    # Convert the data using One Hot Encoding, if needed
    if any([d.LEAF_TYPE == LeafType.DISCRETE for d in distributions]):
        data = mixed_ohe_data(data, distributions, domains)

    # Apply K-Means
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)  # Ignore convergence warnings for K-Means
        return cluster.KMeans(n_clusters=n).fit_predict(data)
