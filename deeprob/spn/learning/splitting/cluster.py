import warnings
import numpy as np
from sklearn import mixture, cluster
from sklearn.exceptions import ConvergenceWarning

from deeprob.spn.utils.data import ohe_data


def gmm(data, distributions, domains, n=2):
    """
    Execute GMM clustering on some data.

    :param data: The data.
    :param distributions: The data distributions.
    :param domains: The data domains.
    :param n: The number of clusters.
    :return: An array where each element is the cluster where the corresponding data belong.
    """
    # Convert the data using One Hot Encoding, in case of non-binary discrete features
    if any([len(d) > 2 for d in domains]):
        data = mixed_ohe_data(data, domains)

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
    # Convert the data using One Hot Encoding, in case of non-binary discrete features
    if any([len(d) > 2 for d in domains]):
        data = mixed_ohe_data(data, domains)

    # Apply K-Means
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)  # Ignore convergence warnings for K-Means
        return cluster.KMeans(n_clusters=n).fit_predict(data)


def mixed_ohe_data(data, domains):
    """
    One-Hot-Encoding function, applied on mixed data (both continuous and non-binary discrete).

    :param data: The 2D data to encode.
    :param domains: The domains to use.
    :return: The One Hot encoded data.
    """
    n_samples, n_features = data.shape
    ohe = []
    for i in range(n_features):
        if len(domains[i]) > 2:
            ohe.append(ohe_data(data[:, i], domains[i]))
        else:
            ohe.append(data[:, i])
    return np.column_stack(ohe)
