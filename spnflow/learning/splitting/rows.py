import warnings

import numpy as np
from sklearn import mixture, cluster
from sklearn.exceptions import ConvergenceWarning

from spnflow.learning.splitting.rdc import rdc_rows
from spnflow.learning.splitting.random import random_rows
from spnflow.structure.leaf import LeafType
from spnflow.utils.data import ohe_data


def split_rows_clusters(data, clusters):
    """
    Split the data horizontally given the clusters.

    :param data: The data.
    :param clusters: The clusters.
    :return: (slices, weights) where slices is a list of partial data and weights is a list of proportions of the
             local data in respect to the original data.
    """
    slices = []
    weights = []
    n_samples = len(data)
    unique_clusters = np.unique(clusters)
    for c in unique_clusters:
        local_data = data[clusters == c, :]
        n_local_samples = len(local_data)
        slices.append(local_data)
        weights.append(n_local_samples / n_samples)
    return slices, weights


def get_split_rows_method(split_rows):
    """
    Get the rows splitting method given a string.

    :param split_rows: The string of the method do get.
    :return: The corresponding rows splitting function.
    """
    if split_rows == 'kmeans':
        return kmeans
    elif split_rows == 'gmm':
        return gmm
    elif split_rows == 'rdc':
        return rdc_rows
    elif split_rows == 'random':
        return random_rows
    else:
        raise NotImplementedError("Unknow split rows method called " + split_rows)


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
