import numpy as np

from deeprob.spn.learning.splitting.cluster import gmm, kmeans
from deeprob.spn.learning.splitting.rdc import rdc_rows
from deeprob.spn.learning.splitting.random import random_rows


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
        raise NotImplementedError("Unknown split rows method called " + split_rows)
