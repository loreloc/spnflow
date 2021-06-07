import numpy as np

from deeprob.learning.splitting.gvs import gvs_cols
from deeprob.learning.splitting.rdc import rdc_cols
from deeprob.learning.splitting.random import random_cols


def split_cols_clusters(data, clusters, scope):
    """
    Split the data vertically given the clusters.

    :param data: The data.
    :param clusters: The clusters.
    :param scope: The original scope.
    :return: (slices, scopes) where slices is a list of partial data and scopes is a list of partial scopes.
    """
    slices = []
    scopes = []
    scope = np.asarray(scope)
    unique_clusters = np.unique(clusters)
    for c in unique_clusters:
        cols = clusters == c
        local_data = data[:, cols]
        slices.append(local_data)
        scopes.append(scope[cols].tolist())
    return slices, scopes


def get_split_cols_method(split_cols):
    """
    Get the columns splitting method given a string.

    :param split_cols: The string of the method do get.
    :return: The corresponding columns splitting function.
    """
    if split_cols == 'gvs':
        return gvs_cols
    elif split_cols == 'rdc':
        return rdc_cols
    elif split_cols == 'random':
        return random_cols
    else:
        raise NotImplementedError("Unknown split rows method called " + split_cols)
