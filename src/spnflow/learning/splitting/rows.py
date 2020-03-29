import numpy as np
from spnflow.learning.splitting.kmeans import kmeans
from spnflow.learning.splitting.gmm import gmm


def split_rows_clusters(data, clusters):
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
    if split_rows == 'kmeans':
        return kmeans
    elif split_rows == 'gmm':
        return gmm
    else:
        raise NotImplementedError("Unknow split rows method called " + split_rows)
