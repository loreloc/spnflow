import numpy as np
from spnflow.learning.splitting.rdc import rdc


def split_cols_clusters(data, clusters, scope):
    slices = []
    scopes = []
    np_scope = np.asarray(scope)
    unique_clusters = np.unique(clusters)
    for c in unique_clusters:
        cols = clusters == c
        local_data = data[:, cols]
        slices.append(local_data)
        scopes.append(np_scope[cols].tolist())
    return slices, scopes


def get_split_cols_method(split_cols):
    if split_cols == 'rdc':
        return rdc
    else:
        raise NotImplementedError("Unknow split rows method called " + split_cols)
