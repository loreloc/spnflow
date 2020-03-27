import sklearn
import numpy as np


def split_rows_clusters(data, clusters):
    local_data = []
    proportions = []
    len_data = len(data)
    unique_clusters = np.unique(clusters)
    for c in unique_clusters:
        local_data.append(data[clusters == c, :])
        proportions.append(len(local_data) / len_data)
    return local_data, proportions


def get_split_rows_method(split_rows):
    if split_rows == 'kmeans':
        return kmeans
    elif split_rows == 'gmm':
        return gmm
    else:
        raise NotImplementedError("Unknow split rows method called " + split_rows)


def kmeans(data, k=2):
    return sklearn.cluster.KMeans(n_clusters=k).fit_predict(data)


def gmm(data, k=2):
    return sklearn.mixture.GaussianMixture(n_components=k).fit_predict(data)
