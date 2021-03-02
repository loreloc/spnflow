import warnings
from sklearn import cluster
from sklearn.exceptions import ConvergenceWarning


def kmeans(data, distributions, domains, n=2, n_init=5):
    """
    Execute KMeans clustering on some data.

    :param data: The data.
    :param distributions: The data distributions (not used).
    :param domains: The data domains (not used).
    :param n: The number of clusters.
    :param n_init: The number of restarts.
    :return: An array where each element is the cluster where the corresponding data belong.
    """
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)  # Ignore convergence warnings for K-Means
        return cluster.KMeans(n_clusters=n, n_init=n_init).fit_predict(data)
