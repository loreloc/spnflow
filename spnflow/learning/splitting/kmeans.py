from sklearn import cluster


def kmeans(data, distributions, domains, n=2, n_init=3, max_iter=512):
    """
    Execute KMeans clustering on some data.

    :param data: The data.
    :param distributions: The data distributions (not used).
    :param domains: The data domains (not used).
    :param n: The number of clusters.
    :param n_init: The number of restarts.
    :param max_iter: Maximum number of iterations.
    :return: An array where each element is the cluster where the corresponding data belong.
    """
    return cluster.KMeans(n_clusters=n, n_init=n_init, max_iter=max_iter).fit_predict(data)
