from sklearn import cluster


def kmeans(data, distributions, domains, n=2):
    """
    Execute KMeans clustering on some data.

    :param data: The data.
    :param n: The number of clusters.
    :param distributions: The data distributions (not used).
    :param domains: The data domains (not used).
    :return: An array where each element is the cluster where the corresponding data belong.
    """
    return cluster.KMeans(n_clusters=n).fit_predict(data)
