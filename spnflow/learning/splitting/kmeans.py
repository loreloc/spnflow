from sklearn import cluster


def kmeans(data, n=2):
    """
    Execute KMeans clustering on some data.

    :param data: The data.
    :param n: The number of clusters.
    :return: An array where each element is the cluster where the corresponding data belong.
    """
    return cluster.KMeans(n_clusters=n).fit_predict(data)
