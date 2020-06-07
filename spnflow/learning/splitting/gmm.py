from sklearn import mixture


def gmm(data, n=2):
    """
    Execute GMM clustering on some data.

    :param data: The data.
    :param n: The number of clusters.
    :return: An array where each element is the cluster where the corresponding data belong.
    """
    return mixture.GaussianMixture(n_components=n).fit_predict(data)
