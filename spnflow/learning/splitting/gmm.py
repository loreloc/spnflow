from sklearn import mixture


def gmm(data, distributions, domains, n=2):
    """
    Execute GMM clustering on some data.

    :param data: The data.
    :param distributions: The data distributions (not used).
    :param domains: The data domains (not used).
    :param n: The number of clusters.
    :return: An array where each element is the cluster where the corresponding data belong.
    """
    return mixture.GaussianMixture(n_components=n).fit_predict(data)
