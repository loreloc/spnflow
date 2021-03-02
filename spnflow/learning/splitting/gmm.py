import warnings
from sklearn import mixture
from sklearn.exceptions import ConvergenceWarning


def gmm(data, distributions, domains, n=2):
    """
    Execute GMM clustering on some data.

    :param data: The data.
    :param distributions: The data distributions (not used).
    :param domains: The data domains (not used).
    :param n: The number of clusters.
    :return: An array where each element is the cluster where the corresponding data belong.
    """
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)  # Ignore convergence warnings for GMM
        return mixture.GaussianMixture(n_components=n).fit_predict(data)
