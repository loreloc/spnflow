import warnings
from sklearn import cluster
from sklearn.exceptions import ConvergenceWarning

from spnflow.utils.data import mixed_ohe_data


def kmeans(data, distributions, domains, n=2):
    """
    Execute KMeans clustering on some data.

    :param data: The data.
    :param distributions: The data distributions.
    :param domains: The data domains.
    :param n: The number of clusters.
    :return: An array where each element is the cluster where the corresponding data belong.
    """
    # Convert the data using One Hot Encoding
    data = mixed_ohe_data(data, distributions, domains)
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)  # Ignore convergence warnings for K-Means
        return cluster.KMeans(n_clusters=n).fit_predict(data)
