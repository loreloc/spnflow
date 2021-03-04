import warnings
from sklearn import mixture
from sklearn.exceptions import ConvergenceWarning

from spnflow.structure.leaf import LeafType
from spnflow.utils.data import mixed_ohe_data


def gmm(data, distributions, domains, n=2):
    """
    Execute GMM clustering on some data.

    :param data: The data.
    :param distributions: The data distributions.
    :param domains: The data domains.
    :param n: The number of clusters.
    :return: An array where each element is the cluster where the corresponding data belong.
    """
    # Convert the data using One Hot Encoding, if needed
    if any([d.LEAF_TYPE == LeafType.DISCRETE for d in distributions]):
        data = mixed_ohe_data(data, distributions, domains)

    # Apply GMM
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)  # Ignore convergence warnings for GMM
        return mixture.GaussianMixture(n_components=n, n_init=3).fit_predict(data)
