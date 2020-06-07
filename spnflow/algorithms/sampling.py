import numpy as np
import scipy.stats as stats
from spnflow.algorithms.inference import log_likelihood
from spnflow.algorithms.evaluation import eval_top_down


def sample(root, x):
    """
    Sample some features from the distribution represented by the SPN.

    :param root: The root of the SPN.
    :param x: The inputs (must have at least one NaN value where to put the sample).
    :return: The inputs that are NaN-filled with samples from appropriate distributions.
    """
    _, ls = log_likelihood(root, x, return_results=True)
    return eval_top_down(root, x, ls, leaf_sample, sum_sample)


def leaf_sample(node, size):
    """
    Sample some values from the distribution leaf.

    :param node: The distribution leaf node.
    :param size: The number of samples.
    :return: Some samples.
    """
    return node.sample(size)


def sum_sample(node, size, lc):
    """
    Choose the sub-distribution from which sample.

    :param node: The sum node.
    :param size: The number of samples.
    :param lc: The log likelihoods of the children of the sum node.
    :return: The index of the sub-distribution to follow.
    """
    n_weights = len(node.weights)
    wcl = np.zeros((size, n_weights, 1))
    for i, l in enumerate(lc):
        r = stats.gumbel_l.rvs(0.0, 1.0, size=(size, 1))
        wcl[:, i] = l + np.log(node.weights[i]) + r
    return np.argmax(wcl, axis=1)
