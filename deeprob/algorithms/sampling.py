import numpy as np
import scipy.stats as stats
from deeprob.algorithms.inference import log_likelihood
from deeprob.algorithms.evaluation import eval_top_down


def sample(root, x):
    """
    Sample some features from the distribution represented by the SPN.

    :param root: The root of the SPN.
    :param x: The inputs with possible NaN values to fill with sampled values.
    :return: The inputs that are NaN-filled with samples from appropriate distributions.
    """
    # Firstly, evaluate the SPN bottom-up, then top-down
    _, ls = log_likelihood(root, x, return_results=True)
    return eval_top_down(root, x, ls, leaf_sample, sum_sample)


def leaf_sample(node, x):
    """
    Sample some values from the distribution leaf.

    :param node: The distribution leaf node.
    :param x: The inputs with possible NaN values to fill with sampled values.
    :return: The completed samples.
    """
    return node.sample(x)


def sum_sample(node, lc):
    """
    Choose the sub-distribution from which sample.

    :param node: The sum node.
    :param lc: The log likelihoods of the children nodes.
    :return: The index of the sub-distribution to follow.
    """
    n_samples, n_features = lc.shape
    gumbel = stats.gumbel_l.rvs(0.0, 1.0, size=(n_samples, n_features))
    weighted_ls = lc + np.log(node.weights) + gumbel
    return np.argmax(weighted_ls, axis=1)
