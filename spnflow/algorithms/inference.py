import numpy as np
from spnflow.algorithms.evaluation import eval_bottom_up


def likelihood(root, x, return_results=False):
    """
    Compute the likelihoods of the SPN given some inputs.

    :param root: The root of the SPN.
    :param x: The inputs.
    :param return_results: A flag indicating if this function must return the likelihoods of each node of the SPN.
    :return: The likelihood values. Additionally it returns the likelihood values of each node.
    """
    return eval_bottom_up(root, x, leaf_likelihood, node_likelihood, return_results)


def log_likelihood(root, x, return_results=False):
    """
    Compute the logarithmic likelihoods of the SPN given some inputs.

    :param root: The root of the SPN.
    :param x: The inputs.
    :param return_results: A flag indicating if this function must return the log likelihoods of each node of the SPN.
    :return: The log likelihood values. Additionally it returns the log likelihood values of each node.
    """
    return eval_bottom_up(root, x, leaf_log_likelihood, node_log_likelihood, return_results)


def leaf_likelihood(node, x, m):
    """
    Compute the likelihood of a leaf given an input and its NaN mask.
    It also handles of NaN and infinite likelihoods.

    :param node: The leaf node.
    :param x: The input.
    :param m: The NaN mask of the input.
    :return: The likelihood of the leaf given the input.
    """
    z = np.ones(shape=(x.shape[0], 1))
    z[~m] = node.likelihood(x[~m])
    z[np.isnan(z)] = 1.0
    z[np.isinf(z)] = 0.0
    return z


def leaf_log_likelihood(node, x, m):
    """
    Compute the logarithmic likelihood of a leaf given an input and its NaN mask.
    It also handles of NaN and infinite log likelihoods.

    :param node: The leaf node.
    :param x: The input.
    :param m: The NaN mask of the input.
    :return: The log likelihood of the leaf given the input.
    """
    z = np.zeros(shape=(x.shape[0], 1))
    z[~m] = node.log_likelihood(x[~m])
    z[np.isnan(z)] = 0.0
    z[np.isinf(z)] = np.finfo(float).min
    return z


def node_likelihood(node, lc):
    """
    Compute the likelihood of a node given the list of likelihoods of its children.
    It also handles of NaN and infinite likelihoods.

    :param node: The internal node.
    :param lc: The list of likelihoods of the children.
    :return: The likelihood of the node given the inputs.
    """
    x = np.concatenate(lc, axis=1)
    z = node.likelihood(x)
    z[np.isnan(z)] = 1.0
    z[np.isinf(z)] = 0.0
    return z


def node_log_likelihood(node, lc):
    """
    Compute the logarithmic likelihood of a node given the list of logarithmic likelihoods of its children.
    It also handles of NaN and infinite log likelihoods.

    :param node: The internal node.
    :param lc: The list of log likelihoods of the children.
    :return: The log likelihood of the node given the inputs.
    """
    x = np.concatenate(lc, axis=1)
    z = node.log_likelihood(x)
    z[np.isnan(z)] = 0.0
    z[np.isinf(z)] = np.finfo(float).min
    return z
