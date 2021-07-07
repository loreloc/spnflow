import numpy as np

from deeprob.spn.algorithms.evaluation import eval_bottom_up, eval_top_down


def likelihood(root, x, return_results=False):
    """
    Compute the likelihoods of the SPN given some inputs.

    :param root: The root of the SPN.
    :param x: The inputs. They can be marginalized using NaNs.
    :param return_results: A flag indicating if this function must return the likelihoods of each node of the SPN.
    :return: The likelihood values. Additionally it returns the likelihood values of each node.
    """
    return eval_bottom_up(root, x, node_likelihood, return_results)


def log_likelihood(root, x, return_results=False):
    """
    Compute the logarithmic likelihoods of the SPN given some inputs.

    :param root: The root of the SPN.
    :param x: The inputs. They can be marginalized using NaNs.
    :param return_results: A flag indicating if this function must return the log likelihoods of each node of the SPN.
    :return: The log likelihood values. Additionally it returns the log likelihood values of each node.
    """
    return eval_bottom_up(root, x, node_log_likelihood, return_results)


def mpe(root, x):
    """
    Compute the Maximum Posterior Estimate of a SPN given some inputs.

    :param root: The root of the SPN.
    :param x: The inputs. They can be marginalized using NaNs.
    :return: The NaN-filled inputs.
    """
    _, ls = log_likelihood(root, x, return_results=True)
    return eval_top_down(root, x, ls, leaf_mpe, sum_mpe)


def node_likelihood(node, x):
    """
    Compute the likelihood of a node given the list of likelihoods of its children.
    It also handles of NaN and infinite likelihoods.

    :param node: The internal node.
    :param x: The array of likelihoods of the children.
    :return: The likelihood of the node given the inputs.
    """
    z = node.likelihood(x)
    z[np.isnan(z)] = 1.0
    z[np.isinf(z)] = 0.0
    return np.squeeze(z)


def node_log_likelihood(node, x):
    """
    Compute the logarithmic likelihood of a node given the list of logarithmic likelihoods of its children.
    It also handles of NaN and infinite log likelihoods.

    :param node: The internal node.
    :param x: The array of log likelihoods of the children.
    :return: The log likelihood of the node given the inputs.
    """
    z = node.log_likelihood(x)
    z[np.isnan(z)] = 0.0
    z[np.isinf(z)] = np.finfo(np.float16).min
    return np.squeeze(z)


def leaf_mpe(node, x):
    """
    Compute the maximum likelihood estimate of a leaf node.

    :param node: The leaf node.
    :param x: Some evidence.
    :return: The maximum likelihood estimate.
    """
    return node.mpe(x)


def sum_mpe(node, lc):
    """
    Choose the branch that maximize the posterior estimate likelihood.

    :param node: The sum node.
    :param lc: The log likelihoods of the children nodes.
    :return: The branch that maximize the posterior estimate likelihood.
    """
    weighted_ls = lc + np.log(node.weights)
    return np.argmax(weighted_ls, axis=1)
