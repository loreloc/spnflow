import numpy as np
from spnflow.algorithms.inference import log_likelihood
from spnflow.algorithms.evaluation import eval_top_down


def mpe(root, x):
    """
    Compute the Maximum Posterior Estimate of a SPN given some inputs.

    :param root: The root of the SPN.
    :param x: The inputs (must have at least one NaN value).
    :return: The NaN-filled inputs.
    """
    _, ls = log_likelihood(root, x, return_results=True)
    return eval_top_down(root, x, ls, leaf_mpe, sum_mpe)


def leaf_mpe(node, size):
    """
    Compute the maximum likelihood estimate of a leaf node.

    :param node: The leaf node.
    :param size: The number of values to replace.
    :return: The maximum likelihood estimate.
    """
    return node.mode()


def sum_mpe(node, size, lc):
    """
    Choose the branch that maximize the posterior estimate likelihood.

    :param node: The sum node.
    :param size: The number of values.
    :param lc: The log likelihoods of the children nodes.
    :return: The branch that maximize the posterior estimate likelihood.
    """
    wcl = np.zeros((size, len(node.weights), 1))
    for i, l in enumerate(lc):
        wcl[:, i] = l + np.log(node.weights[i])
    return np.argmax(wcl, axis=1)
