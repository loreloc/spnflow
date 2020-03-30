from spnflow.algorithms.mpe import eval_top_down
from spnflow.algorithms.inference import log_likelihood


def sample(root, x):
    """
    Sample some features from the distribution represented by the SPN.

    :param root: The root of the SPN.
    :param x: The inputs (must have at least one NaN value where to put the sample).
    :return: The inputs that are NaN-filled with samples from appropriate distributions.
    """
    _, ls = log_likelihood(root, x, return_results=True)
    return eval_top_down(root, x, ls, lambda n, s: n.sample(s))
