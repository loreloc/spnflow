import numpy as np
from spnflow.structure.leaf import Leaf
from spnflow.structure.node import Sum, Mul, bfs, dfs_post_order
from spnflow.utils.validity import assert_is_valid


def eval_bottom_up(root, x, leaf_func, node_func, return_results=False):
    """
    Evaluate the SPN bottom up given some inputs and leaves and nodes evaluation functions.

    :param root: The root of the SPN.
    :param x: The inputs.
    :param leaf_func: The function to compute at the leaves.
    :param node_func: The function to compute at each internal node.
    :param return_results: A flag indicating if this function must return the log likelihoods of each node of the SPN.
    :return: The outputs. Additionally it returns the output of each node.
    """
    assert_is_valid(root)

    ls = {}
    x = np.array(x)
    m = np.isnan(x)

    def evaluate(node):
        if isinstance(node, Leaf):
            ls[node] = leaf_func(node, x[:, node.scope], m[:, node.scope])
        else:
            ls[node] = node_func(node, [ls[c] for c in node.children])

    dfs_post_order(root, evaluate)

    if return_results:
        return ls[root], ls
    return ls[root]


def eval_top_down(root, x, ls, leaf_func, sum_func):
    """
    Evaluate the SPN top down given some inputs, the likelihoods of each node and a leaves evaluation function.
    The leaves to evaluate are chosen by following the nodes having maximum likelihood top down.

    :param root: The root of the SPN.
    :param x: The inputs (must have at least one NaN value).
    :param ls: The likelihoods of each node.
    :param leaf_func: The leaves evaluation function.
    :param sum_func: The sum node evaluation function.
    :return: The NaN-filled inputs.
    """
    assert_is_valid(root)
    assert np.all(np.any(np.isnan(x), axis=1)), "Each row must have at least a NaN value"

    x_len = len(x)
    result = np.array(x)
    nan_mask = np.isnan(x)
    max_masks = {root: np.full((x_len, 1), True)}

    def evaluate(node):
        if isinstance(node, Leaf):
            m = max_masks[node]
            n = nan_mask[:, node.scope]
            p = np.logical_and(m, n).reshape(x_len)
            s = len(result[p, node.scope])
            result[p, node.scope] = leaf_func(node, s)
        elif isinstance(node, Mul):
            for c in node.children:
                max_masks[c] = np.copy(max_masks[node])
        elif isinstance(node, Sum):
            max_branch = sum_func(node, x_len, [ls[c] for c in node.children])
            for i, c in enumerate(node.children):
                max_masks[c] = np.logical_and(max_masks[node], max_branch == i)
        else:
            raise NotImplementedError("Top down evaluation not implemented for node of type " + type(node).__name__)

    bfs(root, evaluate)
    return result
