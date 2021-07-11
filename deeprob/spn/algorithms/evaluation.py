import numpy as np

from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.structure.node import Sum, Mul, topological_order
from deeprob.spn.utils.validity import assert_smooth, assert_decomposable, assert_labeled


def eval_bottom_up(root, x, node_func, return_results=False):
    """
    Evaluate the SPN bottom up given some inputs and leaves and nodes evaluation functions.

    :param root: The root of the SPN.
    :param x: The inputs.
    :param node_func: The function to compute at each node.
    :param return_results: A flag indicating if this function must return the log likelihoods of each node of the SPN.
    :return: The outputs. Additionally it returns the output of each node.
    """
    assert_smooth(root)
    assert_decomposable(root)
    assert_labeled(root)

    nodes = topological_order(root)
    assert nodes is not None, "The SPN Graph is not acyclic"

    n_samples, n_features = x.shape
    ls = np.empty((len(nodes), n_samples), dtype=np.float32)

    for node in reversed(nodes):
        if isinstance(node, Leaf):
            ls[node.id] = node_func(node, x[:, node.scope])
        else:
            children_ls = np.stack([ls[c.id] for c in node.children], axis=1)
            ls[node.id] = node_func(node, children_ls)

    if return_results:
        return ls[root.id], ls
    return ls[root.id]


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
    assert_smooth(root)
    assert_decomposable(root)
    assert_labeled(root)

    nodes = topological_order(root)
    assert nodes is not None, "The SPN Graph is not acyclic"

    n_samples, n_features = x.shape
    z = np.copy(x)

    # Build the array consisting of maximum masks
    max_masks = np.ones((len(nodes), n_samples), dtype=np.bool_)

    for node in nodes:
        if isinstance(node, Leaf):
            m = max_masks[node.id]
            mask = np.ix_(m, node.scope)
            z[mask] = leaf_func(node, x[mask])
        elif isinstance(node, Mul):
            for c in node.children:
                max_masks[c.id] = max_masks[node.id]
        elif isinstance(node, Sum):
            children_ls = np.stack([ls[c.id] for c in node.children], axis=1)
            max_branch = sum_func(node, children_ls)
            for i, c in enumerate(node.children):
                max_masks[c.id] = max_masks[node.id] & (max_branch == i)
        else:
            raise NotImplementedError(
                'Top down evaluation not implemented for node of type {}'.format(node.__class__.__name__)
            )
    return z
