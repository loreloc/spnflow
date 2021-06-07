import numpy as np

from deeprob.utils.filter import get_nodes
from deeprob.structure.leaf import Leaf
from deeprob.structure.node import Sum, Mul, bfs, dfs_post_order
from deeprob.utils.validity import assert_is_valid


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
    n_samples, n_features = x.shape
    n_nodes = len(get_nodes(root))
    ls = np.empty(shape=(n_nodes, n_samples), dtype=np.float32)

    def evaluate(node):
        if isinstance(node, Leaf):
            ls[node.id] = leaf_func(node, x[:, node.scope])
        else:
            children_ls = np.stack([ls[c.id] for c in node.children], axis=1)
            ls[node.id] = node_func(node, children_ls)

    dfs_post_order(root, evaluate)

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
    assert_is_valid(root)
    n_samples, n_features = x.shape
    n_nodes = len(get_nodes(root))
    x_mpe = np.copy(x)

    # Build the dictionary consisting of maximum masks
    max_masks = np.ones(shape=(n_nodes, n_samples), dtype=np.bool_)

    def evaluate(node):
        if isinstance(node, Leaf):
            m = max_masks[node.id]
            mask = np.ix_(m, node.scope)
            x_mpe[mask] = leaf_func(node, x[mask])
        elif isinstance(node, Mul):
            for c in node.children:
                max_masks[c.id] = max_masks[node.id]
        elif isinstance(node, Sum):
            children_ls = np.stack([ls[c.id] for c in node.children], axis=1)
            max_branch = sum_func(node, children_ls)
            for i, c in enumerate(node.children):
                max_masks[c.id] = max_masks[node.id] & (max_branch == i)
        else:
            raise NotImplementedError("Top down evaluation not implemented for node of type " + node.__class__.__name__)

    bfs(root, evaluate)
    return x_mpe
