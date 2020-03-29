import numpy as np
from spnflow.structure.leaf import Leaf
from spnflow.structure.node import dfs_post_order
from spnflow.utils.validity import assert_is_valid


def likelihood(root, x, return_results=False):
    return eval_bottom_up(root, x, leaf_likelihood, node_likelihood, return_results)


def log_likelihood(root, x, return_results=False):
    return eval_bottom_up(root, x, leaf_log_likelihood, node_log_likelihood, return_results)


def eval_bottom_up(root, x, leaf_func, node_func, return_results=False):
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


def leaf_likelihood(node, x, m):
    z = np.ones(shape=(x.shape[0], 1))
    z[~m] = node.likelihood(x[~m])
    z[np.isnan(z)] = 1.0
    z[np.isinf(z)] = np.finfo(float).eps
    return z


def leaf_log_likelihood(node, x, m):
    z = np.zeros(shape=(x.shape[0], 1))
    z[~m] = node.log_likelihood(x[~m])
    z[np.isnan(z)] = 0.0
    z[np.isinf(z)] = np.finfo(float).min
    return z


def node_likelihood(node, lc):
    x = np.concatenate(lc, axis=1)
    z = node.likelihood(x)
    z[np.isnan(z)] = 1.0
    z[np.isinf(z)] = np.finfo(float).eps
    return z


def node_log_likelihood(node, lc):
    x = np.concatenate(lc, axis=1)
    z = node.log_likelihood(x)
    z[np.isnan(z)] = 0.0
    z[np.isinf(z)] = np.finfo(float).min
    return z
