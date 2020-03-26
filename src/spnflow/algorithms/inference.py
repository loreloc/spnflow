import numpy as np
from spnflow.structure.leaf import Leaf
from spnflow.structure.node import dfs_post_order


def likelihood(root, x, return_results=False):
    ls = {}
    x = np.array(x)
    m = np.isnan(x)

    def evaluate(node):
        if isinstance(node, Leaf):
            z = x[:, node.scope]
            sm = m[:, node.scope]
            v = np.ones(shape=(z.shape[0], 1))
            v[~sm] = node.likelihood(z[~sm])
            ls[node] = v
        else:
            z = np.concatenate([ls[n] for n in node.children], axis=1)
            ls[node] = node.likelihood(z)

    dfs_post_order(root, evaluate)

    if return_results:
        return ls[root], ls
    return ls[root]


def log_likelihood(root, x, return_results=False):
    ls = {}
    x = np.array(x)
    m = np.isnan(x)

    def evaluate(node):
        if isinstance(node, Leaf):
            z = x[:, node.scope]
            sm = m[:, node.scope]
            v = np.zeros(shape=(z.shape[0], 1))
            v[~sm] = node.log_likelihood(z[~sm])
            ls[node] = v
        else:
            z = np.concatenate([ls[n] for n in node.children], axis=1)
            ls[node] = node.log_likelihood(z)

    dfs_post_order(root, evaluate)

    if return_results:
        return ls[root], ls
    return ls[root]


