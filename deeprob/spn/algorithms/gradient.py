import numpy as np

from collections import defaultdict
from scipy.special import logsumexp
from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.structure.node import Sum, Mul, topological_order
from deeprob.spn.utils.validity import assert_smooth, assert_decomposable, assert_labeled


def eval_backward(root, lls):
    """
    Compute the log-gradients at each node.

    :param root: The root of the SPN.
    :param lls: The log-likelihoods at each node.
    :return: A dictionary having keys the node ids and values the log-gradients w.r.t. the input nodes.
    """
    assert_smooth(root)
    assert_decomposable(root)
    assert_labeled(root)

    nodes = topological_order(root)
    assert nodes is not None, "The SPN Graph is not acyclical"

    n_nodes, n_samples = lls.shape
    assert n_nodes == len(nodes), "Incompatible log-likelihoods at each node"

    # Initialize the log-gradients array and the cached log-gradients dictionary of lists
    grads = np.empty(shape=(n_nodes, n_samples), dtype=np.float32)
    cached_grads = defaultdict(list)

    # Initialize the identity log-gradient at root node
    grads[root.id] = 0.0

    for node in nodes:
        # Compute log-gradient at the underlying node by logsumexp
        # Note that at this point of topological ordering, the node have no incoming arcs
        # Hence, we can finally compute the log-gradients w.r.t. this node
        if node.id != root.id:
            grads[node.id] = logsumexp(cached_grads[node.id], axis=0)
            del cached_grads[node.id]  # Cached log-gradients no longer necessary

        if isinstance(node, Sum):
            for c, w in zip(node.children, node.weights):
                g = grads[node.id] + np.log(w)
                cached_grads[c.id].append(g)
        elif isinstance(node, Mul):
            for c in node.children:
                g = grads[node.id] + lls[node.id] - lls[c.id]
                cached_grads[c.id].append(g)
        elif isinstance(node, Leaf):
            pass  # Leaves have no children
        else:
            raise NotImplementedError(
                'Gradient evaluation not implemented for node of type {}'.format(node.__class__.__name__)
            )
    return grads
