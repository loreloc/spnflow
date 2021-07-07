import numpy as np

from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.structure.node import Sum, Mul, bfs
from deeprob.spn.utils.validity import assert_is_valid


def eval_backward(root, lls):
    """
    Compute the log-gradients at each node.

    :param root: The root of the SPN.
    :param lls: The log-likelihoods at each node.
    :return: A dictionary having keys the node ids and values the log-gradients w.r.t. the input nodes.
    """
    assert_is_valid(root)
    n_nodes, n_samples = lls.shape
    grads = np.empty(shape=(n_nodes, n_samples), dtype=np.float32)
    parents = dict()

    # Initialize the identity log-gradient at root node
    grads[root.id] = 0.0

    def evaluate(node):
        if isinstance(node, Leaf):  # Skip leaves
            return
        parent_grad = 0.0 if node.id == root.id else grads[parents[node.id]]
        if isinstance(node, Sum):
            for c, w in zip(node.children, node.weights):
                parents[c.id] = node.id
                grads[c.id] = parent_grad + np.log(w)
        elif isinstance(node, Mul):
            for c in node.children:
                parents[c.id] = node.id
                grads[c.id] = parent_grad + lls[node.id] - lls[c.id]
        else:
            raise NotImplementedError(
                'Gradient evaluation not implemented for node of type {}'.format(node.__class__.__name__)
            )

    bfs(root, evaluate)
    return grads
