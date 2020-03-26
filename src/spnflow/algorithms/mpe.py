import numpy as np
from spnflow.structure.leaf import Leaf
from spnflow.structure.node import Sum, Mul, bfs
from spnflow.algorithms.inference import log_likelihood
from spnflow.utils.validity import assert_is_valid


def mpe(root, x):
    _, ls = log_likelihood(root, x, return_results=True)
    return eval_top_down(root, x, ls, lambda n, s: n.mode())


def eval_top_down(root, x, ls, leaf_func):
    assert_is_valid(root)
    assert np.all(np.any(np.isnan(x), axis=1)), "Each row must have at least a NaN value"

    x_len = len(x)
    result = np.array(x)
    nan_mask = np.isnan(x)
    masks = {root: np.repeat(True, x_len).reshape(-1, 1)}

    def evaluate(node):
        if isinstance(node, Leaf):
            m = masks[node]
            n = nan_mask[:, node.scope]
            p = np.logical_and(m, n).reshape(x_len)
            s = len(result[p, node.scope])
            result[p, node.scope] = leaf_func(node, s)
        elif isinstance(node, Mul):
            for c in node.children:
                masks[c] = np.copy(masks[node])
        elif isinstance(node, Sum):
            wcl = np.zeros((x_len, len(node.weights), 1))
            for i, c in enumerate(node.children):
                wcl[:, i] = ls[c] + np.log(node.weights[i])
            max_branch = np.argmax(wcl, axis=1)
            for i, c in enumerate(node.children):
                masks[c] = np.logical_and(masks[node], max_branch == i)
        else:
            raise NotImplementedError("MPE not implemented for node of type " + type(node).__name__)

    bfs(root, evaluate)
    return result

