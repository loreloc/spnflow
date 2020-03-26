import numpy as np
from spnflow.structure.leaf import Leaf
from spnflow.structure.node import Sum, Mul, bfs
from spnflow.algorithms.inference import log_likelihood


def sample(root, x):
    assert np.all(np.any(np.isnan(x), axis=1)), "Each row must have at least a NaN value"
    x_len = len(x)
    result = np.array(x)
    nan_mask = np.isnan(x)
    masks = {root: np.repeat(True, x_len).reshape(-1, 1)}
    _, ls = log_likelihood(root, x, return_results=True)

    def evaluate(node):
        if isinstance(node, Leaf):
            m = masks[node]
            n = nan_mask[:, node.scope]
            p = np.logical_and(m, n).reshape(x_len)
            s = len(result[p, node.scope])
            result[p, node.scope] = node.sample(s)
        elif isinstance(node, Mul):
            for c in node.children:
                masks[c] = np.copy(masks[node])
        elif isinstance(node, Sum):
            wcl = np.zeros((x_len, len(node.weights), 1))
            for i, c in enumerate(node.children):
                wcl[:, i] = ls[c] + np.log(node.weights[i])
            max_branch = np.argmax(wcl, axis=1)
            for i, c in enumerate(node.children):
                masks[c] = max_branch == i
        else:
            raise NotImplementedError("MPE not implemented for node of type " + str(type(node)))

    bfs(root, evaluate)
    return result
