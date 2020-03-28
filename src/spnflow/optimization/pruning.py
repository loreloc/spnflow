from spnflow.structure.leaf import Leaf
from spnflow.structure.node import Sum, Mul, bfs, assign_ids
from spnflow.utils.validity import assert_is_valid


def prune(root):
    assert_is_valid(root)

    def evaluate(node):
        if isinstance(node, Leaf):
            return
        for i, c in enumerate(node.children):
            if len(c.children) == 1:
                node.children[i] = c.children[0]
                continue
            if type(node) == type(c):
                del node.children[i]
                node.children.extend(c.children)
                if isinstance(node, Sum):
                    weights = [w * node.weights[i] for w in c.weights]
                    node.weights.extend(weights)
                    del node.weights[i]

    bfs(root, evaluate)

    if len(root.children) == 1:
        root = root.children[0]

    return assign_ids(root)
