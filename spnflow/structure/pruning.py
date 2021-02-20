from spnflow.structure.leaf import Leaf
from spnflow.structure.node import Sum, bfs, assign_ids
from spnflow.utils.validity import assert_is_valid


def prune(root):
    """
    Prune (or simplify) the given SPN to a minimal and equivalent SPN.

    :param root: The root of the SPN.
    :return: A minimal and equivalent SPN.
    """
    assert_is_valid(root)

    def evaluate(node):
        if isinstance(node, Leaf):
            return
        i = 0
        while i < len(node.children):
            c = node.children[i]
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
                continue
            i += 1

    bfs(root, evaluate)

    if len(root.children) == 1:
        root = root.children[0]

    return assign_ids(root)
