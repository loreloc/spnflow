from spnflow.structure.leaf import Leaf
from spnflow.structure.node import bfs, Sum, Mul
from spnflow.utils.filter import get_nodes, filter_nodes_type


def get_statistics(root):
    """
    Compute some statistics of a SPN given its root.

    :param root: The root of the SPN.
    :return: A dictionary containing the following statistics:
             - n_nodes, the number of nodes
             - n_sum, the number of sum nodes
             - n_mul, the number of multiplication nodes
             - n_leaves, the number of leaves
             - n_edges, the number of edges
             - n_params, the number of parameters
             - depth, the depth of the network
    """
    stats = {}
    stats['n_nodes'] = len(get_nodes(root))
    stats['n_sum'] = len(filter_nodes_type(root, Sum))
    stats['n_mul'] = len(filter_nodes_type(root, Mul))
    stats['n_leaves'] = len(filter_nodes_type(root, Leaf))
    stats['n_edges'] = get_edges_count(root)
    stats['n_params'] = get_parameters_count(root)
    stats['depth'] = get_depth(root)
    return stats


def get_edges_count(root):
    """
    Get the number of edges of a SPN given its root.

    :param root: The root of the SPN.
    :return: The number of edges.
    """
    return sum([len(n.children) for n in filter_nodes_type(root, (Sum, Mul))])


def get_parameters_count(root):
    """
    Get the number of parameters of a SPN given its root

    :param root:  The root of the SPN.
    :return: The number of parameters.
    """
    n_weights = sum([len(n.weights) for n in filter_nodes_type(root, Sum)])
    n_leaf_params = sum([n.params_count() for n in filter_nodes_type(root, Leaf)])
    return n_weights + n_leaf_params


def get_depth(root):
    """
    Get the depth of the SPN given its root.

    :param root: The root of the SPN.
    :return: The depth of the network.
    """
    depths = {}

    def evaluate(node):
        d = depths.setdefault(node, 0)
        for c in node.children:
            depths[c] = d + 1

    bfs(root, evaluate)
    return max(depths.values())
