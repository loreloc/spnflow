from spnflow.structure.leaf import Leaf
from spnflow.structure.node import bfs, Sum, Mul
from spnflow.utils.filter import get_nodes, filter_nodes_type


def get_statistics(root):
    stats = {}
    stats['n_nodes'] = len(get_nodes(root))
    stats['n_sum'] = len(filter_nodes_type(root, Sum))
    stats['n_mul'] = len(filter_nodes_type(root, Mul))
    stats['n_leaf'] = len(filter_nodes_type(root, Leaf))
    stats['n_edges'] = get_edges_count(root)
    stats['n_params'] = get_parameters_count(root)
    stats['depth'] = get_depth(root)
    return stats


def get_edges_count(root):
    return sum([len(n.children) for n in filter_nodes_type(root, (Sum, Mul))])


def get_parameters_count(root):
    n_weights = sum([len(n.weights) for n in filter_nodes_type(root, Sum)])
    n_leaf_params = sum([n.params_count() for n in filter_nodes_type(root, Leaf)])
    return n_weights + n_leaf_params


def get_depth(root):
    depths = {}

    def evaluate(node):
        d = depths.setdefault(node, 1)
        for c in node.children:
            depths[c] = d + 1

    bfs(root, evaluate)
    return max(depths.values()) - 1
