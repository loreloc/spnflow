from spnflow.structure.node import Node, bfs


def get_nodes(root):
    return filter_nodes_type(root, Node)


def filter_nodes_type(root, ntype):
    assert root is not None
    nodes = []

    def evaluate(node):
        if isinstance(node, ntype):
            nodes.append(node)

    bfs(root, evaluate)
    return nodes


