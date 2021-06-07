from deeprob.structure.node import Node, bfs


def get_nodes(root):
    """
    Get all the nodes in a SPN.

    :param root: The root of the SPN.
    :return: A list of nodes.
    """
    return filter_nodes_type(root, Node)


def filter_nodes_type(root, ntype):
    """
    Get the nodes of a specified type in a SPN.

    :param root:  The root of the SPN.
    :param ntype: The node type.
    :return: A list of nodes of a specific type.
    """
    assert root is not None

    nodes = []

    def evaluate(node):
        if isinstance(node, ntype):
            nodes.append(node)

    bfs(root, evaluate)
    return nodes
