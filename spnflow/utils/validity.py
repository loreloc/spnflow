import numpy as np
from spnflow.structure.node import Sum, Mul
from spnflow.utils.filter import get_nodes, filter_nodes_type


def assert_is_valid(root):
    """
    Assert if the SPN is valid.

    :param root: The SPN root.
    """
    v, msg = is_valid(root)
    assert v, "SPN not valid: " + msg


def is_valid(root):
    """
    Check if the SPN is valid.

    :param root: The SPN root.
    :return: (True, None) if the SPN is valid;
             (False, reason) otherwise.
    """
    valid, msg = is_complete(root)
    if not valid:
        return valid, msg

    valid, msg = is_consistent(root)
    if not valid:
        return valid, msg

    valid, msg = is_labeled(root)
    if not valid:
        return valid, msg

    return True, None


def is_complete(root):
    """
    Check if the SPN is complete.
    It checks that each child of a sum node has the same scope.
    Furthermore, it checks that the sum of the weights of a sum node is close to 1.

    :param root: The root of the SPN.
    :return: (True, None) if the SPN is complete;
             (False, reason) otherwise.
    """
    for n in filter_nodes_type(root, Sum):
        if not np.isclose(sum(n.weights), 1.0):
            return False, "Sum of weights of node #%s is not 1.0" % n.id
        if len(n.children) == 0:
            return False, "Sum node #%s has no children" % n.id
        if len(n.children) != len(n.weights):
            return False, "Weights and children length mismatch in node #%s" % n.id
        n_scope = set(n.scope)
        for c in n.children:
            if n_scope != set(c.scope):
                return False, "Children of sum node #%s have different scopes" % n.id
    return True, None


def is_consistent(root):
    """
    Check if the SPN is consistent.
    It checks that each child of a sum node has disjointed scopes.

    :param root: The root of the SPN.
    :return: (True, None) if the SPN is consistent;
             (False, reason) otherwise.
    """
    for n in filter_nodes_type(root, Mul):
        if len(n.children) == 0:
            return False, "Mul node #%s has no children" % n.id
        sum_features = 0
        all_scope = set()
        n_scope = set(n.scope)
        for c in n.children:
            sum_features += len(c.scope)
            all_scope.update(c.scope)
        if n_scope != all_scope or sum_features != len(all_scope):
            return False, "Children of mul node #%s don't have disjointed scopes" % n.id
    return True, None


def is_labeled(root):
    """
    Check if the SPN is labeled correctly.
    It checks that the initial id is zero and each id is consecutive.

    :param root: The root of the SPN.
    :return: (True, None) if the SPN is labeled correctly;
             (False, reason) otherwise.
    """
    ids = set()
    nodes = get_nodes(root)
    for n in nodes:
        if n.id is not None:
            ids.add(n.id)
    if len(ids) != len(nodes):
        return False, "Some nodes have missing or repeated ids"
    if min(ids) != 0:
        return False, "Node ids not starting at 0"
    if max(ids) != len(ids) - 1:
        return False, "Node ids not consecutive"
    return True, None
