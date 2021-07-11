import numpy as np

from deeprob.spn.structure.node import Sum, Mul
from deeprob.spn.structure.cltree import BinaryCLTree
from deeprob.spn.utils.filter import get_nodes, filter_nodes_type


def assert_smooth(root):
    """
    Assert the SPN is smooth (or complete).

    :param root: The root of the SPN.
    """
    result = is_smooth(root)
    if result is not None:
        assert False, result


def assert_decomposable(root):
    """
    Assert the SPN is decomposable (or consistent).

    :param root: The root of the SPN.
    """
    result = is_decomposable(root)
    if result is not None:
        assert False, result


def assert_structured_decomposable(root):
    """
    Assert the SPN is structured decomposable.

    :param root: The root of the SPN.
    """
    result = is_structured_decomposable(root)
    if result is not None:
        assert False, result


def assert_labeled(root):
    """
    Assert the SPN is correctly labeled by IDs.

    :param root: The root of the SPN.
    """
    result = is_labeled(root)
    if result is not None:
        assert False, result


def is_smooth(root):
    """
    Check if the SPN is smooth (or complete).
    It checks that each child of a sum node has the same scope.
    Furthermore, it checks that the sum of the weights of a sum node is close to 1.

    :param root: The root of the SPN.
    :return: None if the SPN is complete, a reason otherwise.
    """
    for n in filter_nodes_type(root, Sum):
        if not np.isclose(np.sum(n.weights), 1.0):
            return "Sum of weights of node #{} is not 1.0".format(n.id)
        if len(n.children) == 0:
            return "Sum node #{} has no children".format(n.id)
        if len(n.children) != len(n.weights):
            return "Weights and children length mismatch in node #{}".format(n.id)
        n_scope = set(n.scope)
        for c in n.children:
            if n_scope != set(c.scope):
                return "Children of sum node #{} have different scopes".format(n.id)
    return None


def is_decomposable(root):
    """
    Check if the SPN is decomposable (or consistent).
    It checks that each child of a product node has disjointed scopes.

    :param root: The root of the SPN.
    :return: None if the SPN is decomposable, a reason otherwise.
    """
    for n in filter_nodes_type(root, Mul):
        if len(n.children) == 0:
            return "Mul node #{} has no children".format(n.id)
        sum_features = 0
        all_scope = set()
        n_scope = set(n.scope)
        for c in n.children:
            sum_features += len(c.scope)
            all_scope.update(c.scope)
        if n_scope != all_scope or sum_features != len(all_scope):
            return "Children of mul node #{} don't have disjointed scopes".format(n.id)
    return None


def is_structured_decomposable(root, verbose=False):
    """
    Check if the PC is structured decomposable.
    It checks that product nodes follow a vtree.
    Note that if a PC is structured decomposable then it's also decomposable / consistent.

    :param root: The root of the PC.
    :param verbose: if True, it prints the product nodes scopes in a relevant order.
    :return: None if the PC is structured decomposable, a reason otherwise.
    """
    nodes = get_nodes(root)

    scope_set = set()
    for n in nodes:
        if isinstance(n, Mul):
            scope_set.add(tuple(n.scope))
        elif isinstance(n, BinaryCLTree):
            raise Exception("Case not yet considered.")

    scope_l = list(scope_set)
    scope_l = [set(t) for t in scope_l]

    # Ordering is not needed, but useful for printing when verbose = True
    if verbose:
        scope_l.sort(key=len)
        for scope in scope_l:
            print(scope)

    # Quadratic in the number of product nodes, but at least does not require a vtree structure
    for i in range(len(scope_l)):
        for j in range(len(scope_l)):
            int_len = len(scope_l[i].intersection(scope_l[j]))
            if int_len != 0 and int_len != min(len(scope_l[i]), len(scope_l[j])):
                return "Intersection between scope {} and scope {}".format(scope_l[i], scope_l[j])
    return None


def is_labeled(root):
    """
    Check if the SPN is labeled correctly.
    It checks that the initial id is zero and each id is consecutive.

    :param root: The root of the SPN.
    :return: None if the SPN is labeled correctly, a reason otherwise.
    """
    ids = set()
    nodes = get_nodes(root)
    for n in nodes:
        if n.id is not None:
            ids.add(n.id)
    if len(ids) != len(nodes):
        return "Some nodes have missing or repeated ids"
    if min(ids) != 0:
        return "Node ids not starting at 0"
    if max(ids) != len(ids) - 1:
        return "Node ids not consecutive"
    return None
