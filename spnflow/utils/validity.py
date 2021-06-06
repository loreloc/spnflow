import numpy as np
from spnflow.structure.node import Sum, Mul
from spnflow.structure.cltree import BinaryCLTree
from spnflow.utils.filter import get_nodes, filter_nodes_type


def assert_is_valid(root):
    """
    Assert if the SPN is valid.

    :param root: The SPN root.
    """
    v, msg = is_valid(root)
    assert v, "SPN not valid: {}".format(msg)


def is_valid(root):
    """
    Check if the SPN is valid.

    :param root: The SPN root.
    :return: (True, None) if the SPN is valid;
             (False, reason) otherwise.
    """
    valid, msg = is_smooth(root)
    if not valid:
        return valid, msg

    valid, msg = is_decomposable(root)
    if not valid:
        return valid, msg

    valid, msg = is_labeled(root)
    if not valid:
        return valid, msg

    return True, None


def is_smooth(root):
    """
    Check if the SPN is smooth (or complete).
    It checks that each child of a sum node has the same scope.
    Furthermore, it checks that the sum of the weights of a sum node is close to 1.

    :param root: The root of the SPN.
    :return: (True, None) if the SPN is complete;
             (False, reason) otherwise.
    """
    for n in filter_nodes_type(root, Sum):
        if not np.isclose(np.sum(n.weights), 1.0):
            return False, "Sum of weights of node #{} is not 1.0".format(n.id)
        if len(n.children) == 0:
            return False, "Sum node #{} has no children".format(n.id)
        if len(n.children) != len(n.weights):
            return False, "Weights and children length mismatch in node #{}".format(n.id)
        n_scope = set(n.scope)
        for c in n.children:
            if n_scope != set(c.scope):
                return False, "Children of sum node #{} have different scopes".format(n.id)
    return True, None


def is_decomposable(root):
    """
    Check if the SPN is decomposable (or consistent).
    It checks that each child of a product node has disjointed scopes.

    :param root: The root of the SPN.
    :return: (True, None) if the SPN is consistent;
             (False, reason) otherwise.
    """
    for n in filter_nodes_type(root, Mul):
        if len(n.children) == 0:
            return False, "Mul node #{} has no children".format(n.id)
        sum_features = 0
        all_scope = set()
        n_scope = set(n.scope)
        for c in n.children:
            sum_features += len(c.scope)
            all_scope.update(c.scope)
        if n_scope != all_scope or sum_features != len(all_scope):
            return False, "Children of mul node #{} don't have disjointed scopes".format(n.id)
    return True, None


def is_structured_decomposable(root, verbose=False):
    """
    Check if the PC is structured decomposable.
    It checks that product nodes follow a vtree.
    Note that if a PC is structured decomposable then it's also decomposable / consistent.

    :param root: The root of the PC.
    :param verbose: if True, it prints the product nodes scopes in a relevant order.
    :return: (True, None) if the PC is structured decomposable;
             (False, reason) otherwise.
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

    # ordering is not needed, but useful for printing when verbose = True
    if verbose:
        scope_l.sort(key=len)
        for scope in scope_l:
            print(scope)

    # quadratic in the number of product nodes, but at least does not require a vtree structure
    for i in range(len(scope_l)):
        for j in range(len(scope_l)):
            int_len = len(scope_l[i].intersection(scope_l[j]))
            if int_len != 0 and int_len != min(len(scope_l[i]), len(scope_l[j])):
                return False, "Intersection between scope {} and scope {}".format(scope_l[i], scope_l[j])

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
