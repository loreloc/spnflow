import numpy as np
from enum import Enum
from collections import deque
from spnflow.structure.node import Sum, Mul, assign_ids
from spnflow.learning.splitting.rows import get_split_rows_method, split_rows_clusters
from spnflow.learning.splitting.cols import get_split_cols_method, split_cols_clusters


class Operation(Enum):
    """
    The possible operation to execute for SPN structure learning.
    """
    CREATE_LEAF = 1,
    REM_FEATURE = 2
    SPLIT_NAIVE = 3,
    SPLIT_ROWS = 4,
    SPLIT_COLS = 5


def learn_structure(data, distributions, domains,
                    split_rows='kmeans', split_cols='rdc',
                    split_rows_params={}, split_cols_params={},
                    min_rows_slice=128, min_cols_slice=2):
    """
    Learn the structure and parameters of a SPN given some training data and several hyperparameters.

    :param data: The training data.
    :param distributions: A list of distributions classes (one for each feature).
    :param domains: A list of domains (one for each feature).
    :param split_rows: The rows splitting method (it can be 'kmeans', 'gmm' or 'random').
    :param split_cols: The columns splitting method (it can be 'rdc' or 'random').
    :param split_rows_params: The parameters of the rows splitting method.
    :param split_cols_params: The parameters of the cols splitting method.
    :param min_rows_slice: The minimum number of samples required to split horizontally.
    :param min_cols_slice: The minimum number of features required to split vertically.
    :return: A learned valid SPN.
    """
    assert data is not None
    assert len(distributions) > 0
    assert len(domains) > 0
    assert split_rows is not None
    assert split_cols is not None
    assert min_rows_slice > 1
    assert min_cols_slice > 1

    n_samples, n_features = data.shape
    assert len(distributions) == n_features, "Each feature must have a distribution"

    split_rows_func = get_split_rows_method(split_rows)
    split_cols_func = get_split_cols_method(split_cols)
    initial_scope = list(range(n_features))

    tasks = deque()
    tmp_node = Mul([], initial_scope)
    tasks.append((tmp_node, data, initial_scope, False, False))

    while tasks:
        task = (parent, local_data, scope, no_rows_split, no_cols_split) = tasks.popleft()
        op, params = choose_next_operation(task, min_rows_slice, min_cols_slice)

        if op == Operation.CREATE_LEAF:
            idx = scope[0]
            leaf = distributions[idx](scope)
            leaf.fit(local_data, domains[idx])
            parent.children.append(leaf)
        elif op == Operation.REM_FEATURE:
            node = Mul([], scope)
            rem_scope = [scope[x.item()] for x in np.argwhere( params)]
            oth_scope = [scope[x.item()] for x in np.argwhere(~params)]
            tasks.append((node, local_data[:,  params], rem_scope, True, True))
            tasks.append((node, local_data[:, ~params], oth_scope, False, False))
            parent.children.append(node)
        elif op == Operation.SPLIT_NAIVE:
            node = Mul([], scope)
            n_local_samples, n_local_features = local_data.shape
            for i in range(n_local_features):
                s = local_data[:, i].reshape(n_local_samples, -1)
                tasks.append((node, s, [scope[i]], True, True))
            parent.children.append(node)
        elif op == Operation.SPLIT_ROWS:
            clusters = split_rows_func(local_data, **split_rows_params)
            slices, weights = split_rows_clusters(local_data, clusters)
            if len(slices) == 1:
                tasks.append((parent, local_data, scope, True, no_cols_split))
                continue
            node = Sum(weights, [], scope)
            for s in slices:
                tasks.append((node, s, scope, False, no_cols_split))
            parent.children.append(node)
        elif op == Operation.SPLIT_COLS:
            clusters = split_cols_func(local_data, **split_cols_params)
            slices, scopes = split_cols_clusters(local_data, clusters, scope)
            if len(slices) == 1:
                tasks.append((parent, local_data, scope, no_rows_split, True))
                continue
            node = Mul([], scope)
            for i, s in enumerate(slices):
                tasks.append((node, s, scopes[i], no_rows_split, False))
            parent.children.append(node)
        else:
            raise NotImplementedError("Operation of kind " + op.__name__ + " not implemented")

    root = tmp_node.children[0]
    return assign_ids(root)


def choose_next_operation(task, min_rows_slice, min_cols_slice):
    """
    Choose the next operation to execute.

    :param task: The next task, a tuple composed by the parent node, the local data, the scope and two booleans
                 indicating if rows and columns splitting have failed or not respectively.
    :param min_rows_slice: The minimum number of samples required to split horizontally.
    :param min_cols_slice: The minimum number of features required to split vertically.
    :return: (op, params) where op is the operation to execute and params are the optional parameters of the operation.
    """
    parent, local_data, scope, no_rows_split, no_cols_split = task
    n_samples, n_features = local_data.shape

    if no_rows_split and no_cols_split:
        if n_features == 1:
            return Operation.CREATE_LEAF, None
        else:
            return Operation.SPLIT_NAIVE, None

    zero_var_idx = np.isclose(np.var(local_data, axis=0), 0.0)
    if np.all(zero_var_idx):
        return Operation.SPLIT_NAIVE, None
    if np.any(zero_var_idx):
        return Operation.REM_FEATURE, zero_var_idx

    if n_samples < min_rows_slice:
        if no_cols_split:
            if n_features == 1:
                return Operation.CREATE_LEAF, None
            else:
                return Operation.SPLIT_NAIVE, None
        elif n_features >= min_cols_slice:
            return Operation.SPLIT_COLS, None

    if n_features < min_cols_slice:
        if no_rows_split:
            if n_features == 1:
                return Operation.CREATE_LEAF, None
            else:
                return Operation.SPLIT_NAIVE, None
        elif n_samples >= min_rows_slice:
            return Operation.SPLIT_ROWS, None

    if no_cols_split:
        return Operation.SPLIT_ROWS, None
    if no_rows_split:
        return Operation.SPLIT_COLS, None

    if n_features >= min_cols_slice:
        return Operation.SPLIT_COLS, None
    elif n_samples >= min_rows_slice:
        return Operation.SPLIT_ROWS, None

    return Operation.SPLIT_NAIVE, None
