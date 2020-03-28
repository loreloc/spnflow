from enum import Enum
from collections import deque
from spnflow.structure.node import Sum, Mul, assign_ids
from spnflow.learning.splitting.rows import get_split_rows_method, split_rows_clusters
from spnflow.learning.splitting.cols import get_split_cols_method, split_cols_clusters


class Operation(Enum):
    CREATE_LEAF = 1,
    SPLIT_NAIVE = 2,
    SPLIT_ROWS = 3,
    SPLIT_COLS = 4


def learn_structure(data, distributions, domains,
                    split_rows='kmeans', split_cols='rdc',
                    min_rows_slice=128, min_cols_slice=1,
                    n_clusters=2, threshold=0.25):
    assert data is not None
    assert len(distributions) > 0
    assert len(domains) > 0
    assert split_rows is not None
    assert split_cols is not None
    assert min_rows_slice > 0
    assert min_cols_slice > 0
    assert n_clusters > 1
    assert threshold > 0.0

    n_samples, n_features = data.shape
    split_rows_func = get_split_rows_method(split_rows)
    split_cols_func = get_split_cols_method(split_cols)
    initial_scope = list(range(n_features))

    tasks = deque()
    tmp_node = Mul([], initial_scope)
    tasks.append((tmp_node, data, initial_scope, False, False))

    while tasks:
        task = (parent, local_data, scope, no_rows_split, no_cols_split) = tasks.popleft()
        op = choose_next_operation(task, min_rows_slice, min_cols_slice)

        if op == Operation.CREATE_LEAF:
            idx = scope[0]
            leaf = distributions[idx](scope)
            leaf.fit(local_data, domains[idx])
            parent.children.append(leaf)
        elif op == Operation.SPLIT_NAIVE:
            node = Mul([], scope)
            n_local_samples, n_local_features = local_data.shape
            for i in range(n_local_features):
                s = local_data[:, i].reshape(n_local_samples, -1)
                tasks.append((node, s, [scope[i]], True, True))
            parent.children.append(node)
        elif op == Operation.SPLIT_ROWS:
            clusters = split_rows_func(local_data, n_clusters)
            slices, weights = split_rows_clusters(local_data, clusters)
            if len(slices) == 1:
                tasks.append((parent, local_data, scope, True, no_cols_split))
                continue
            node = Sum(weights, [], scope)
            for s in slices:
                tasks.append((node, s, scope, False, no_cols_split))
            parent.children.append(node)
        elif op == Operation.SPLIT_COLS:
            clusters = split_cols_func(local_data, threshold)
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
    parent, local_data, scope, no_rows_split, no_cols_split = task
    n_samples, n_features = local_data.shape

    split_rows_op = Operation.SPLIT_ROWS
    split_cols_op = Operation.SPLIT_COLS

    split_end_op = Operation.CREATE_LEAF
    if n_features > 1:
        split_end_op = Operation.SPLIT_NAIVE

    if no_rows_split and no_cols_split:
        return split_end_op

    if n_samples < min_rows_slice:
        if no_cols_split:
            return split_end_op
        elif n_features >= min_cols_slice:
            return split_cols_op

    if n_features < min_cols_slice:
        if no_rows_split:
            return split_end_op
        elif n_samples >= min_rows_slice:
            return split_rows_op

    if no_cols_split:
        return split_rows_op
    if no_rows_split:
        return split_cols_op

    if n_features >= min_cols_slice:
        return split_cols_op
    elif n_samples >= min_rows_slice:
        return split_rows_op

    return split_end_op
