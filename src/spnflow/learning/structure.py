from spnflow.structure.node import Sum, Mul, assign_ids
from spnflow.learning.tasks import OperationKind, TaskQueue
from spnflow.learning.splitting.rows import get_split_rows_method, split_rows_clusters
from spnflow.learning.splitting.cols import get_split_cols_method, split_cols_clusters


def learn_structure(data, distributions, root_split='cols',
                    split_rows='kmeans', split_cols='rdc',
                    min_rows_slice=128, min_cols_slice=1,
                    n_clusters=2, threshold=0.25):
    assert data is not None
    assert split_rows is not None
    assert split_cols is not None
    assert min_rows_slice > 0
    assert min_cols_slice > 0
    assert threshold > 0.0

    root = None
    initial_split = None
    data_len, scopes_len = data.shape
    split_rows_func = get_split_rows_method(split_rows)
    split_cols_func = get_split_cols_method(split_cols)
    initial_scope = list(range(scopes_len))

    if root_split == 'rows':
        initial_split = OperationKind.ROOT_SPLIT_ROWS
    elif root_split == 'cols':
        initial_split = OperationKind.ROOT_SPLIT_COLS
    else:
        raise NotImplementedError("Root split " + root_split + " not implemented")

    tasks = TaskQueue(min_rows_slice, min_cols_slice)
    tasks.push(None, initial_split, data, initial_scope)

    while tasks:
        (parent, op, local_data, scope) = tasks.pop()

        if op == OperationKind.CREATE_LEAF:
            leaf = distributions[scope](scope)
            leaf.fit(local_data)
            parent.children.append(leaf)
        elif op == OperationKind.SPLIT_ROWS:
            clusters = split_rows_func(local_data, n_clusters)
            weights, slices = split_rows_clusters(local_data, clusters)
            node = Sum(weights, [], scope)
            for s in slices:
                tasks.push(node, OperationKind.SPLIT_COLS, s, scope)
            parent.children.append(node)
        elif op == OperationKind.SPLIT_COLS:
            clusters = split_cols_func(local_data, threshold)
            slices, scopes = split_cols_clusters(local_data, clusters, scope)
            if len(slices) == 1:
                tasks.push(parent, OperationKind.SPLIT_ROWS, local_data, scope)
                continue
            node = Mul([], scope)
            for i, s in enumerate(slices):
                tasks.push(node, OperationKind.SPLIT_ROWS, s, scopes[i])
            parent.children.append(node)
        elif op == OperationKind.NAIVE_FACTORIZE:
            node = Mul()
            _, n_features = local_data.shape
            for i in range(n_features):
                tasks.push(node, OperationKind.CREATE_LEAF, local_data[:, i], i)
        elif op == OperationKind.ROOT_SPLIT_ROWS:
            clusters = split_rows_func(local_data, n_clusters)
            weights, slices = split_rows_clusters(local_data, clusters)
            root = Sum(weights, [], scope)
            for s in slices:
                tasks.push(root, OperationKind.SPLIT_COLS, s, scope)
        elif op == OperationKind.ROOT_SPLIT_COLS:
            clusters = split_cols_func(local_data, threshold)
            slices, scopes = split_cols_clusters(local_data, scope)
            if len(slices) == 1:
                tasks.push(None, OperationKind.ROOT_SPLIT_ROWS, local_data, scope)
                continue
            root = Mul([], scope)
            for i, s in enumerate(slices):
                tasks.push(root, OperationKind.SPLIT_ROWS, s, scopes[i])
        else:
            raise NotImplementedError("Operation of kind " + op.__name__ + " not implemented")

    return assign_ids(root)
