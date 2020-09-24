import numpy as np
from enum import Enum
from collections import deque
from tqdm import tqdm

from spnflow.structure.node import Sum, Mul, assign_ids
from spnflow.learning.leaf import get_learn_leaf_method
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


def learn_structure(data,
                    distributions,
                    domains,
                    learn_leaf='mle',
                    learn_leaf_params=None,
                    split_rows='kmeans',
                    split_cols='rdc',
                    split_rows_params=None,
                    split_cols_params=None,
                    min_rows_slice=256,
                    min_cols_slice=2,
                    id_bar=None,
                    ):
    """
    Learn the structure and parameters of a SPN given some training data and several hyperparameters.

    :param data: The training data.
    :param distributions: A list of distributions classes (one for each feature).
    :param domains: A list of domains (one for each feature).
    :param learn_leaf: The method to use to learn a distribution leaf node (it can be 'mle' or 'isotonic').
    :param learn_leaf_params: The parameters of the learn leaf method.
    :param split_rows: The rows splitting method (it can be 'kmeans', 'gmm', 'rdc' or 'random').
    :param split_cols: The columns splitting method (it can be 'rdc_cols' or 'random').
    :param split_rows_params: The parameters of the rows splitting method.
    :param split_cols_params: The parameters of the cols splitting method.
    :param min_rows_slice: The minimum number of samples required to split horizontally.
    :param min_cols_slice: The minimum number of features required to split vertically.
    :param id_bar: The id used for the progress bar. This can be used to make nested tqdm bars.
    :return: A learned valid SPN.
    """
    assert data is not None
    assert len(distributions) > 0
    assert len(domains) > 0
    assert split_rows is not None
    assert split_cols is not None
    assert min_rows_slice > 1
    assert min_cols_slice > 1
    assert id_bar is None or id_bar > 0

    if learn_leaf_params is None:
        learn_leaf_params = {}
    if split_rows_params is None:
        split_rows_params = {}
    if split_cols_params is None:
        split_cols_params = {}

    n_samples, n_features = data.shape
    assert len(distributions) == n_features, "Each feature must have a distribution"

    learn_leaf_func = get_learn_leaf_method(learn_leaf)
    split_rows_func = get_split_rows_method(split_rows)
    split_cols_func = get_split_cols_method(split_cols)
    initial_scope = list(range(n_features))

    tasks = deque()
    tmp_node = Mul([], initial_scope)
    tasks.append((tmp_node, data, initial_scope, False, False, True))
    tk = tqdm(total=np.inf, position=id_bar, leave=False)

    while tasks:
        task = (parent, local_data, scope, no_rows_split, no_cols_split, is_first) = tasks.popleft()
        op, params = choose_next_operation(task, min_rows_slice, min_cols_slice)

        if op == Operation.CREATE_LEAF:
            idx = scope[0]
            dist, dom = distributions[idx], domains[idx]
            leaf = learn_leaf_func(local_data, dist, dom, scope, **learn_leaf_params)
            parent.children.append(leaf)
        elif op == Operation.REM_FEATURE:
            node = Mul([], scope)
            rem_scope = [scope[x.item()] for x in np.argwhere( params)]
            oth_scope = [scope[x.item()] for x in np.argwhere(~params)]
            is_next_first = is_first and len(tasks) == 0
            tasks.append((node, local_data[:,  params], rem_scope, True, True, False))
            tasks.append((node, local_data[:, ~params], oth_scope, False, False, is_next_first))
            parent.children.append(node)
        elif op == Operation.SPLIT_NAIVE:
            node = Mul([], scope)
            n_local_samples, n_local_features = local_data.shape
            for i in range(n_local_features):
                s = local_data[:, i].reshape(n_local_samples, -1)
                tasks.append((node, s, [scope[i]], True, True, False))
            parent.children.append(node)
        elif op == Operation.SPLIT_ROWS:
            clusters = split_rows_func(local_data, **split_rows_params)
            slices, weights = split_rows_clusters(local_data, clusters)
            if len(slices) == 1:
                tasks.append((parent, local_data, scope, True, no_cols_split, False))
                continue
            node = Sum(weights, [], scope)
            for s in slices:
                tasks.append((node, s, scope, False, False, False))
            parent.children.append(node)
        elif op == Operation.SPLIT_COLS:
            clusters = split_cols_func(local_data, **split_cols_params)
            slices, scopes = split_cols_clusters(local_data, clusters, scope)
            if len(slices) == 1:
                tasks.append((parent, local_data, scope, no_rows_split, True, False))
                continue
            node = Mul([], scope)
            for i, s in enumerate(slices):
                tasks.append((node, s, scopes[i], False, False, False))
            parent.children.append(node)
        else:
            raise NotImplementedError("Operation of kind " + op.__name__ + " not implemented")

        # Update the progress bar
        tk.update()
        tk.refresh()

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
    parent, local_data, scope, no_rows_split, no_cols_split, is_first = task
    n_samples, n_features = local_data.shape

    min_samples = n_samples < min_rows_slice
    min_features = n_features < min_cols_slice

    if min_features:
        if n_features == 1:
            return Operation.CREATE_LEAF, None
        else:
            return Operation.SPLIT_NAIVE, None

    if min_samples or (no_rows_split and no_cols_split):
        return Operation.SPLIT_NAIVE, None

    if no_cols_split:
        return Operation.SPLIT_ROWS, None
    if no_rows_split:
        return Operation.SPLIT_COLS, None

    zero_var_idx = np.isclose(np.var(local_data, axis=0), 0.0)
    if np.all(zero_var_idx):
        return Operation.SPLIT_NAIVE, None
    if np.any(zero_var_idx):
        return Operation.REM_FEATURE, zero_var_idx

    if is_first:
        return Operation.SPLIT_ROWS, None

    return Operation.SPLIT_COLS, None
