import numpy as np
from enum import Enum
from tqdm import tqdm

from spnflow.structure.node import Sum, Mul, assign_ids
from spnflow.learning.leaf import get_learn_leaf_method
from spnflow.learning.splitting.rows import get_split_rows_method, split_rows_clusters
from spnflow.learning.splitting.cols import get_split_cols_method, split_cols_clusters


class OperationKind(Enum):
    CREATE_LEAF = 1,
    REM_FEATURE = 2
    SPLIT_NAIVE = 3,
    SPLIT_ROWS = 4,
    SPLIT_COLS = 5


class Task:
    def __init__(self, parent, data, scope, no_rows_split=False, no_cols_split=False, is_first=False):
        self.parent = parent
        self.data = data
        self.scope = scope
        self.no_rows_split = no_rows_split
        self.no_cols_split = no_cols_split
        self.is_first = is_first


def learn_structure(data,
                    distributions,
                    domains,
                    learn_leaf='mle',
                    learn_leaf_params=None,
                    split_rows='kmeans',
                    split_cols='rdc',
                    split_rows_kwargs=None,
                    split_cols_kwargs=None,
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
    :param split_rows_kwargs: The parameters of the rows splitting method.
    :param split_cols_kwargs: The parameters of the cols splitting method.
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
    if split_rows_kwargs is None:
        split_rows_kwargs = {}
    if split_cols_kwargs is None:
        split_cols_kwargs = {}

    n_samples, n_features = data.shape
    assert len(distributions) == n_features, "Each feature must have a distribution"

    learn_leaf_func = get_learn_leaf_method(learn_leaf)
    split_rows_func = get_split_rows_method(split_rows)
    split_cols_func = get_split_cols_method(split_cols)
    initial_scope = list(range(n_features))

    tasks = []
    tmp_node = Mul([], initial_scope)
    tasks.append(Task(tmp_node, data, initial_scope, is_first=True))

    # Initialize the progress bar (with unspecified total)
    tk = tqdm(total=np.inf, position=id_bar, leave=False)

    while tasks:
        # Get the next task
        task = tasks.pop()

        # First of all, check for uninformative features
        op, zero_var_idx = None, None
        n_samples, n_features = task.data.shape
        if n_features > 1:
            zero_var_idx = np.isclose(np.var(task.data, axis=0), 0.0)
            if np.all(zero_var_idx):
                op = OperationKind.SPLIT_NAIVE
            elif np.any(zero_var_idx):
                op = OperationKind.REM_FEATURE

        # Choose the next operation according to the task and hyper-parameters
        if op is None:
            op = choose_next_operation(task, min_rows_slice, min_cols_slice)

        if op == OperationKind.CREATE_LEAF:
            idx = task.scope[0]
            dist, dom = distributions[idx], domains[idx]
            leaf = learn_leaf_func(task.data, dist, dom, task.scope, **learn_leaf_params)
            task.parent.children.append(leaf)
        elif op == OperationKind.REM_FEATURE:
            node = Mul([], task.scope)
            rem_scope = [task.scope[x.item()] for x in np.argwhere( zero_var_idx)]
            oth_scope = [task.scope[x.item()] for x in np.argwhere(~zero_var_idx)]
            is_first = task.is_first and len(tasks) == 0
            tasks.append(Task(node, task.data[:,  zero_var_idx], rem_scope, True, True))
            tasks.append(Task(node, task.data[:, ~zero_var_idx], oth_scope, False, False, is_first))
            task.parent.children.append(node)
        elif op == OperationKind.SPLIT_NAIVE:
            node = Mul([], task.scope)
            for i in range(n_features):
                local_data = task.data[:, i].reshape(n_samples, -1)
                tasks.append(Task(node, local_data, [task.scope[i]], True, True))
            task.parent.children.append(node)
        elif op == OperationKind.SPLIT_ROWS:
            clusters = split_rows_func(task.data, **split_rows_kwargs)
            slices, weights = split_rows_clusters(task.data, clusters)
            if len(slices) == 1:
                tasks.append(Task(task.parent, task.data, task.scope, True, task.no_cols_split))
                continue
            node = Sum(weights, [], task.scope)
            for local_data in slices:
                tasks.append(Task(node, local_data, task.scope, False, False))
            task.parent.children.append(node)
        elif op == OperationKind.SPLIT_COLS:
            clusters = split_cols_func(task.data, **split_cols_kwargs)
            slices, scopes = split_cols_clusters(task.data, clusters, task.scope)
            if len(slices) == 1:
                tasks.append(Task(task.parent, task.data, task.scope, task.no_rows_split, True))
                continue
            node = Mul([], task.scope)
            for i, local_data in enumerate(slices):
                tasks.append(Task(node, local_data, scopes[i], False, False))
            task.parent.children.append(node)
        else:
            raise NotImplementedError("Operation of kind " + op.__name__ + " not implemented")

        # Update the progress bar
        tk.update()
        tk.refresh()

    root = tmp_node.children[0]
    return assign_ids(root)


def choose_next_operation(task, min_rows_slice, min_cols_slice):
    n_samples, n_features = task.data.shape

    # First of all, check if previous consecutive rows or columns splitting have failed
    if task.no_rows_split and task.no_cols_split:
        if n_features == 1:
            return OperationKind.CREATE_LEAF
        else:
            return OperationKind.SPLIT_NAIVE

    # Check task's data size
    min_samples = n_samples < min_rows_slice
    min_features = n_features < min_cols_slice
    if min_samples:
        if min_features:
            # If we cannot both split horizontally and vertically use naive splitting
            if n_features == 1:
                op = OperationKind.CREATE_LEAF
            else:
                op = OperationKind.SPLIT_NAIVE
        else:
            # If we cannot split horizontally but can split vertically use columns splitting
            if n_features == 1:
                op = OperationKind.CREATE_LEAF
            else:
                op = OperationKind.SPLIT_COLS
    else:
        if min_features:
            # If we can split horizontally but cannot split vertically use rows splitting
            if n_features == 1:
                op = OperationKind.CREATE_LEAF
            else:
                op = OperationKind.SPLIT_ROWS
        else:
            # Defaults to columns splitting (but if task is first use rows splitting)
            if n_features == 1:
                op = OperationKind.CREATE_LEAF
            else:
                if task.is_first:
                    op = OperationKind.SPLIT_ROWS
                else:
                    op = OperationKind.SPLIT_COLS

    # Check the last operation and adjust accordingly
    if task.no_rows_split and op == OperationKind.SPLIT_ROWS:
        if min_features:
            return OperationKind.SPLIT_NAIVE
        else:
            return OperationKind.SPLIT_COLS
    elif task.no_cols_split and op == OperationKind.SPLIT_COLS:
        if min_samples:
            return OperationKind.SPLIT_NAIVE
        else:
            return OperationKind.SPLIT_ROWS
    else:
        return op
