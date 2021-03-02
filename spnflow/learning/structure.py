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
    def __init__(self, parent, data, scope, no_cols_split=False, no_rows_split=False, is_first=False):
        self.parent = parent
        self.data = data
        self.scope = scope
        self.no_cols_split = no_cols_split
        self.no_rows_split = no_rows_split
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
                    verbose=True,
                    ):
    """
    Learn the structure and parameters of a SPN given some training data and several hyperparameters.

    :param data: The training data.
    :param distributions: A list of distributions classes (one for each feature).
    :param domains: A list of domains (one for each feature).
    :param learn_leaf: The method to use to learn a distribution leaf node (it can be 'mle' or 'isotonic').
    :param learn_leaf_params: The parameters of the learn leaf method.
    :param split_rows: The rows splitting method (it can be 'kmeans', 'gmm', 'rdc' or 'random').
    :param split_cols: The columns splitting method (it can be 'gvs', 'rdc' or 'random').
    :param split_rows_kwargs: The parameters of the rows splitting method.
    :param split_cols_kwargs: The parameters of the cols splitting method.
    :param min_rows_slice: The minimum number of samples required to split horizontally.
    :param min_cols_slice: The minimum number of features required to split vertically.
    :param verbose: Whether to enable verbose mode.
    :return: A learned valid SPN.
    """
    assert data is not None
    assert len(distributions) > 0
    assert len(domains) > 0
    assert split_rows is not None
    assert split_cols is not None
    assert min_rows_slice > 1
    assert min_cols_slice > 1

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

    # Initialize the progress bar (with unspecified total), if verbose is enabled
    tk = tqdm(total=np.inf, leave=None) if verbose else None

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
                tasks.append(Task(node, local_data, [task.scope[i]], True))
            task.parent.children.append(node)
        elif op == OperationKind.SPLIT_ROWS:
            dists = [distributions[s] for s in task.scope]
            doms = [domains[s] for s in task.scope]
            clusters = split_rows_func(task.data, dists, doms, **split_rows_kwargs)
            slices, weights = split_rows_clusters(task.data, clusters)
            if len(slices) == 1:
                tasks.append(Task(task.parent, task.data, task.scope, False, True))
                continue
            node = Sum(weights, [], task.scope)
            for local_data in slices:
                tasks.append(Task(node, local_data, task.scope))
            task.parent.children.append(node)
        elif op == OperationKind.SPLIT_COLS:
            dists = [distributions[s] for s in task.scope]
            doms = [domains[s] for s in task.scope]
            clusters = split_cols_func(task.data, dists, doms, **split_cols_kwargs)
            slices, scopes = split_cols_clusters(task.data, clusters, task.scope)
            if len(slices) == 1:
                tasks.append(Task(task.parent, task.data, task.scope, True, False))
                continue
            node = Mul([], task.scope)
            for i, local_data in enumerate(slices):
                tasks.append(Task(node, local_data, scopes[i]))
            task.parent.children.append(node)
        else:
            raise NotImplementedError("Operation of kind " + op.__name__ + " not implemented")

        if verbose:
            tk.update()
            tk.refresh()

    if verbose:
        tk.close()

    root = tmp_node.children[0]
    return assign_ids(root)


def choose_next_operation(task, min_rows_slice, min_cols_slice):
    n_samples, n_features = task.data.shape

    # Check if the data is univariate
    if n_features == 1:
        return OperationKind.CREATE_LEAF

    # Check if the data has low number of features
    if n_features < min_cols_slice:
        return OperationKind.SPLIT_NAIVE

    # Check if the data has low number of samples
    if n_samples < min_rows_slice:
        return OperationKind.SPLIT_NAIVE

    # Check if previous rows splitting have failed
    if task.no_rows_split:
        return OperationKind.SPLIT_NAIVE

    # Check if previous columns splitting have failed
    if task.no_cols_split:
        return OperationKind.SPLIT_ROWS
    else:
        if task.is_first:
            return OperationKind.SPLIT_ROWS
        else:
            return OperationKind.SPLIT_COLS
