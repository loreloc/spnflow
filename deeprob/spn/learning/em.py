import numpy as np

from tqdm import tqdm
from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.structure.node import Sum
from deeprob.spn.utils.filter import filter_nodes_type
from deeprob.spn.algorithms.inference import log_likelihood
from deeprob.spn.algorithms.gradient import eval_backward


def expectation_maximization(
        spn,
        data,
        num_iter=100,
        batch_perc=0.1,
        step_size=0.5,
        random_init=True,
        random_state=None,
        verbose=True
):
    """
    Learn the parameters of a SPN by batch Expectation-Maximization (EM).
    See https://arxiv.org/abs/1604.07243 and https://arxiv.org/abs/2004.06231 for details.

    :param spn: The spn structure.
    :param data: The data to use to learn the parameters.
    :param num_iter: The number of iterations.
    :param batch_perc: The percentage of data to use for each step.
    :param step_size: The step size for batch EM.
    :param random_init: Whether to random initialize the weights of the SPN.
    :param random_state: The random state. It can be either None, a seed integer or a Numpy RandomState.
    :param verbose: Whether to enable verbose learning.
    :return: The spn with learned parameters.
    """
    assert num_iter > 0, "The number of iterations must be positive"
    assert 0.0 < batch_perc < 1.0, "The batch percentage must be in (0, 1)"
    assert 0.0 < step_size < 1.0, "The step size must be in (0, 1)"

    # Compute the batch size
    n_samples = len(data)
    batch_size = int(batch_perc * n_samples)

    # Compute a list-based cache for accessing nodes
    cached_nodes = {
        'sum': filter_nodes_type(spn, Sum),
        'leaf': filter_nodes_type(spn, Leaf)
    }

    # Initialize the random state
    if random_state is None:
        random_state = np.random.RandomState()
    elif type(random_state) == int:
        random_state = np.random.RandomState(random_state)
    elif not isinstance(random_state, np.random.RandomState):
        raise ValueError("The random state must be either None, a seed integer or a Numpy RandomState")

    # Random initialize the parameters of the SPN, if specified
    if random_init:
        # Initialize the sum parameters
        for n in cached_nodes['sum']:
            n.em_init(random_state)

        # Initialize the leaf parameters
        for n in cached_nodes['leaf']:
            n.em_init(random_state)

    # Initialize the tqdm bar, if verbose is specified
    tk = tqdm(total=np.inf, leave=None) if verbose else None

    for it in range(num_iter):
        # Sample a batch of data randomly with uniform distribution
        batch_indices = random_state.choice(n_samples, size=batch_size, replace=False)
        batch_data = data[batch_indices]

        # Forward step, obtaining the LLs at each node
        root_ll, lls = log_likelihood(spn, batch_data, return_results=True)
        mean_ll = np.mean(root_ll)

        # Backward step, compute the log-gradients required to compute the sufficient statistics
        grads = eval_backward(spn, lls)

        # Update the weights of each sum node
        for n in cached_nodes['sum']:
            children_ll = lls[list(map(lambda c: c.id, n.children))]
            stats = np.exp(children_ll - root_ll + grads[n.id])
            params = n.em_step(stats)
            update_node_parameters(n, params, step_size)

        # Update the parameters of each leaf node
        for n in cached_nodes['leaf']:
            sc = n.scope[0] if len(n.scope) == 1 else n.scope
            stats = np.exp(lls[n.id] - root_ll + grads[n.id])
            params = n.em_step(stats, batch_data[:, sc])
            update_node_parameters(n, params, step_size)

        # Update the progress bar
        if verbose:
            tk.update()
            tk.set_description('Mean LL: {:.4f}'.format(mean_ll), refresh=False)
            tk.refresh()

    if verbose:
        tk.close()
    return spn


def update_node_parameters(node, params_dict, step_size):
    """
    Update the node parameters by a batch EM step.

    :param node: The node to update.
    :param params_dict: A dictionary containing the parameters to update.
    :param step_size: The batch EM step size.
    """
    for name, value in params_dict.items():
        old_param = getattr(node, name)
        new_param = (1.0 - step_size) * old_param + step_size * value
        setattr(node, name, new_param)
