import numpy as np

from tqdm import tqdm
from scipy.special import softmax
from scipy.stats import ttest_1samp
from deeprob.spn.utils.filter import filter_nodes_type
from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.structure.node import Sum, Mul, bfs
from deeprob.spn.algorithms.inference import log_likelihood
from deeprob.spn.algorithms.gradient import eval_backward


def expectation_maximization(
        spn,
        data,
        max_iter=100,
        p_thresh=0.1,
        random_init=True,
        random_state=None,
        verbose=True
):
    """
    Learn the parameters of a SPN by Expectation-Maximization (EM).
    See https://arxiv.org/abs/1604.07243 and https://arxiv.org/abs/2004.06231 for details.

    :param spn: The spn structure.
    :param data: The data to use to learn the parameters.
    :param max_iter: The maximum number of iterations.
    :param p_thresh: The threshold for the T-Test's p-value to ensure a statistically relevant EM step improvement.
    :param random_init: Whether to random initialize the weights of the SPN.
    :param random_state: The random state to use. It can be None.
    :param verbose: Whether to enable verbose learning.
    :return: The spn with learned parameters.
    """
    assert max_iter > 0, "The number of maximum iterations must be greater than zero"
    assert 0.0 < p_thresh < 0.5, "The T-Test's p-value threshold must be in (0.0, 0.5)"

    # Random initialize the parameters of the SPN, if specified
    if random_init:
        # Initialize the random state
        if random_state is None:
            random_state = np.random.RandomState(42)
        elif type(random_state) == int:
            random_state = np.random.RandomState(random_state)
        else:
            random_state = random_state
        random_initialize_parameters(spn, random_state)

    # Initialize the tqdm bar, if verbose is specified
    tk = tqdm(total=np.inf, leave=None) if verbose else None

    it = 0
    last_mean_ll = -np.inf
    while it < max_iter:
        # Forward step, obtaining the LLs at each node
        root_ll, lls = log_likelihood(spn, data, return_results=True)
        mean_ll = np.mean(root_ll)

        # Update the progress bar's description
        if verbose:
            tk.set_description('Mean LL: {:.4f}'.format(mean_ll))

        # Do a T-Test to ensure statistically relevant improvements on the LL
        _, p = ttest_1samp(root_ll, last_mean_ll, alternative='greater')
        if p > p_thresh:
            break

        # Backward step, compute the log-gradients required to compute the expected sufficient statistics
        grads = eval_backward(spn, lls)

        # Update the weights of each sum node
        for s in filter_nodes_type(spn, Sum):
            update_sum_weights(s, root_ll, lls, grads)

        # Update the counter, statistics and progress bar
        it += 1
        last_mean_ll = mean_ll
        if verbose:
            tk.update()
            tk.refresh()

    if verbose:
        tk.close()
    return spn


def update_sum_weights(node, root_ll, lls, grads):
    """
    Update the weights of a sum node.

    :param node: The sum node.
    :param root_ll: The log-likelihood at the root node of the SPN.
    :param lls: The log-likelihoods at each node in the SPN.
    :param grads: The log-gradients w.r.t. the sum node.
    """
    stats = np.zeros(len(node.weights), dtype=np.float32)
    for i, (c, w) in enumerate(zip(node.children, node.weights)):
        stats[i] = w * np.sum(np.exp(lls[c.id] - root_ll + grads[node.id]))
    node.weights = stats / np.sum(stats)


def random_initialize_parameters(spn, random_state):
    """
    Random initialize the parameters of a SPN.

    :param spn: The SPN to random initialize.
    :param random_state: The random state to use.
    """
    def rand_init(node):
        if isinstance(node, Sum):
            w = random_state.randn(len(node.weights)).astype(np.float32)
            node.weights = softmax(w)
        elif isinstance(node, Mul):
            pass
        elif isinstance(node, Leaf):
            pass  # TODO (possible handling EM for any kind of leaves)
        else:
            raise NotImplementedError(
                'Random parameters initialization not implemented for type {}'.format(node.__class__.__name__)
            )

    bfs(spn, rand_init)


if __name__ == '__main__':
    from experiments.datasets import load_binary_dataset
    from deeprob.spn.structure.leaf import Bernoulli
    from deeprob.spn.learning.wrappers import learn_estimator

    data_train, data_valid, data_test = load_binary_dataset('../../../experiments/datasets', 'netflix', raw=True)
    n_features = data_train.shape[1]
    distributions = [Bernoulli] * n_features
    spn = learn_estimator(
        data_train, distributions, learn_leaf='mle',
        split_rows='gmm', split_cols='gvs'
    )

    test_ll = log_likelihood(spn, data_test)
    print('Mean LL: {}'.format(np.mean(test_ll)))

    spn = expectation_maximization(spn, data_train)

    test_ll = log_likelihood(spn, data_test)
    print('Mean LL: {}'.format(np.mean(test_ll)))
