from spnflow.model import build_spn
from spnflow.layers import GaussianLayer, AutoregressiveFlowLayer


def build_gaussian_spn(
        n_features, n_classes, depth=2, n_dists=2,
        n_sum=2, n_reps=1, dropout=1.0, seed=42
        ):
    """
    Build a RAT-SPN model with Gaussian distributions at leaves.

    :param n_features: The number of features.
    :param n_classes: The number of classes.
    :param depth: The depth of the network.
    :param n_sum: The number of sum nodes.
    :param n_dists: The number of distributions.
    :param n_reps: The number of independent repetitions of the region graph.
    :param dropout: The rate of the dropout layer.
    :param seed: The seed to use to randomly generate the region graph.
    :return: A Keras based RAT-SPN model.
    """
    return build_spn(
        n_features, n_classes, depth,
        base_dist_class=GaussianLayer, base_dist_params={'n_dists': n_dists},
        n_sum=n_sum, n_reps=n_reps, dropout=dropout, seed=seed
    )


def build_autoregressive_flow_spn(
        n_features, n_classes, depth=2,
        hidden_units=[32, 32], regularization=1e-6,
        n_sum=2, n_reps=1, dropout=1.0, seed=42
        ):
    """
    Build a RAT-SPN model with Autoregressive Flow distributions at leaves.

    :param n_features: The number of features.
    :param n_classes: The number of classes.
    :param depth: The depth of the network.
    :param hidden_units: A list of the number of units for each layer for the autoregressive network.
    :param regularization: The regularization factor for the autoregressive network kernels.
    :param n_sum: The number of sum nodes.
    :param n_reps: The number of independent repetitions of the region graph.
    :param dropout: The rate of the dropout layer.
    :param seed: The seed to use to randomly generate the region graph.
    :return: A Keras based RAT-SPN model.
    """
    return build_spn(
        n_features, n_classes, depth,
        base_dist_class=AutoregressiveFlowLayer,
        base_dist_params={'hidden_units': hidden_units, 'regularization': regularization},
        n_sum=n_sum, n_reps=n_reps, dropout=dropout, seed=seed
    )
