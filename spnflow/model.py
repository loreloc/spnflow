import tensorflow as tf
from spnflow.region import RegionGraph
from spnflow.layers import GaussianLayer, ProductLayer, SumLayer, DropoutLayer


def build_spn(n_features, n_classes, depth, n_sum=2, n_dists=2, n_reps=1, dropout=1.0, seed=42):
    """
    Build a RAT-SPN model.

    :param n_features: The number of features.
    :param n_classes: The number of classes.
    :param depth: The depth of the network.
    :param n_sum: The number of sum nodes.
    :param n_dists: The number of distributions.
    :param n_reps: The number of independent repetitions of the region graph.
    :param dropout: The rate of the dropout layer.
    :return: A Keras based RAT-SPN model.
    """
    # Instantiate the region graph
    region_graph = RegionGraph(n_features, depth=depth, seed=seed)

    # Generate the layers
    layers = list(reversed(region_graph.random_graph(n_reps)))

    # Instantiate the sequential model
    model = tf.keras.Sequential()

    # Add the input distributions layer
    model.add(GaussianLayer(layers[0], n_dists, input_shape=(n_features,)))

    # Alternate between product and sum layer
    for i in range(1, len(layers) - 1):
        if i % 2 == 1:
            model.add(ProductLayer())
            if dropout < 1.0:
                model.add(DropoutLayer(dropout))
        else:
            model.add(SumLayer(n_sum))

    # Add the flatten layer
    model.add(tf.keras.layers.Flatten())

    # Add the root sum layer
    model.add(SumLayer(n_classes, is_root=True))

    return model
