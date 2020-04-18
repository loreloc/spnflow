from spnflow.layers.rat import *
from spnflow.utils.region import RegionGraph


def build_rat_spn(
        n_features,
        n_classes,
        depth=2,
        n_batch=2,
        n_sum=2,
        n_repetitions=1,
        dropout=1.0,
        seed=42
        ):
    """
    Build a RAT-SPN model.

    :param n_features: The number of features.
    :param n_classes: The number of classes.
    :param depth: The depth of the network.
    :param n_batch: The number of distributions.
    :param n_sum: The number of sum nodes.
    :param n_repetitions: The number of independent repetitions of the region graph.
    :param dropout: The rate of the dropout layer.
    :param seed: The seed to use to randomly generate the region graph.
    :return: A Keras based RAT-SPN model.
    """
    # Instantiate the region graph
    region_graph = RegionGraph(n_features, depth=depth, seed=seed)

    # Generate the layers
    layers = list(reversed(region_graph.random_graph(n_repetitions)))

    # Instantiate the sequential model
    model = tf.keras.Sequential()

    # Add the input distributions layer
    model.add(GaussianLayer(layers[0], n_batch, input_shape=(n_features,)))

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
    model.add(RootLayer(n_classes))

    return model
