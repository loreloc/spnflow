from spnflow.layers.rat import *
from spnflow.layers.flow import *
from spnflow.utils.region import RegionGraph


def build_rat_spn_flow(
        n_features,
        n_classes,
        depth=2,
        n_batch=2,
        hidden_units=[32, 32],
        activation='relu',
        n_sum=2,
        n_repetitions=1,
        log_scale=False,
        seed=42
        ):
    """
    Build a RAT-SPN model with Autoregressive Flow distributions at leaves.

    :param n_features: The number of features.
    :param n_classes: The number of classes.
    :param depth: The depth of the network.
    :param n_batch: The number of distributions.
    :param hidden_units: A list of the number of units for each layer for the autoregressive network.
    :param activation: The activation function for the autoregressive network.
    :param n_sum: The number of sum nodes.
    :param n_repetitions: The number of independent repetitions of the region graph.
    :param log_scale: Whatever to apply shift + log scale transformation or shift only.
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
    input_layer = MAFLayer(
        layers[0],
        n_batch,
        hidden_units,
        activation,
        log_scale,
        input_shape=(n_features,)
    )
    model.add(input_layer)

    # Alternate between product and sum layer
    for i in range(1, len(layers) - 1):
        if i % 2 == 1:
            model.add(ProductLayer())
        else:
            model.add(SumLayer(n_sum))

    # Add the flatten layer
    model.add(tf.keras.layers.Flatten())

    # Add the root sum layer
    model.add(RootLayer(n_classes))

    return model
