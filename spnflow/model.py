import tensorflow as tf
from spnflow.region import RegionGraph
from spnflow.layers import GaussianLayer, ProductLayer, SumLayer


def build_spn(n_features, n_classes, depth, n_sum=2, n_dists=2, n_reps=1, seed=42):
    """
    Build a RAT-SPN model.

    :param n_features: The number of features.
    :param n_classes: The number of classes.
    :param depth: The depth of the network.
    :param n_sum: The number of sum nodes.
    :param n_dists: The number of distributions.
    :param n_reps: The number of independent repetitions of the region graph.
    :return: A Keras based RAT-SPN model.
    """
    # Instantiate the region graph
    region_graph = RegionGraph(n_features, depth=depth, seed=seed)

    # Create the input layer
    input_layer = tf.keras.layers.Input(shape=(n_features,))

    # Hidden layers build loop
    hidden_layers = []
    for _ in range(n_reps):
        # Compute a random graph layers
        layers = list(reversed(region_graph.random_layers()))

        # Build the input distributions layer
        x = GaussianLayer(layers[0], n_dists)(input_layer)

        # Alternate between product and sum layer
        for i in range(1, len(layers) - 1):
            if i % 2 == 1:
                x = ProductLayer()(x)
            else:
                x = SumLayer(n_sum)(x)
        hidden_layers.append(x)

    # Build the concatenation layer
    concat_layer = tf.keras.layers.Concatenate()(hidden_layers)

    # Build the root layer
    root_layer = SumLayer(n_classes)(concat_layer)

    # Build the flatten layer
    output_layer = tf.keras.layers.Flatten()(root_layer)

    # Build the model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model
