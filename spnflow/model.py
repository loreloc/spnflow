import tensorflow as tf
from spnflow.region import RegionGraph
from spnflow.layers import DistributionsLayer, ProductLayer, SumLayer


def build_spn(n_features, n_classes, depth, dist_class, n_sum=2, n_dists=2, n_reps=1, seed=42):
    """
    Build a RAT-SPN model.

    :param n_features: The number of features.
    :param n_classes: The number of classes.
    :param depth: The depth of the network.
    :param dist_class: The base distributions class.
    :param n_sum: The number of sum nodes.
    :param n_dists: The number of distributions.
    :param n_reps: The number of independent repetitions of the region graph.
    :return: A Keras based RAT-SPN model.
    """
    # Instantiate the features list
    features = list(range(n_features))

    # Instantiate the region graph
    region_graph = RegionGraph(features, seed=seed)

    # Create the input layer
    input_layer = tf.keras.layers.Input(shape=(n_features,))

    # Hidden layers build loop
    hidden_layers = []
    for _ in range(n_reps):
        region_graph.clear()
        region_graph.random_split(depth)
        region_graph.make_layers()

        # Build the input distributions layer
        layers = region_graph.layers()
        x = DistributionsLayer(layers[0], dist_class, n_dists)(input_layer)
        for i in range(1, len(layers) - 1):
            if i % 2 == 1:
                x = ProductLayer()(x)
            else:
                x = SumLayer(n_sum)(x)
        hidden_layers.append(x)

    # Build the concatenation layer
    concat_layer = tf.keras.layers.Concatenate()(hidden_layers)

    # Build the flatten layer
    flatten_layer = tf.keras.layers.Flatten()(concat_layer)

    # Build the root layer
    root_layer = SumLayer(n_classes, is_root=True)(flatten_layer)

    # Build the model
    model = tf.keras.Model(inputs=input_layer, outputs=root_layer)

    return model
