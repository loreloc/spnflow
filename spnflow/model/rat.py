import numpy as np
from spnflow.layers.rat import *
from spnflow.utils.region import RegionGraph


def build_rat_spn(
        n_features,
        depth=2,
        n_batch=2,
        n_sum=2,
        n_repetitions=1,
        dropout=0.0,
        optimize_scale=True,
        rand_state=None
        ):
    """
    Build a RAT-SPN model.

    :param n_features: The number of features.
    :param depth: The depth of the network.
    :param n_batch: The number of distributions.
    :param n_sum: The number of sum nodes.
    :param n_repetitions: The number of independent repetitions of the region graph.
    :param dropout: The rate of the dropout layers.
    :param optimize_scale: Whatever to train scale and mean jointly.
    :param rand_state: The random state used to generate the random graph.
    :return: A Keras based RAT-SPN model.
    """
    # If necessary, instantiate a random state
    if rand_state is None:
        rand_state = np.random.RandomState(42)

    # Instantiate the region graph
    region_graph = RegionGraph(n_features, depth, rand_state)

    # Generate the layers
    layers = region_graph.make_layers(n_repetitions)
    layers = list(reversed(layers))

    # Instantiate the sequential model
    model = tf.keras.Sequential()

    # Add the input layer
    model.add(tf.keras.layers.InputLayer(input_shape=(n_features,)))

    # Add the distributions layer
    model.add(GaussianLayer(layers[0], n_batch, optimize_scale))

    # Alternate between product and sum layer
    for i in range(1, len(layers) - 1):
        if i % 2 == 1:
            model.add(ProductLayer())
            if dropout > 0.0:
                model.add(DropoutLayer(dropout))
        else:
            model.add(SumLayer(n_sum))

    # Add the flatten layer
    model.add(tf.keras.layers.Flatten())

    # Add the root sum layer
    model.add(RootLayer())

    return model
