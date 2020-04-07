import tensorflow as tf
from spnflow.layers import NormalLayer, InputLayer, ProductLayer, SumLayer


def build_spn(n_classes, n_sum, n_distributions, rg_layers):
    """
    Build a RAT-SPN Sequential model.

    :param n_classes: The number of classes.
    :param n_sum: The number of sum nodes.
    :param n_distributions: The number of distributions.
    :param rg_layers: The region graph's layers.
    :return:
    """
    # Get the number of features
    n_features = len(rg_layers[-1][0])

    # Instantiate the sequential model
    spn = tf.keras.models.Sequential(name='RAT-SPN')

    # Add the input distributions layers
    input_layer = InputLayer(rg_layers[0], n_distributions, NormalLayer, input_shape=(n_features,))
    spn.add(input_layer)

    # Alternate between product and sum layers
    for i in range(1, len(rg_layers) - 1):
        if i % 2 == 1:
            product_layer = ProductLayer()
            spn.add(product_layer)
        else:
            sum_layer = SumLayer(n_sum)
            spn.add(sum_layer)

    # Flat the result
    spn.add(tf.keras.layers.Flatten())

    # Add the root sum layer
    spn.add(SumLayer(n_classes, is_root=True))

    return spn
