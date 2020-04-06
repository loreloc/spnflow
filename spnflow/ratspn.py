import tensorflow as tf
from layers import NormalLayer, ProductLayer, SumLayer


class RatSpn(tf.keras.Model):
    """
    Randomized and Tensorized Sum-Product Network Keras model class.
    """
    def __init__(self, n_classes, n_sum, n_distributions, rg_layers):
        """
        Initialize a RAT-SPN.
        :param n_classes: The number of classes.
        :param n_sum: The number of sum nodes for each sum layer.
        :param n_distributions: The number of distributions for each distribution leaf.
        :param rg_layers: The random graph's layers.
        """
        super(RatSpn, self).__init__()
        self.n_classes = n_classes
        self.n_sum = n_sum
        self.n_features = len(rg_layers[-1][0])
        self.n_distributions = n_distributions
        self._rg_layers = rg_layers
        self._base_layers = []
        self._layers = []

    def input_shape(self):
        """
        Get the input shape.

        :return: The input shape.
        """
        return self.n_features

    def output_shape(self):
        """
        Get the output shape.

        :return: The output shape.
        """
        return self.n_classes

    def build(self, input_shape):
        """
        Build the layers.

        :param input_shape: The input shape.
        """
        # Add the input distributions layers
        for region in self._rg_layers[0]:
            self._base_layers.append(NormalLayer(region, self.n_distributions))

        # Alternate between product and sum layers
        for i in range(1, len(self._rg_layers) - 1):
            if i % 2 == 1:
                product_layer = ProductLayer()
                self._layers.append(product_layer)
            else:
                sum_layer = SumLayer(self.n_sum)
                self._layers.append(sum_layer)

        # Flat the result
        self._layers.append(tf.keras.layers.Flatten())

        # Add the root sum layer
        self._layers.append(SumLayer(self.n_classes))

        # Call the parent class build method
        super(RatSpn, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        """
        Run the model on some inputs.

        :param inputs: The inputs.
        :param training: Training flag.
        :param mask: Mask flag.
        :return: The result of the model.
        """
        # Compute the base distributions log likelihoods
        x = None
        if training:
            x = tf.stack([dist(inputs) for dist in self._base_layers], axis=1)
        else:
            x = tf.stack([dist(inputs) for dist in self._base_layers], axis=0)

        # Compute the each layers
        for i in range(len(self._layers)):
            x = self._layers[i](x)

        return x
