import numpy as np
import tensorflow as tf
from spnflow.utils.region import RegionGraph
from spnflow.tensorflow.layers.rat import GaussianLayer, ProductLayer, SumLayer, RootLayer, DropoutLayer


class RatSpn(tf.keras.Model):
    """RAT-SPN Keras model class."""
    def __init__(self,
                 n_features,
                 depth=2,
                 n_batch=2,
                 n_sum=2,
                 n_repetitions=1,
                 dropout=0.0,
                 optimize_scale=True,
                 rand_state=None,
                 **kwargs
                 ):
        """
        Initialize a RAT-SPN.

        :param depth: The depth of the network.
        :param n_batch: The number of distributions.
        :param n_sum: The number of sum nodes.
        :param n_repetitions: The number of independent repetitions of the region graph.
        :param dropout: The rate of the dropout layers.
        :param optimize_scale: Whatever to train scale and mean jointly.
        :param rand_state: The random state used to generate the random graph.
        :param kwargs: Other arguments.
        """
        super(RatSpn, self).__init__(**kwargs)
        self.n_features = n_features
        self.depth = depth
        self.n_batch = n_batch
        self.n_sum = n_sum
        self.n_repetitions = n_repetitions
        self.dropout = dropout
        self.optimize_scale = optimize_scale
        self.rand_state = rand_state
        self.base_layer = None
        self.inner_layers = None
        self.flatten_layer = None
        self.root_layer = None

        # If necessary, instantiate a random state
        if self.rand_state is None:
            self.rand_state = np.random.RandomState(42)

        # Instantiate the region graph
        region_graph = RegionGraph(self.n_features, self.depth, self.rand_state)

        # Generate the layers
        self.rg_layers = region_graph.make_layers(self.n_repetitions)
        self.rg_layers = list(reversed(self.rg_layers))

    def build(self, input_shape):
        """
        Build the model.

        :param input_shape: The input shape.
        """

        # Add the base distributions layer
        self.base_layer = GaussianLayer(self.depth, self.rg_layers[0], self.n_batch, self.optimize_scale)

        # Alternate between product and sum layer
        self.inner_layers = []
        for i in range(1, len(self.rg_layers) - 1):
            if i % 2 == 1:
                self.inner_layers.append(ProductLayer())
            else:
                if self.dropout > 0.0:
                    self.inner_layers.append(DropoutLayer(self.dropout))
                self.inner_layers.append(SumLayer(self.n_sum))

        # Add the flatten layer
        self.flatten_layer = tf.keras.layers.Flatten()

        # Add the sum root layer
        self.root_layer = RootLayer()

        # Call the parent class build method
        super(RatSpn, self).build(input_shape)

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        """
        Call the model.

        :param inputs: The inputs tensor.
        :param training: Whatever the model is training or not.
        :param kwargs: Other arguments.
        :return: The output of the model.
        """
        # Calculate the base log-likelihoods
        x = self.base_layer(inputs, training=training, **kwargs)

        # Forward through the inner layers
        for layer in self.inner_layers:
            x = layer(x, training=training, **kwargs)

        # Flatten the result and forward through the sum root layer
        x = self.flatten_layer(x)
        x = self.root_layer(x)
        return x

    @tf.function
    def sample(self, n_samples=1):
        """
        Sample from the modeled distribution.

        :param n_samples: The number of samples.
        :return: The samples.
        """
        idx = self.root_layer.sample(n_samples)
        idx = tf.transpose(idx)
        idx_partition = idx // (self.n_sum ** 2)
        idx_offset = idx % (self.n_sum ** 2)

        for layer in reversed(self.inner_layers):
            if isinstance(layer, DropoutLayer):
                continue
            idx_partition, idx_offset = layer.sample(n_samples, idx_partition, idx_offset)

        samples = self.base_layer.sample(n_samples, idx_partition, idx_offset)
        return samples
