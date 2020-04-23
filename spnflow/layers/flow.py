import tensorflow as tf
import tensorflow_probability as tfp


class BatchMAFLayer(tf.keras.layers.Layer):
    """
    Multiple batch Masked Autoregressive Flow sub-layer.
    """
    def __init__(self, region, n_batch, hidden_units, regularization, activation, **kwargs):
        """
        Initialize a Autoregressive Flow transformed gaussian input distribution layer.

        :param region: The regions of the distributions.
        :param n_batch: The number of distributions.
        :param hidden_units: A list of the number of units for each layer for the autoregressive network.
        :param regularization: The regularization factor for the autoregressive network kernels.
        :param activation: The activation function for the autoregressive network.
        :param kwargs: Other arguments.
        """
        super(BatchMAFLayer, self).__init__(**kwargs)
        self.region = region
        self.n_batch = n_batch
        self.hidden_units = hidden_units
        self.regularization = regularization
        self.activation = activation
        self._mades = None
        self._distributions = None

    def build(self, input_shape):
        """
        Build the layer.

        :param input_shape: The input shape.
        """
        # Initialize the MAFs distributions
        self._mades = []
        self._distributions = []
        for _ in range(self.n_batch):
            # Initialize the MADE model
            made = tfp.bijectors.AutoregressiveNetwork(
                params=2,
                use_bias=False,
                input_order='random',
                hidden_units=self.hidden_units,
                activation=self.activation,
                kernel_regularizer=tf.keras.regularizers.l2(self.regularization)
            )
            self._mades.append(made)

            # Initialize the distribution
            dist = tfp.distributions.TransformedDistribution(
                distribution=tfp.distributions.Normal(loc=0.0, scale=1.0),
                bijector=tfp.bijectors.MaskedAutoregressiveFlow(made),
                event_shape=[len(self.region)]
            )
            self._distributions.append(dist)

        # Call the parent class's build method
        super(BatchMAFLayer, self).build(input_shape)

    @tf.function
    def call(self, inputs, **kwargs):
        """
        Execute the layer on some inputs.

        :param inputs: The inputs.
        :param kwargs: Other arguments.
        :return: The log likelihood of each distribution leaf.
        """
        # Concatenate the result of each distribution batch
        return tf.stack([d.log_prob(tf.gather(inputs, self.region, axis=1)) for d in self._distributions], axis=1)


class MAFLayer(tf.keras.layers.Layer):
    """
    Masked Autoregressive Flow layer.
    """
    def __init__(self, regions, n_batch, hidden_units, regularization, activation, **kwargs):
        """
        Initialize a Autoregressive Flow transformed gaussian input distribution layer.

        :param regions: The regions of the distributions.
        :param n_batch: The number of distributions.
        :param hidden_units: A list of the number of units for each layer for the autoregressive network.
        :param regularization: The regularization factor for the autoregressive network kernels.
        :param activation: The activation function for the autoregressive network.
        :param kwargs: Other arguments.
        """
        super(MAFLayer, self).__init__(**kwargs)
        self.regions = regions
        self.n_batch = n_batch
        self.hidden_units = hidden_units
        self.regularization = regularization
        self.activation = activation
        self._mafs = None

    def build(self, input_shape):
        """
        Build the layer.

        :param input_shape: The input shape.
        """
        # Initialize the MAFs multiple batch sub-layers
        self._mafs = []
        for region in self.regions:
            maf = BatchMAFLayer(region, self.n_batch, self.hidden_units, self.regularization, self.activation)
            self._mafs.append(maf)

        # Call the parent class's build method
        super(MAFLayer, self).build(input_shape)

    @tf.function
    def call(self, inputs, **kwargs):
        """
        Execute the layer on some inputs.

        :param inputs: The inputs.
        :param kwargs: Other arguments.
        :return: The log likelihood of each distribution leaf.
        """
        # Concatenate the results of each MAF multiple batch sub-layer
        return tf.stack([maf(inputs, **kwargs) for maf in self._mafs], axis=1)
