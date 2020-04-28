import tensorflow as tf
import tensorflow_probability as tfp


class BatchMAFLayer(tf.keras.layers.Layer):
    """
    Multiple batch Masked Autoregressive Flow sub-layer.
    """
    def __init__(self, region, n_batch, hidden_units, activation, log_scale, **kwargs):
        """
        Initialize a Autoregressive Flow transformed gaussian input distribution layer.

        :param region: The regions of the distributions.
        :param n_batch: The number of distributions.
        :param hidden_units: A list of the number of units for each layer for the autoregressive network.
        :param activation: The activation function for the autoregressive network.
        :param log_scale: Whatever to apply shift + log scale transformation or shift only.
        :param kwargs: Other arguments.
        """
        super(BatchMAFLayer, self).__init__(**kwargs)
        self.region = region
        self.n_batch = n_batch
        self.hidden_units = hidden_units
        self.activation = activation
        self.log_scale = log_scale
        self._mades = None
        self._distributions = None

    def build(self, input_shape):
        """
        Build the layer.

        :param input_shape: The input shape.
        """
        def get_bijector_fn(made):
            # Shift only bijector function
            def bijector_shift_only(x, **condition_kwargs):
                x = made(x, **condition_kwargs)
                shift = tf.squeeze(x, axis=-1)
                return tfp.bijectors.Shift(shift)

            # Shift + Log Scale bijector function
            def bijector_shift_log_scale(x, **condition_kwargs):
                x = made(x, **condition_kwargs)
                shift, log_scale = tf.unstack(x, num=2, axis=-1)
                return tfp.bijectors.Shift(shift)(tfp.bijectors.Scale(tf.exp(log_scale)))

            if self.log_scale:
                return bijector_shift_log_scale
            return bijector_shift_only

        # Set the number of paramters of the MADE model
        n_params = 2 if self.log_scale else 1

        # Initialize the MAFs distributions
        self._mades = []
        self._distributions = []
        for _ in range(self.n_batch):
            # Initialize the MADE model
            made = tfp.bijectors.AutoregressiveNetwork(
                params=n_params,
                input_order='random',
                hidden_units=self.hidden_units,
                activation=self.activation,
                kernel_regularizer=tf.keras.regularizers.l2(1e-6)
            )
            self._mades.append(made)

            # Initialize the distribution
            dist = tfp.distributions.TransformedDistribution(
                distribution=tfp.distributions.Normal(loc=0.0, scale=1.0),
                bijector=tfp.bijectors.Chain([
                    tfp.bijectors.BatchNormalization(),
                    tfp.bijectors.MaskedAutoregressiveFlow(
                        bijector_fn=get_bijector_fn(made)
                    )
                ]),
                event_shape=[len(self.region)]
            )
            self._distributions.append(dist)

        # Call the parent class's build method
        super(BatchMAFLayer, self).build(input_shape)

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
    def __init__(self, regions, n_batch, hidden_units, activation, log_scale, **kwargs):
        """
        Initialize a Autoregressive Flow transformed gaussian input distribution layer.

        :param regions: The regions of the distributions.
        :param n_batch: The number of distributions.
        :param hidden_units: A list of the number of units for each layer for the autoregressive network.
        :param activation: The activation function for the autoregressive network.
        :param log_scale: Whatever to apply shift + log scale transformation or shift only.
        :param kwargs: Other arguments.
        """
        super(MAFLayer, self).__init__(**kwargs)
        self.regions = regions
        self.n_batch = n_batch
        self.hidden_units = hidden_units
        self.activation = activation
        self.log_scale = log_scale
        self._mafs = None

    def build(self, input_shape):
        """
        Build the layer.

        :param input_shape: The input shape.
        """
        # Initialize the MAFs multiple batch sub-layers
        self._mafs = []
        for region in self.regions:
            maf = BatchMAFLayer(region, self.n_batch, self.hidden_units, self.activation, self.log_scale)
            self._mafs.append(maf)

        # Call the parent class's build method
        super(MAFLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Execute the layer on some inputs.

        :param inputs: The inputs.
        :param kwargs: Other arguments.
        :return: The log likelihood of each distribution leaf.
        """
        # Concatenate the results of each MAF multiple batch sub-layer
        return tf.stack([maf(inputs, **kwargs) for maf in self._mafs], axis=1)
