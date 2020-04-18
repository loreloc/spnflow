import tensorflow as tf
import tensorflow_probability as tfp


class AutoregressiveFlowLayer(tf.keras.layers.Layer):
    """
    Autoregressive Flow layer.
    """
    def __init__(self, regions, n_batch, hidden_units, regularization, **kwargs):
        """
        Initialize a Autoregressive Flow transformed gaussian input distribution layer.

        :param regions: The regions of the distributions.
        :param n_batch: The number of distributions.
        :param hidden_units: A list of the number of units for each layer for the autoregressive network.
        :param regularization: The regularization factor for the autoregressive network kernels.
        :param kwargs: Other arguments.
        """
        super(AutoregressiveFlowLayer, self).__init__(**kwargs)
        self.regions = regions
        self.n_batch = n_batch
        self.hidden_units = hidden_units
        self.regularization = regularization
        self._mafs = None
        self._mades = None

    def build(self, input_shape):
        """
        Build the layer.

        :param input_shape: The input shape.
        """
        # Initialize the MADEs models (multiple batches for each region)
        self._mades = []
        for _ in self.regions:
            batch_mades = []
            for _ in range(self.n_batch):
                made = tfp.bijectors.AutoregressiveNetwork(
                    params=2, input_order='random',
                    hidden_units=self.hidden_units, activation='relu',
                    use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(self.regularization)
                )
                batch_mades.append(made)
            self._mades.append(batch_mades)

        # Initialize the transformed distributions (MAFs)
        self._mafs = []
        for region, batch_mades in zip(self.regions, self._mades):
            batch_dists = []
            for made in batch_mades:
                dist = tfp.distributions.TransformedDistribution(
                    distribution=tfp.distributions.Normal(loc=0.0, scale=1.0),
                    bijector=tfp.bijectors.MaskedAutoregressiveFlow(made),
                    event_shape=[len(region)]
                )
                batch_dists.append(dist)
            self._mafs.append(batch_dists)

        # Call the parent class's build method
        super(AutoregressiveFlowLayer, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        """
        Execute the layer on some inputs.

        :param inputs: The inputs.
        :return: The log likelihood of each distribution leaf.
        """
        # Mask the inputs over the regions
        inputs = [tf.gather(inputs, r, axis=1) for r in self.regions]

        # Concatenate the results of each batch distributions
        ll = [
            tf.stack([d.log_prob(y) for d in batch_dists], axis=1)
            for y, batch_dists in zip(inputs, self._mafs)
        ]

        x = tf.stack(ll, axis=1)
        return x
