import tensorflow as tf
import tensorflow_probability as tfp
from spnflow.tensorflow.model.rat import RatSpn


class AutoregressiveRatSpn(tf.keras.Model):
    """RAT-SPN base distribution improved with Autoregressive Normalizing Flows."""
    def __init__(self,
                 depth=2,
                 n_batch=2,
                 n_sum=2,
                 n_repetitions=1,
                 dropout=0.0,
                 optimize_scale=True,
                 n_mafs=3,
                 hidden_units=[32, 32],
                 activation='relu',
                 regularization=1e-6,
                 batch_norm=True,
                 rand_state=None,
                 **kwargs
                 ):
        """
        Initialize an Autoregressive RAT-SPN.

        :param depth: The depth of the network.
        :param n_batch: The number of distributions.
        :param n_sum: The number of sum nodes.
        :param n_repetitions: The number of independent repetitions of the region graph.
        :param dropout: The rate of the dropout layers.
        :param optimize_scale: Whatever to train scale and mean jointly.
        :param n_mafs: The number of chained Masked Autoregressive Flows (MAFs).
        :param hidden_units: A list of the number of units for each layer of the autoregressive network.
        :param activation: The activation function for the autoregressive network.
        :param regularization: The L2 regularization weight for the autoregressive network.
        :param rand_state: The random state to use to generate the RAT-SPN model.
        :param batch_norm: Whatever to use batch normalization after each MAF layer.
        :param kwargs: Other arguments.
        """
        super(AutoregressiveRatSpn, self).__init__(**kwargs)
        self.depth = depth
        self.n_batch = n_batch
        self.n_sum = n_sum
        self.n_repetitions = n_repetitions
        self.dropout = dropout
        self.optimize_scale = optimize_scale
        self.n_mafs = n_mafs
        self.hidden_units = hidden_units
        self.activation = activation
        self.regularization = regularization
        self.batch_norm = batch_norm
        self.rand_state = rand_state
        self.spn = None
        self.mades = None
        self.mafs = None
        self.bns = None
        self.bijector = None

    def build(self, input_shape):
        """
        Build the model.

        :param input_shape: The input shape.
        """
        # Get a bijector function given a Masked Autoregressive Density Estimator (MADE)
        def get_bijector_fn(made):
            # Shift + Log Scale bijector function
            def bijector_shift_log_scale(x, **condition_kwargs):
                x = made(x, **condition_kwargs)
                shift, log_scale = tf.unstack(x, num=2, axis=-1)
                return tfp.bijectors.Shift(shift)(tfp.bijectors.Scale(tf.exp(log_scale)))
            return bijector_shift_log_scale

        # Build the RAT-SPN that represents the base distribution
        _, n_features = input_shape
        self.spn = RatSpn(
            n_features,
            self.depth,
            self.n_batch,
            self.n_sum,
            self.n_repetitions,
            self.dropout,
            self.optimize_scale,
            self.rand_state
        )
        self.spn.build(input_shape)

        # Build the MADE models
        self.mades = []
        input_order = 'left-to-right'
        for _ in range(self.n_mafs):
            # Build the MADE model
            made = tfp.bijectors.AutoregressiveNetwork(
                params=2,
                input_order=input_order,
                hidden_units=self.hidden_units,
                activation=self.activation,
                kernel_regularizer=tf.keras.regularizers.l2(self.regularization)
            )
            self.mades.append(made)
            # Change the input order
            input_order = 'right-to-left' if input_order == 'left-to-right' else 'left-to-right'

        # Build the MAFs
        self.mafs = []
        for made in self.mades:
            maf = tfp.bijectors.MaskedAutoregressiveFlow(
                bijector_fn=get_bijector_fn(made)
            )
            self.mafs.append(maf)

        # Build the bijector by chaining multiple MAFs
        self.bns = []
        bijectors = []
        for maf in self.mafs:
            # Append the maf bijection
            bijectors.append(maf)
            # Append batch normalization bijection, if specified
            if self.batch_norm:
                bn = tfp.bijectors.BatchNormalization()
                self.bns.append(bn)
                bijectors.append(bn)
        self.bijector = tfp.bijectors.Chain(bijectors)

        # Call the parent class build method
        super(AutoregressiveRatSpn, self).build(input_shape)

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        """
        Call the model.

        :param inputs: The inputs tensor.
        :param training: Whatever the model is training or not.
        :param kwargs: Other arguments.
        :return: The output of the model.
        """
        u = self.bijector.inverse(inputs, training=training)
        p = self.spn(u, training=training, **kwargs)
        d = self.bijector.inverse_log_det_jacobian(inputs, event_ndims=1, training=training)
        p = p + tf.expand_dims(d, axis=-1)
        return p

    @tf.function
    def sample(self, n_samples=1):
        """
        Sample from the modeled distribution.

        :param n_samples: The number of samples.
        :return: The samples.
        """
        samples = self.spn.sample(n_samples)
        samples = self.bijector.forward(samples)
        return samples
