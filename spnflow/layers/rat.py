import tensorflow as tf
import tensorflow_probability as tfp


class GaussianLayer(tf.keras.layers.Layer):
    """
    The Gaussian distributions input layer class.
    """
    def __init__(self, regions, n_batch, optimize_scale, **kwargs):
        """
        Initialize a Gaussian distributions input layer.

        :param regions: The regions of the distributions.
        :param n_batch: The number of distributions.
        :param optimize_scale: Whatever to train scale and mean jointly.
        :param kwargs: Other arguments.
        """
        super(GaussianLayer, self).__init__(**kwargs)
        self.regions = regions
        self.n_batch = n_batch
        self.optimize_scale = optimize_scale
        self._means = None
        self._scales = None
        self._distributions = None

    def build(self, input_shape):
        """
        Build the layer.

        :param input_shape: The input shape.
        """
        # Create the means variables
        self._means = [
            tf.Variable(
                tf.random.normal(shape=(self.n_batch, len(r)), stddev=1e-1),
                trainable=True
            )
            for r in self.regions
        ]

        # Create the scales variables
        if self.optimize_scale:
            self._scales = [
                tf.Variable(
                    1.0 + 1e-1 * (1.0 - 2.0 * tf.math.sigmoid(tf.random.normal(shape=(self.n_batch, len(r))))),
                    trainable=True
                )
                for r in self.regions
            ]
        else:
            self._scales = [
                tf.Variable(
                    tf.ones(shape=(self.n_batch, len(r))),
                    trainable=False
                )
                for r in self.regions
            ]

        # Create the multi-batch multivariate distributions
        self._distributions = [
            tfp.distributions.MultivariateNormalDiag(mean, scale)
            for mean, scale in zip(self._means, self._scales)
        ]

        # Call the parent class's build method
        super(GaussianLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Execute the layer on some inputs.

        :param inputs: The inputs.
        :param kwargs: Other arguments.
        :return: The log likelihood of each distribution leaf.
        """
        # Concatenate the results of each distribution's batch result
        ll = [
            d.log_prob(tf.expand_dims(tf.gather(inputs, r, axis=1), axis=1))
            for r, d in zip(self.regions, self._distributions)
        ]

        x = tf.stack(ll, axis=1)
        return x


class ProductLayer(tf.keras.layers.Layer):
    """
    Product node layer class.
    """
    def __init__(self, **kwargs):
        """
        Initialize the Product layer.

        :param partitions: The partitions list.
        :param kwargs: Parent class arguments.
        """
        super(ProductLayer, self).__init__(**kwargs)
        self.n_partitions = None
        self.n_nodes = None

    def build(self, input_shape):
        """
        Build the layer.

        :param input_shape: The input shape.
        """
        # Set the number of partitions and the number of nodes for each region channel
        self.n_partitions = input_shape[1] // 2
        self.n_nodes = input_shape[2]

        # Call the parent class build method
        super(ProductLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Evaluate the layer given some inputs.

        :param inputs: The inputs.
        :param kwargs: Other arguments.
        :return: The tensor result of the layer.
        """
        # Compute the outer product (the "outer sum" in log domain)
        x = tf.reshape(inputs, [-1, self.n_partitions, 2, self.n_nodes])  # (n, p, 2, s)
        x0 = tf.expand_dims(x[:, :, 0], 3)  # (n, p, s, 1)
        x1 = tf.expand_dims(x[:, :, 1], 2)  # (n, p, 1, s)
        x = x0 + x1  # (n, p, s, s)
        x = tf.reshape(x, [-1, self.n_partitions, self.n_nodes ** 2])  # (n, p, s * s)
        return x


class SumLayer(tf.keras.layers.Layer):
    """
    Sum node layer.
    """
    def __init__(self, n_sum, **kwargs):
        """
        Initialize the sum layer.

        :param n_sum: The number of sum node per region.
        :param kwargs: Parent class arguments.
        """
        super(SumLayer, self).__init__(**kwargs)
        self.n_sum = n_sum
        self.kernel = None

    def build(self, input_shape):
        """
        Build the layer.

        :param input_shape: The input shape.
        """
        # Set the kernel shape
        kernel_shape = (input_shape[1], self.n_sum, input_shape[2])

        # Construct the weights
        self.kernel = tf.Variable(
            initial_value=tf.random.normal(kernel_shape, stddev=1e-1),
            trainable=True
        )

        # Call the parent class build method
        super(SumLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Evaluate the layer given some inputs.

        :param inputs: The inputs.
        :param kwargs: Other arguments.
        :return: The tensor result of the layer.
        """
        # Calculate the log likelihood using the "logsumexp" trick
        x = tf.expand_dims(inputs, axis=2)  # (n, p, 1, k)
        w = tf.math.log_softmax(self.kernel, axis=2)  # (n, p, k)
        x = tf.math.reduce_logsumexp(x + w, axis=-1)  # (n, p, s)
        return x


class RootLayer(tf.keras.layers.Layer):
    """
    Root sum node layer.
    """
    def __init__(self, **kwargs):
        """
        Initialize the root layer.

        :param kwargs: Parent class arguments.
        """
        super(RootLayer, self).__init__(**kwargs)
        self.kernel = None

    def build(self, input_shape):
        """
        Build the layer.

        :param input_shape: The input shape.
        """
        # Construct the weights
        kernel_shape = (1, input_shape[1])
        self.kernel = tf.Variable(
            initial_value=tf.random.normal(kernel_shape, stddev=1e-1),
            trainable=True
        )

        # Call the parent class build method
        super(RootLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Evaluate the layer given some inputs.

        :param inputs: The inputs.
        :param kwargs: Other arguments.
        :return: The tensor result of the layer.
        """
        # Calculate the log likelihood using the "logsumexp" trick
        x = tf.expand_dims(inputs, axis=1)  # (n, 1, k)
        w = tf.math.log_softmax(self.kernel, axis=1)  # (n, k)
        x = tf.math.reduce_logsumexp(x + w, axis=-1)  # (n, 1)
        return x


class DropoutLayer(tf.keras.layers.Layer):
    """
    Dropout layer.
    """
    def __init__(self, rate=0.2, **kwargs):
        """
        Initialize the dropout layer.

        :param rate: The dropout rate.
        """
        super(DropoutLayer, self).__init__(**kwargs)
        self.rate = 1.0 - rate

    def build(self, input_shape):
        """
        Build the layer.

        :param input_shape: The input shape.
        """
        # Call the parent class build method
        super(DropoutLayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        """
        Evaluate the layer given some inputs.

        :param inputs: The inputs.
        :param training: An boolean indicating if the layer must be used in training mode or not.
        :param kwargs: Other arguments.
        :return: The tensor result of the layer.
        """
        if not training:
            return inputs

        # Apply the dropout to the inputs
        x = tf.random.uniform(shape=tf.shape(inputs))
        x = tf.math.log(tf.math.floor(self.rate + x))
        return inputs + x
