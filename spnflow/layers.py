import tensorflow as tf


class DistributionsLayer(tf.keras.layers.Layer):
    """
    The distributions input layer class.
    """
    def __init__(self, regions, dist_class, n_dists, **kwargs):
        """
        Initialize a distributions input layer.

        :param regions: The regions of the distributions.
        :param dist_class: The leaves distributions class.
        :param n_dists: The number of distributions.
        :param kwargs: Other arguments.
        """
        super(DistributionsLayer, self).__init__(**kwargs)
        self.regions = regions
        self.dist_class = dist_class
        self.n_dists = n_dists
        self._layers = []

    def build(self, input_shape):
        """
        Build the layer.

        :param input_shape: The input shape.
        """
        # Add the gaussian distribution leaves
        for region in self.regions:
            self._layers.append(self.dist_class(region, self.n_dists))

        # Call the parent class's build method
        super(DistributionsLayer, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        """
        Execute the layer on some inputs.

        :param inputs: The inputs.
        :return: The log likelihood of each distribution leaf.
        """
        return tf.stack([d(inputs) for d in self._layers], axis=1)


class ProductLayer(tf.keras.layers.Layer):
    """
    Product node layer class.
    """
    def __init__(self, **kwargs):
        """
        Initialize the Product layer.

        :param kwargs: Parent class arguments.
        """
        super(ProductLayer, self).__init__(**kwargs)
        self.n_regions = None
        self.n_nodes = None

    def build(self, input_shape):
        """
        Build the layer.

        :param input_shape: The input shape.
        """
        # Set the number of child regions and the number of product nodes per partition
        self.n_regions = input_shape[1]
        self.n_nodes = input_shape[2] ** 2

        # Call the parent class build method
        super(ProductLayer, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        """
        Evaluate the layer given some inputs.

        :param inputs: The inputs.
        :return: The tensor result of the layer.
        """
        dist0 = tf.gather(inputs, [i for i in range(self.n_regions) if i % 2 == 0], axis=1)
        dist1 = tf.gather(inputs, [i for i in range(self.n_regions) if i % 2 == 1], axis=1)
        result = tf.expand_dims(dist0, 2) + tf.expand_dims(dist1, 3)
        return tf.reshape(result, [-1, self.n_regions // 2, self.n_nodes])


class SumLayer(tf.keras.layers.Layer):
    """
    Sum node layer.
    """
    def __init__(self, n_sum, is_root=False, **kwargs):
        """
        Initialize the sum layer.

        :param n_sum: The number of sum node per region.
        :param is_root: A boolean indicating if the sum layer is a root layer.
        :param kwargs: Parent class arguments.
        """
        super(SumLayer, self).__init__(**kwargs)
        self.n_sum = n_sum
        self.is_root = is_root
        self.kernel = None

    def build(self, input_shape):
        """
        Build the layer.

        :param input_shape: The input shape.
        """
        # Set the kernel shape
        kernel_shape = None
        if self.is_root:
            kernel_shape = (1, input_shape[1], self.n_sum)
        else:
            kernel_shape = (input_shape[1], input_shape[2], self.n_sum)

        # Construct the weights
        self.kernel = tf.Variable(
            initial_value=tf.random.normal(kernel_shape, stddev=5e-1),
            trainable=True
        )

        # Call the parent class build method
        super(SumLayer, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        """
        Evaluate the layer given some inputs.

        :param inputs: The inputs.
        :return: The tensor result of the layer.
        """
        # Calculate the log likelihood using the logsumexp trick
        x = tf.expand_dims(inputs, axis=-1)
        x = x + tf.math.log_softmax(self.kernel, axis=2)
        x = tf.math.reduce_logsumexp(x, axis=-2)
        return x
