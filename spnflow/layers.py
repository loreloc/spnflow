import tensorflow as tf
import tensorflow_probability as tfp


class NormalLayer(tf.keras.layers.Layer):
    """
    Multi-batch Multivariate Normal distribution layer class.
    """
    def __init__(self, region, n_batch):
        """
        Initialize the multi-batch and multivariate normal distribution.

        :param n_batch: The number of batches.
        :param n_features: The number of random variables.
        """
        super(NormalLayer, self).__init__()
        self.region = region
        self.n_batch = n_batch
        self._mean = None
        self._scale = None
        self._distribution = None

    def build(self, input_shape):
        """
        Build the multi-batch and multivariate distribution.
        """
        # Create the mean multi-batch multivariate variable
        self._mean = tf.Variable(
            tf.random.normal(shape=(self.n_batch, len(self.region)), stddev=0.1),
            trainable=True
        )

        # Create the scale matrix multi-batch multivariate variable
        self._scale = tfp.util.TransformedVariable(
            tf.eye(len(self.region), batch_shape=(self.n_batch,)),
            tfp.bijectors.FillScaleTriL(),
            trainable=True
        )

        # Create the multi-batch multivariate variable
        self._distribution = tfp.distributions.MultivariateNormalTriL(self._mean, tf.linalg.cholesky(self._scale))

        # Call the parent class build method
        super(NormalLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape.

        :param input_shape: The input shape.
        :return: The output shape.
        """
        return None, self.n_batch, len(self.region)

    def call(self, inputs, training=None, **kwargs):
        """
        Compute the log likelihoods given some inputs.

        :param inputs: The inputs.
        :param training: A boolean indicating if it's training.
        :param kwargs: Other arguments.
        :return: The log likelihoods.
        """
        # Calculate the log likelihoods given some inputs
        masked_input = tf.gather(inputs, self.region, axis=1)
        masked_input = tf.expand_dims(masked_input, 1)
        return self._distribution.log_prob(masked_input)

    def sample(self, sample_shape):
        """
        Sample some values from the distribution.

        :param sample_shape: The sample shape.
        :return: Some samples.
        """
        # Sample some values from the distributions
        return self._distribution.sample(sample_shape)


class InputLayer(tf.keras.layers.Layer):
    """
    The distributions input layer class.
    """
    def __init__(self, regions, n_distributions, **kwargs):
        """
        Initialize an input layer.

        :param regions: The regions of the distributions.
        :param n_distributions: The number of distributions.
        :param kwargs: Other arguments.
        """
        super(InputLayer, self).__init__(**kwargs)
        self.regions = regions
        self.n_distributions = n_distributions
        self._layers = []

    def build(self, input_shape):
        """
        Build the layer.

        :param input_shape: The input shape.
        """
        # Add the gaussian distribution leaves
        for region in self.regions:
            self._layers.append(NormalLayer(region, self.n_distributions))

        # Call the parent class's build method
        super(InputLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape.

        :param input_shape: The input shape.
        :return: The output shape.
        """
        return None, len(self.regions), self.n_distributions

    def call(self, inputs, training=None, **kwargs):
        """
        Execute the layer on some inputs.

        :param inputs: The inputs.
        :param training: A flag indicating if it's training.
        :param kwargs: Other arguments.
        :return: The log likelihood of each distribution leaf.
        """
        x = tf.stack([dist(inputs) for dist in self._layers], axis=1)
        return x


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

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape.

        :param input_shape: The input shape.
        :return: The output shape.
        """
        return None, input_shape[1] // 2, input_shape[2] ** 2

    def call(self, inputs, training=None, **kwargs):
        """
        Evaluate the layer given some inputs.

        :param inputs: The inputs.
        :param training: A boolean indicating if it's training.
        :param kwargs: Other arguments.
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
            kernel_shape = (1, self.n_sum, input_shape[1])
        else:
            kernel_shape = (input_shape[1], self.n_sum, input_shape[2])

        # Construct the weights as a matrix of shape (N, S, M) where N is the number of input product layers, S is the
        # number of sum nodes for each layer and M is the number of product node of each product layer
        self.kernel = tf.Variable(
            initial_value=tf.random.uniform(kernel_shape, minval=0.0, maxval=1.0),
            trainable=True
        )

        # Call the parent class build method
        super(SumLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape.
        :param input_shape: The input shape.
        :return: The output shape.
        """
        return None, self.n_partitions, self.n_sum

    def call(self, inputs, **kwargs):
        """
        Evaluate the layer  given some inputs.

        :param inputs: The inputs.
        :param kwargs: Other arguments.
        :return: The tensor result of the layer.
        """
        # Calculate the log likelihood using the logsumexp trick (3-D tensor with 2-D tensor multiplication here)
        return tf.math.log(tf.linalg.matvec(self.kernel, tf.math.exp(inputs)))
