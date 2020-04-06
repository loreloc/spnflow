import tensorflow as tf
from distributions import MultivariateNormal


class InputLayer(tf.keras.layers.Layer):
    """
    Input distributions layer class.
    """
    def __init__(self, regions, n_distributions, **kwargs):
        """
        Initialize the Input of distributions layer.

        :param regions: The list of regions of the input layer.
        :param n_distributions: The number of distributions.
        :param kwargs: Parent class arguments.
        """
        super(InputLayer, self).__init__(**kwargs)
        self.regions = regions
        self.n_distributions = n_distributions
        self.vectors = []
        self.n_features = None

    def build(self, input_shape):
        """
        Build the layer.

        :param input_shape: The number of distributions per region.
        """
        self.n_features = input_shape[-1]

        # Set the number of features and distributions and create a normal distribution vector for each region
        for region in self.regions:
            distribution = MultivariateNormal(self.n_distributions, len(region))
            distribution.build()
            self.vectors.append(distribution)

        # Call the parent class build method
        super(InputLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer.

        :param input_shape: The input shape.
        :return: The output shape.
        """
        return len(self.regions), self.n_distributions

    def call(self, inputs, **kwargs):
        """
        Evaluate the layer given some inputs.

        :param inputs: The inputs.
        :param kwargs: Other arguments.
        :return: The tensor result of the layer.
        """
        results = []

        # Compute the log likelihoods for each region
        for i, region in enumerate(self.regions):
            # Create the masked input from the region indices
            masked_input = tf.gather(inputs, region, axis=1)

            # Calculate the log likelihoods of the given distribution vector
            results.append(self.vectors[i].log_prob(masked_input))

        return tf.stack(results)


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
        self.n_outputs = None

    def build(self, input_shape):
        """
        Build the layer.

        :param input_shape: The input shape.
        """
        # Call the parent class build method
        super(ProductLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape.

        :param input_shape: The input shape.
        :return: The output shape.
        """
        return input_shape[0] // 2, input_shape[1] ** 2

    def call(self, inputs, **kwargs):
        """
        Evaluate the layer given some inputs.

        :param inputs: The inputs.
        :param kwargs: Other arguments.
        :return: The tensor result of the layer.
        """
        n_regions = inputs.get_shape()[0]

        # Pairs the results using even-odd indexing
        dist0 = tf.gather(inputs, [i for i in range(n_regions) if i % 2 == 0])
        dist1 = tf.gather(inputs, [i for i in range(n_regions) if i % 2 == 1])

        # Compute the outer product (the "outer sum" in the log space)
        result = tf.expand_dims(dist0, 1) + tf.expand_dims(dist1, 2)

        # Flatten the results of each product layer
        return tf.reshape(result, [n_regions // 2, -1])


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
        # Construct the weights as a matrix of shape (N, S, M) where N is the number of input product layers, S is the
        # number of sum nodes for each layer and M is the number of product node of each product layer
        self.kernel = self.add_weight('kernel', shape=(input_shape[0], self.n_sum, input_shape[1]), trainable=True)

        # Call the parent class build method
        super(SumLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape.

        :param input_shape: The input shape.
        :return: The output shape.
        """
        return input_shape[0], self.n_sum

    def call(self, inputs, **kwargs):
        """
        Evaluate the layer  given some inputs.

        :param inputs: The inputs.
        :param kwargs: Other arguments.
        :return: The tensor result of the layer.
        """
        # Calculate the log likelihood using the logsumexp trick (3-D tensor with 2-D tensor multiplication here)
        return tf.math.log(tf.linalg.matvec(self.kernel, tf.math.exp(inputs)))
