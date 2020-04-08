import tensorflow as tf
import tensorflow_probability as tfp


class NormalLayer(tf.keras.layers.Layer):
    """
    Multi-batch Multivariate Normal distribution layer class.
    """
    def __init__(self, region, n_batch, sigma_min=1.0-1e-1, sigma_max=1.0+1e-1, **kwargs):
        """
        Initialize the multi-batch and multivariate normal distribution.

        :param region: The region of the layer.
        :param n_batch: The number of batches.
        :param sigma_min: Variance initialization minimum value.
        :param sigma_max: Variance initialization maximum value.
        :param kwargs: Other arguments.
        """
        super(NormalLayer, self).__init__(**kwargs)
        self.region = region
        self.n_batch = n_batch
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._mean = None
        self._scale = None
        self._distribution = None

    def build(self, input_shape):
        """
        Build the multi-batch and multivariate distribution.

        :param input_shape: The input shape.
        """
        # Create the mean variable
        self._mean = tf.Variable(
            tf.random.normal(shape=(self.n_batch, len(self.region)), stddev=1e-1),
            trainable=True
        )

        # Create the log variance diagonal variable
        sigmoid_params = tf.random.normal(shape=(self.n_batch, len(self.region)), stddev=1e-1)
        self._scale = tf.Variable(
            self.sigma_min + (self.sigma_max - self.sigma_min) * tf.math.sigmoid(sigmoid_params),
            trainable=True
        )

        # Create the multi-batch multivariate distribution
        self._distribution = tfp.distributions.MultivariateNormalDiag(self._mean, self._scale)

        # Call the parent class build method
        super(NormalLayer, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        """
        Compute the log likelihoods given some inputs.

        :param inputs: The inputs.
        :return: The log likelihoods.
        """
        # Calculate the log likelihoods given some inputs
        masked_input = tf.gather(inputs, self.region, axis=1)
        masked_input = tf.expand_dims(masked_input, 1)
        return self._distribution.log_prob(masked_input)

    @tf.function
    def sample(self, sample_shape):
        """
        Sample some values from the distribution.

        :param sample_shape: The sample shape.
        :return: Some samples.
        """
        # Sample some values from the distributions
        return self._distribution.sample(sample_shape)
