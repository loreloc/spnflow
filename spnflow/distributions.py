import tensorflow as tf
import tensorflow_probability as tfp


class MultivariateNormal:
    """
    Multi-batch Multivariate Normal distribution.
    """
    def __init__(self, n_batch, n_features):
        """
        Initialize the multi-batch and multivariate normal distribution.

        :param n_batch: The number of batches.
        :param n_features: The number of random variables.
        """
        super(MultivariateNormal, self).__init__()
        self.n_batch = n_batch
        self.n_features = n_features
        self.mean = None
        self.scale = None
        self.distribution = None

    def build(self):
        """
        Build the multi-batch and multivariate distribution.
        """
        # Create the mean multi-batch multivariate variable
        self.mean = tf.Variable(
            tf.random.normal(shape=(self.n_batch, self.n_features), stddev=0.1),
            trainable=True
        )

        # Create the scale matrix multi-batch multivariate variable
        self.scale = tfp.util.TransformedVariable(
            tf.eye(self.n_features, batch_shape=(self.n_batch,)),
            tfp.bijectors.FillScaleTriL(),
            trainable=True
        )

        # Create the multi-batch multivariate variable
        self.distribution = tfp.distributions.MultivariateNormalTriL(self.mean, tf.linalg.cholesky(self.scale))

    def log_prob(self, inputs, **kwargs):
        """
        Compute the log likelihoods given some inputs.

        :param inputs: The inputs.
        :param kwargs: Other arguments.
        :return: The log likelihoods.
        """
        # Calculate the log likelihoods given some inputs
        return self.distribution.log_prob(inputs)

    def sample(self, sample_shape):
        """
        Sample some values from the distribution.

        :param sample_shape: The sample shape.
        :return: Some samples.
        """
        # Sample some values from the distributions
        return self.distribution.sample(sample_shape)
