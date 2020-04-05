import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class NormalLayer(tf.keras.layers.Layer):
    """
    Multi-batch Multivariate Normal distributions Keras layer class.
    """
    def __init__(self):
        super(NormalLayer, self).__init__()
        self.n_batch = None
        self.n_features = None
        self.mean = None
        self.scale = None
        self.distribution = None

    def build(self, input_shape):
        self.n_batch, self.n_features = input_shape

        # Create the mean multi-batch multivariate variable
        self.mean = tf.Variable(
            tf.random.normal(shape=(self.n_batch, self.n_features), stddev=0.1),
            trainable=True
        )

        # Create the scale matrix multi-batch multivariate variable
        self.scale = tfp.util.TransformedVariable(
            tf.eye(self.n_features, batch_shape=(self.n_batch,)),
            tfp.bijectors.FillScaleTriL,
            trainable=True
        )

        # Create the multi-batch multivariate variable
        self.distribution = tfp.distributions.MultivariateNormalTriL(self.mean, tf.linalg.cholesky(self.scale))

        # Call the parent class build method
        super(NormalLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, **kwargs):
        return self.distribution.log_prob(inputs)

    def sample(self, sample_shape):
        return self.distribution.sample(sample_shape)


class InputLayer(tf.keras.layers.Layer):
    """
    Input distributions layer class.
    """
    def __init__(self, regions):
        super(InputLayer, self).__init__()
        self.regions = regions
        self.vectors = []
        self.n_features = None
        self.n_distributions = None

    def build(self, input_shape):
        # Set the number of features and distributions and create a normal distribution vector for each region
        self.n_features, n_distributions = input_shape
        for region in self.regions:
            distribution = NormalLayer()
            distribution.build((self.n_distributions, len(region)))
            self.vectors.append(distribution)

        # Call the parent class build method
        super(InputLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return len(self.regions) // 2, 2, input_shape[1]

    def call(self, inputs, **kwargs):
        outputs = []

        # Compute the log likelihoods for each region
        for i, region in enumerate(self.regions):
            # Create the masked input from the region indices
            mask = np.zeros(self.n_features, dtype=np.bool)
            mask[np.asarray(region)] = True
            masked_input = inputs[mask]

            # Calculate the log likelihoods of the given distribution vector
            outputs.append(self.vectors[i](masked_input))

        # Merge all the log likelihoods value
        merged_outputs = tf.concat(outputs, axis=0)

        # Reshape the log likelihoods tensor pairing the log likelihoods of the distribution vectors
        return tf.reshape(merged_outputs, -1, 2, self.n_distributions)


class ProductLayer(tf.keras.layers.Layer):
    """
    Product node layer.
    """
    def __init__(self):
        super(ProductLayer, self).__init__()
        self.n_outputs = None

    def build(self, input_shape):
        # Call the parent class build method
        super(ProductLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2] ** 2

    def call(self, inputs, **kwargs):
        outputs = []

        # For each log likelihoods pairing
        for i in range(tf.shape(inputs)[0]):
            # Compute the outer product ("outer sum" in the log space)
            dist0 = tf.expand_dims(inputs[i][0], 0)
            dist1 = tf.expand_dims(inputs[i][1], 1)
            log_prod = dist0 + dist1

            # Flatten the results
            outputs.append(np.reshape(log_prod, [-1]))

        # Stack the results of the multiplication nodes
        return tf.stack(outputs, axis=0)


class SumLayer(tf.keras.layers.Layer):
    """
    Sum node layer.
    """
    def __init__(self, n_sum):
        super(SumLayer, self).__init__()
        self.n_sum = n_sum
        self.kernel = None

    def build(self, input_shape):
        # Construct the weights as a matrix of shape (N, S, M) where N is the number of input product layers, S is the
        # number of sum nodes for each layer and M is the number of product node of each product layer
        self.kernel = self.add_weight('kernel', shape=(input_shape[0], self.n_sum, input_shape[1]), trainable=True)

        # Call the parent class build method
        super(SumLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0] // 2, 2, self.n_sum

    def call(self, inputs, **kwargs):
        # Calculate the log likelihood using the logsumexp trick (3-D tensor with 2-D tensor multiplication here)
        merged_results = tf.math.log(tf.linalg.matvec(self.kernel, tf.math.exp(inputs)))
        return tf.reshape(merged_results, -1, 2, self.n_sum)
