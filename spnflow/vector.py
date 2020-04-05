import numpy as np
import tensorflow as tf
from spnflow.distributions import NormalVector


class InputVector(tf.keras.layers.Layer):
    def __init__(self, regions, n_distributions):
        super(InputVector, self).__init__()
        self.regions = regions
        self.n_distributions = n_distributions
        self.vectors = []
        self.n_features = None

    def build(self, input_shape):
        # Set the number of features and create a normal distribution vector for each region
        self.n_features = input_shape
        for region in self.regions:
            self.vectors.append(NormalVector(self.n_distributions, len(region)))

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


class MulVector(tf.keras.layers.Layer):
    """
    Multiplication node layer.
    """
    def __init__(self):
        super(MulVector, self).__init__()

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


class SumVector(tf.keras.layers.Layer):
    """
    Sum node layer.
    """
    def __init__(self, n_sum):
        super(SumVector, self).__init__()
        self.n_outputs = n_sum
        self.kernel = None

    def build(self, input_shape):
        # Construct the weights as a matrix of shape (N, S, M) where N is the number of input product layers, S is the
        # number of sum nodes for each layer and M is the number of product node of each product layer
        self.kernel = self.add_weight('kernel', shape=(input_shape[0], self.n_outputs, input_shape[1]), trainable=True)

    def call(self, inputs, **kwargs):
        # Calculate the log likelihood using the logsumexp trick (3-D tensor with 2-D tensor multiplication here)
        return tf.math.log(tf.linalg.matvec(self.kernel, tf.math.exp(inputs)))
