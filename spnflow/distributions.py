import tensorflow as tf
import tensorflow_probability as tfp


class NormalVector(tf.keras.layers.Layer):
    """
    Multivariate Normal distributions Keras layer.
    """
    def __init__(self, n_batch, n_dim):
        super(NormalVector, self).__init__()

        self.mean = tf.Variable(
            tf.random.normal(shape=(n_batch, n_dim), stddev=0.1),
            trainable=True
        )

        self.scale = tfp.util.TransformedVariable(
            tf.eye(n_dim, batch_shape=(n_batch,)),
            tfp.bijectors.FillScaleTriL,
            trainable=True
        )

        self.distribution = tfp.distributions.MultivariateNormalTriL(self.mean, tf.linalg.cholesky(self.scale))

    def call(self, inputs, **kwargs):
        return self.distribution.log_prob(inputs)

    def sample(self, sample_shape):
        return self.distribution.sample(sample_shape)
