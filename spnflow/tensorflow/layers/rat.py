import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class GaussianLayer(tf.keras.layers.Layer):
    """The Gaussian distributions input layer class."""
    def __init__(self, depth, regions, n_batch, optimize_scale, **kwargs):
        """
        Initialize a Gaussian distributions input layer.

        :param depth: The depth of the RAT-SPN.
        :param regions: The regions of the distributions.
        :param n_batch: The number of distributions.
        :param optimize_scale: Whatever to train scale and mean jointly.
        :param kwargs: Other arguments.
        """
        super(GaussianLayer, self).__init__(**kwargs)
        self.depth = depth
        self.regions = regions
        self.n_batch = n_batch
        self.optimize_scale = optimize_scale
        self._pad = None
        self._mean = None
        self._scale = None
        self._distribution = None

    def build(self, input_shape):
        """
        Build the layer.

        :param input_shape: The input shape.
        """
        n_features = input_shape[1]
        n_regions = len(self.regions)

        # Compute the padding
        self._pad = -n_features % (2 ** self.depth)
        dim_gauss = (n_features + self._pad) // (2 ** self.depth)

        # Append dummy variables to regions orderings
        if self._pad > 0:
            for i in range(n_regions):
                n_dummy = dim_gauss - len(self.regions[i])
                self.regions[i] = list(self.regions[i]) + list(self.regions[i])[:n_dummy]
        self.regions = np.asarray(self.regions)

        # Instantiate the mean variable
        self._mean = self.add_weight(
            'mean',
            shape=[n_regions, self.n_batch, dim_gauss],
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-1),
            trainable=True
        )

        # Instantiate the scale variable
        if self.optimize_scale:
            self._scale = self.add_weight(
                'scale',
                shape=[n_regions, self.n_batch, dim_gauss],
                initializer=tf.keras.initializers.TruncatedNormal(mean=0.5, stddev=1e-1),
                trainable=True
            )
        else:
            self._scale = self.add_weight(
                'scale',
                shape=[n_regions, self.n_batch, dim_gauss],
                initializer=tf.keras.initializers.Ones(),
                trainable=False
            )

        # Create the multi-batch multivariate distribution
        self._distribution = tfp.distributions.MultivariateNormalDiag(self._mean, self._scale)

        # Call the parent class's build method
        super(GaussianLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Execute the layer on some inputs.

        :param inputs: The inputs.
        :param kwargs: Other arguments.
        :return: The log likelihood of each distribution leaf.
        """
        # Compute the log-likelihoods
        x = tf.gather(inputs, self.regions, axis=1)
        x = tf.expand_dims(x, axis=2)
        x = self._distribution.log_prob(x)
        return x

    def sample(self, n_samples, idx_partition, idx_offset):
        """
        Sample from the modeled distribution.

        TODO: remove dummy variables padding from generated samples.

        :param n_samples: The number of samples.
        :param idx_partiton: The indices of the partitions.
        :param idx_offset: The indices of the nodes
        :return: The indices of the modeled sub-distributions.
        """
        # Sample from the distributions layer and gather the correct samples
        idx = tf.stack([idx_partition, idx_offset], axis=-1)
        samples = self._distribution.sample(n_samples)
        samples = tf.gather_nd(samples, idx, batch_dims=1)
        samples = tf.reshape(samples, shape=[n_samples, -1])

        # Invert the region permutations
        region_perm = tf.gather(self.regions, idx_partition)
        region_perm = tf.reshape(region_perm, shape=[n_samples, -1])
        region_perm = tf.map_fn(tf.math.invert_permutation, region_perm)

        # Reorder the samples features using the inverted region permutations
        samples = tf.stack([
            tf.gather(samples[i], region_perm[i])
            for i in range(n_samples)
        ])
        return samples


class ProductLayer(tf.keras.layers.Layer):
    """Product node layer class."""
    def __init__(self, **kwargs):
        """
        Initialize the Product layer.

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
        # Set the number of nodes for each region channel
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

    def sample(self, n_samples, idx_partition, idx_offset):
        """
        Sample from the modeled distribution.

        :param n_samples: The number of samples.
        :param idx_partition: The indices of the partitions.
        :param idx_offset: The indices of the nodes
        :return: The indices of the modeled sub-distributions.
        """
        # Compute the sub-distributions region indices and the offset of the nodes
        idx_region0 = idx_partition * 2 + 0
        idx_region1 = idx_partition * 2 + 1
        idx_offset0 = idx_offset // self.n_nodes
        idx_offset1 = idx_offset % self.n_nodes
        idx_region = tf.concat([idx_region0, idx_region1], axis=1)
        idx_offset = tf.concat([idx_offset0, idx_offset1], axis=1)
        return idx_region, idx_offset


class SumLayer(tf.keras.layers.Layer):
    """Sum node layer."""
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
        # Construct the weights
        n_nodes = input_shape[-1]
        n_regions = input_shape[-2]
        self.kernel = self.add_weight(
            'kernel',
            shape=[n_regions, self.n_sum, n_nodes],
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=5e-1),
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
        w = tf.math.log_softmax(self.kernel, axis=2)  # (p, s)
        x = tf.math.reduce_logsumexp(x + w, axis=-1)  # (n, p, s)
        return x

    def sample(self, n_samples, idx_region, idx_offset):
        """
        Sample from the modeled distribution.

        :param n_samples: The number of samples.
        :param idx_region: The indices of the regions.
        :param idx_offset: The indices of the nodes
        :return: The indices of the modeled sub-distributions.
        """
        # Gather the weights and transform them into logits space
        idx = tf.stack([idx_region, idx_offset], axis=-1)
        logits = tf.gather_nd(self.kernel, idx)
        logits = tf.math.log_softmax(logits, axis=-1)

        # Sample the new offset indices
        idx_offset = tf.map_fn(lambda x: tf.random.categorical(x, 1), logits, dtype='int64')
        idx_offset = tf.squeeze(idx_offset)
        return idx_region, idx_offset


class RootLayer(tf.keras.layers.Layer):
    """Root sum node layer."""
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
        n_nodes = input_shape[-1]
        self.kernel = self.add_weight(
            'kernel',
            shape=[1, n_nodes],
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=5e-1),
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
        w = tf.math.log_softmax(self.kernel, axis=1)  # (1, k)
        x = tf.math.reduce_logsumexp(x + w, axis=-1)  # (n, 1)
        return x

    def sample(self, n_samples):
        """
        Sample from the modeled distribution.

        :param n_samples: The number of samples.
        :return: The indices of the modeled sub-distributions.
        """
        # Sample the indices of the nodes of the flattened product layer
        logits = tf.math.log_softmax(self.kernel, axis=1)
        idx_offset = tf.random.categorical(logits, n_samples)
        return idx_offset


class DropoutLayer(tf.keras.layers.Layer):
    """Dropout layer."""
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
