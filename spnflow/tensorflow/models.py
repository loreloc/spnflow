import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from spnflow.utils.region import RegionGraph
from spnflow.tensorflow.layers import GaussianLayer, ProductLayer, SumLayer, RootLayer, DropoutLayer


class RatSpnFlow(tf.keras.Model):
    """RAT-SPN base distribution improved with Normalizing Flows."""
    def __init__(self,
                 depth=2,
                 n_batch=2,
                 n_sum=2,
                 n_repetitions=1,
                 dropout=0.0,
                 optimize_scale=True,
                 flow='maf',
                 n_flows=5,
                 hidden_units=[128],
                 activation='relu',
                 regularization=1e-6,
                 batch_norm=True,
                 rand_state=None,
                 **kwargs
                 ):
        """
        Initialize an Autoregressive RAT-SPN.

        :param depth: The depth of the network.
        :param n_batch: The number of distributions.
        :param n_sum: The number of sum nodes.
        :param n_repetitions: The number of independent repetitions of the region graph.
        :param dropout: The rate of the dropout layers.
        :param optimize_scale: Whatever to train scale and mean jointly.
        :param flow: The kind of normalizing flow to use. Must be one of {'nvp', 'maf'}.
        :param n_flows: The number of chained Normalizing Flows.
        :param hidden_units: A list of the number of units for each layer of the conditioner networks.
        :param activation: The activation function for the conditioner networks.
        :param regularization: The L2 regularization weight for the conditioner networks.
        :param rand_state: The random state to use to generate the RAT-SPN model.
        :param batch_norm: Whatever to use batch normalization after each MAF layer.
        :param kwargs: Other arguments.
        """
        super(RatSpnFlow, self).__init__(**kwargs)
        self.depth = depth
        self.n_batch = n_batch
        self.n_sum = n_sum
        self.n_repetitions = n_repetitions
        self.dropout = dropout
        self.optimize_scale = optimize_scale
        self.flow = flow
        self.n_flows = n_flows
        self.hidden_units = hidden_units
        self.activation = activation
        self.regularization = regularization
        self.batch_norm = batch_norm
        self.rand_state = rand_state
        self._spn = None
        self._bns = None
        self._conds = None
        self._flows = None
        self._bijector = None

    def build(self, input_shape):
        """
        Build the model.

        :param input_shape: The input shape.
        """
        # Build the RAT-SPN that represents the base distribution
        self._spn = RatSpn(
            self.depth,
            self.n_batch,
            self.n_sum,
            self.n_repetitions,
            self.dropout,
            self.optimize_scale,
            self.rand_state
        )
        self._spn.build(input_shape)

        # Build the normalizing flows layers
        self.flow = self.flow.lower()
        if self.flow == 'nvp':
            self._build_nvp(input_shape)
        elif self.flow == 'maf':
            self._build_maf(input_shape)
        else:
            raise NotImplementedError('Normalizing flow ' + self.flow + ' not implemented')

        # Call the parent class build method
        super(RatSpnFlow, self).build(input_shape)

    def _build_nvp(self, input_shape):
        """
        Build RealNVP normalizing flows.
        """
        # Build the  normalizing flows
        self._conds = []
        self._flows = []
        for _ in range(self.n_flows):
            # Build the conditioner
            cond = tfp.bijectors.real_nvp_default_template(
                hidden_layers=self.hidden_units,
                activation=self.activation,
                kernel_regularizer=tf.keras.regularizers.l2(self.regularization)
            )
            self._conds.append(cond)

            # Build the RealNVP layer
            nvp = tfp.bijectors.RealNVP(
                shift_and_log_scale_fn=cond, fraction_masked=0.5
            )
            self._flows.append(nvp)

        _, n_features = input_shape
        perm_flag = True
        even_perm = list(range(0, n_features, 2))
        odd_perm = list(range(1, n_features, 2))

        # Build the bijector by chaining multiple flows
        self._bns = []
        bijectors = []
        for flow in self._flows:
            # Append a checkboard-like input permutation bijector
            if perm_flag:
                bijectors.append(tfp.bijectors.Permute(even_perm + odd_perm))
            else:
                bijectors.append(tfp.bijectors.Permute(odd_perm + even_perm))
            perm_flag = not perm_flag

            # Append the flow bijection
            bijectors.append(flow)

            # Append batch normalization bijection, if specified
            if self.batch_norm:
                bn = tfp.bijectors.BatchNormalization()
                self._bns.append(bn)
                bijectors.append(bn)
        self._bijector = tfp.bijectors.Chain(bijectors)

    def _build_maf(self, input_shape):
        """
        Build MAFs normalizing flows.
        """
        # Build the MAFs normalizing flows
        self._conds = []
        self._flows = []
        input_order = 'left-to-right'
        for _ in range(self.n_flows):
            # Build the MADE conditioner
            made = tfp.bijectors.AutoregressiveNetwork(
                params=2,
                input_order=input_order,
                hidden_units=self.hidden_units,
                activation=self.activation,
                kernel_regularizer=tf.keras.regularizers.l2(self.regularization)
            )
            self._conds.append(made)

            # Build the MAF layer
            maf = tfp.bijectors.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=made, unroll_loop=True
            )
            self._flows.append(maf)

            # Change the input order
            input_order = 'right-to-left' if input_order == 'left-to-right' else 'left-to-right'

        # Build the bijector by chaining multiple flows
        self._bns = []
        bijectors = []
        for flow in self._flows:
            # Append the flow bijection
            bijectors.append(flow)

            # Append batch normalization bijection, if specified
            if self.batch_norm:
                bn = tfp.bijectors.BatchNormalization()
                self._bns.append(bn)
                bijectors.append(bn)
        self._bijector = tfp.bijectors.Chain(bijectors)

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        """
        Call the model.

        :param inputs: The inputs tensor.
        :param training: Whatever the model is training or not.
        :param kwargs: Other arguments.
        :return: The output of the model.
        """
        u = self._bijector.inverse(inputs, training=training)
        p = self._spn(u, training=training, **kwargs)
        d = self._bijector.inverse_log_det_jacobian(inputs, event_ndims=1, training=training)
        p = p + tf.expand_dims(d, axis=-1)
        return p

    @tf.function
    def sample(self, n_samples=1):
        """
        Sample from the modeled distribution.

        :param n_samples: The number of samples.
        :return: The samples.
        """
        samples = self._spn.sample(n_samples)
        samples = self._bijector.forward(samples)
        return samples


class RatSpn(tf.keras.Model):
    """RAT-SPN Keras model class."""
    def __init__(self,
                 depth=2,
                 n_batch=2,
                 n_sum=2,
                 n_repetitions=1,
                 dropout=0.0,
                 optimize_scale=True,
                 rand_state=None,
                 **kwargs
                 ):
        """
        Initialize a RAT-SPN.

        :param depth: The depth of the network.
        :param n_batch: The number of distributions.
        :param n_sum: The number of sum nodes.
        :param n_repetitions: The number of independent repetitions of the region graph.
        :param dropout: The rate of the dropout layers.
        :param optimize_scale: Whatever to train scale and mean jointly.
        :param rand_state: The random state used to generate the random graph.
        :param kwargs: Other arguments.
        """
        super(RatSpn, self).__init__(**kwargs)
        self.depth = depth
        self.n_batch = n_batch
        self.n_sum = n_sum
        self.n_repetitions = n_repetitions
        self.dropout = dropout
        self.optimize_scale = optimize_scale
        self.rand_state = rand_state
        self.n_features = None
        self.rg_layers = None
        self._base_layer = None
        self._inner_layers = None
        self._flatten_layer = None
        self._root_layer = None

        # If necessary, instantiate a random state
        if self.rand_state is None:
            self.rand_state = np.random.RandomState(42)

    def build(self, input_shape):
        """
        Build the model.

        :param input_shape: The input shape.
        """
        _, self.n_features = input_shape

        # Instantiate the region graph
        region_graph = RegionGraph(self.n_features, self.depth, self.rand_state)

        # Generate the layers
        self.rg_layers = region_graph.make_layers(self.n_repetitions)
        self.rg_layers = list(reversed(self.rg_layers))

        # Add the base distributions layer
        self._base_layer = GaussianLayer(self.depth, self.rg_layers[0], self.n_batch, self.optimize_scale)

        # Alternate between product and sum layer
        self._inner_layers = []
        for i in range(1, len(self.rg_layers) - 1):
            if i % 2 == 1:
                self._inner_layers.append(ProductLayer())
            else:
                if self.dropout > 0.0:
                    self._inner_layers.append(DropoutLayer(self.dropout))
                self._inner_layers.append(SumLayer(self.n_sum))

        # Add the flatten layer
        self._flatten_layer = tf.keras.layers.Flatten()

        # Add the sum root layer
        self._root_layer = RootLayer()

        # Call the parent class build method
        super(RatSpn, self).build(input_shape)

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        """
        Call the model.

        :param inputs: The inputs tensor.
        :param training: Whatever the model is training or not.
        :param kwargs: Other arguments.
        :return: The output of the model.
        """
        # Calculate the base log-likelihoods
        x = self._base_layer(inputs, training=training, **kwargs)

        # Forward through the inner layers
        for layer in self._inner_layers:
            x = layer(x, training=training, **kwargs)

        # Flatten the result and forward through the sum root layer
        x = self._flatten_layer(x)
        x = self._root_layer(x)
        return x

    @tf.function
    def sample(self, n_samples=1):
        """
        Sample from the modeled distribution.

        :param n_samples: The number of samples.
        :return: The samples.
        """
        idx = self._root_layer.sample(n_samples)
        idx = tf.transpose(idx)
        idx_partition = idx // (self.n_sum ** 2)
        idx_offset = idx % (self.n_sum ** 2)

        for layer in reversed(self._inner_layers):
            if isinstance(layer, DropoutLayer):
                continue
            idx_partition, idx_offset = layer.sample(n_samples, idx_partition, idx_offset)

        samples = self._base_layer.sample(n_samples, idx_partition, idx_offset)
        return samples
