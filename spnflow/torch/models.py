import abc
import torch
import numpy as np
from spnflow.utils.region import RegionGraph
from spnflow.torch.layers.ratspn import GaussianLayer, ProductLayer, SumLayer, RootLayer
from spnflow.torch.layers.flows import CouplingLayer, AutoregressiveLayer, BatchNormLayer
from spnflow.torch.layers.dgcspn import SpatialGaussianLayer, SpatialProductLayer, SpatialSumLayer, SpatialRootLayer
from spnflow.torch.constraints import ScaleClipper
from spnflow.torch.initializers import quantiles_initializer


class AbstractModel(abc.ABC, torch.nn.Module):
    """Abstract class for deep probabilistic models."""
    def __init__(self):
        super(AbstractModel, self).__init__()

    def log_prob(self, x):
        return self(x)

    @abc.abstractmethod
    def forward(self, x):
        pass

    @torch.no_grad()
    @abc.abstractmethod
    def mpe(self, x):
        pass

    @torch.no_grad()
    @abc.abstractmethod
    def sample(self, n_samples):
        pass

    def apply_initializers(self, **kwargs):
        pass

    def apply_constraints(self):
        pass


class RealNVP(AbstractModel):
    """Real Non-Volume-Preserving (RealNVP) normalizing flow model."""
    def __init__(self,
                 in_features,
                 n_flows=5,
                 batch_norm=True,
                 depth=1,
                 units=128,
                 activation=torch.nn.ReLU,
                 in_base=None,
                 ):
        """
        Initialize a RealNVP.

        :param in_features: The number of input features.
        :param n_flows: The number of sequential coupling flows.
        :param batch_norm: Whether to apply batch normalization after each coupling layer.
        :param depth: The number of hidden layers of flows conditioners.
        :param units: The number of hidden units per layer of flows conditioners.
        :param activation: The activation class to use for the flows conditioners hidden layers.
        :param in_base: The input base distribution to use. If None, the standard Normal distribution is used.

        """
        super(RealNVP, self).__init__()
        self.in_features = in_features
        self.n_flows = n_flows
        self.batch_norm = batch_norm
        self.depth = depth
        self.units = units
        self.activation = activation

        # Build the base distribution, if necessary
        if in_base:
            self.in_base = in_base
        else:
            self.in_base_loc = torch.nn.Parameter(torch.zeros([self.in_features], requires_grad=False))
            self.in_base_scale = torch.nn.Parameter(torch.ones([self.in_features], requires_grad=False))
            self.in_base = torch.distributions.Normal(self.in_base_loc, self.in_base_scale)

        # Build the coupling layers
        self.layers = torch.nn.ModuleList()
        reverse = False
        for _ in range(self.n_flows):
            self.layers.append(
                CouplingLayer(self.in_features, self.depth, self.units, self.activation, reverse=reverse)
            )

            # Append batch normalization after each layer, if specified
            if self.batch_norm:
                self.layers.append(BatchNormLayer(self.in_features))

            # Invert the input ordering
            reverse = not reverse

    def forward(self, x):
        """
        Compute the log-likelihood given complete evidence.

        :param x: The inputs tensor.
        :return: The output of the model.
        """
        inv_log_det_jacobian = 0.0
        for layer in self.layers:
            x, ildj = layer.inverse(x)
            inv_log_det_jacobian += ildj
        prior = self.in_base.log_prob(x)
        return torch.sum(prior, dim=1) + inv_log_det_jacobian

    @torch.no_grad()
    def mpe(self, x):
        raise NotImplementedError('Maximum at posteriori estimation is not implemented for RealNVPs')

    @torch.no_grad()
    def sample(self, n_samples):
        """
        Sample some values from the modeled distribution.

        :param n_samples: The number of samples.
        :return: The samples.
        """
        # Sample from the base distribution
        if isinstance(self.in_base, torch.distributions.Distribution):
            x = self.in_base.sample([n_samples])
        else:
            x = self.in_base.sample(n_samples)

        # Apply the normalizing flows transformations
        for layer in reversed(self.layers):
            x, ldj = layer.forward(x)
        return x


class MAF(AbstractModel):
    """Masked Autoregressive Flow (MAF) normalizing flow model."""
    def __init__(self,
                 in_features,
                 n_flows=5,
                 batch_norm=True,
                 depth=1,
                 units=128,
                 activation=torch.nn.ReLU,
                 in_base=None,
                 ):
        """
        Initialize a MAF.

        :param in_features: The number of input features.
        :param n_flows: The number of sequential autoregressive layers.
        :param batch_norm: Whether to apply batch normalization after each autoregressive layer.
        :param depth: The number of hidden layers of flows conditioners.
        :param units: The number of hidden units per layer of flows conditioners.
        :param activation: The activation class to use for the flows conditioners hidden layers.
        :param in_base: The input base distribution to use. If None, the standard Normal distribution is used.
        """
        super(MAF, self).__init__()
        self.in_features = in_features
        self.n_flows = n_flows
        self.batch_norm = batch_norm
        self.depth = depth
        self.units = units
        self.activation = activation

        # Build the base distribution, if necessary
        if in_base:
            self.in_base = in_base
        else:
            self.in_base_loc = torch.nn.Parameter(torch.zeros([self.in_features], requires_grad=False))
            self.in_base_scale = torch.nn.Parameter(torch.ones([self.in_features], requires_grad=False))
            self.in_base = torch.distributions.Normal(self.in_base_loc, self.in_base_scale)

        # Build the autoregressive layers
        self.layers = torch.nn.ModuleList()
        reverse = False
        for _ in range(self.n_flows):
            self.layers.append(
                AutoregressiveLayer(self.in_features, self.depth, self.units, self.activation, reverse=reverse)
            )

            # Append batch normalization after each layer, if specified
            if self.batch_norm:
                self.layers.append(BatchNormLayer(self.in_features))

            # Invert the input ordering
            reverse = not reverse

    def forward(self, x):
        """
        Compute the log-likelihood given complete evidence.

        :param x: The inputs tensor.
        :return: The output of the model.
        """
        inv_log_det_jacobian = 0.0
        for layer in self.layers:
            x, ildj = layer.inverse(x)
            inv_log_det_jacobian += ildj
        prior = self.in_base.log_prob(x)
        return torch.sum(prior, dim=1) + inv_log_det_jacobian

    @torch.no_grad()
    def mpe(self, x):
        raise NotImplementedError('Maximum at posteriori estimation is not implemented for RealNVPs')

    @torch.no_grad()
    def sample(self, n_samples):
        """
        Sample some values from the modeled distribution.

        :param n_samples: The number of samples.
        :return: The samples.
        """
        # Sample from the base distribution
        if isinstance(self.in_base, torch.distributions.Distribution):
            x = self.in_base.sample([n_samples])
        else:
            x = self.in_base.sample(n_samples)

        # Apply the normalizing flows transformations
        for layer in reversed(self.layers):
            x, ldj = layer.forward(x)
        return x


class RatSpn(AbstractModel):
    """RAT-SPN model class."""
    def __init__(self,
                 in_features,
                 out_classes=1,
                 rg_depth=2,
                 rg_repetitions=1,
                 n_batch=2,
                 n_sum=2,
                 dropout=None,
                 optimize_scale=True,
                 rand_state=None,
                 ):
        """
        Initialize a RAT-SPN.

        :param in_features: The number of input features.
        :param out_classes: The number of output classes. Specify 1 in case of plain density estimation.
        :param rg_depth: The depth of the region graph.
        :param rg_repetitions: The number of independent repetitions of the region graph.
        :param n_batch: The number of base distributions batches.
        :param n_sum: The number of sum nodes per region.
        :param dropout: The dropout rate for probabilistic dropout at sum layer inputs (can be None).
        :param optimize_scale: Whether to train scale and location jointly.
        :param rand_state: The random state used to generate the random graph.
        """
        super(RatSpn, self).__init__()
        self.in_features = in_features
        self.out_classes = out_classes
        self.rg_depth = rg_depth
        self.rg_repetitions = rg_repetitions
        self.n_batch = n_batch
        self.n_sum = n_sum
        self.dropout = dropout
        self.optimize_scale = optimize_scale
        self.rand_state = rand_state

        # If necessary, instantiate a random state
        if self.rand_state is None:
            self.rand_state = np.random.RandomState(42)

        # Instantiate the region graph
        region_graph = RegionGraph(self.in_features, self.rg_depth, self.rand_state)

        # Generate the layers
        self.rg_layers = region_graph.make_layers(self.rg_repetitions)
        self.rg_layers = list(reversed(self.rg_layers))

        # Instantiate the base distributions layer
        self.base_layer = GaussianLayer(
            self.in_features, self.n_batch, self.rg_layers[0], self.rg_depth, self.optimize_scale
        )

        # Alternate between product and sum layer
        in_groups = self.base_layer.in_regions
        in_nodes = self.base_layer.out_channels
        self.layers = torch.nn.ModuleList()
        for i in range(1, len(self.rg_layers) - 1):
            if i % 2 == 1:
                layer = ProductLayer(in_groups, in_nodes)
                in_groups = layer.out_partitions
                in_nodes = layer.out_nodes
            else:
                layer = SumLayer(in_groups, in_nodes, self.n_sum, self.dropout)
                in_groups = layer.out_regions
                in_nodes = layer.out_nodes
            self.layers.append(layer)

        # Instantiate the root layer
        self.root_layer = RootLayer(in_groups, in_nodes, self.out_classes)

        # Initialize the scale clipper to apply, if specified
        self.scale_clipper = ScaleClipper() if self.optimize_scale else None

    def forward(self, x):
        """
        Compute the log-likelihood given some evidence.
        Random variables can be marginalized using NaN values.

        :param x: The inputs tensor.
        :return: The output of the model.
        """
        # Compute the base distributions log-likelihoods
        x = self.base_layer(x)

        # Forward through the inner layers
        for layer in self.layers:
            x = layer(x)

        # Forward through the root layer
        return self.root_layer(x)

    @torch.no_grad()
    def mpe(self, x, y=None):
        """
        Compute the maximum at posteriori estimation.
        Random variables can be marginalized using NaN values.

        :param x: The inputs tensor.
        :param y: The target classes tensor. It can be None for unlabeled maximum at posteriori estimation.
        :return: The output of the model.
        """
        lls = []
        inputs = x

        # Compute the base distributions log-likelihoods
        x = self.base_layer(x)

        # Compute in forward mode and gather the inner log-likelihoods
        for layer in self.layers:
            lls.append(x)
            x = layer(x)

        # Compute in forward mode through the root layer and get the class index
        y = y if y else torch.argmax(self.root_layer(x), axis=1)

        # Get the first partitions and offset indices pair
        idx_partition, idx_offset = self.root_layer.mpe(x, y)

        # Compute in top-down mode through the inner layers
        rev_lls = list(reversed(lls))
        rev_layers = list(reversed(self.layers))
        for layer, ll in zip(rev_layers, rev_lls):
            idx_partition, idx_offset = layer.mpe(ll, idx_partition, idx_offset)

        # Compute the maximum at posteriori inference at the base layer
        return self.base_layer.mpe(inputs, idx_partition, idx_offset)

    @torch.no_grad()
    def sample(self, n_samples, y=None):
        """
        Sample some values from the modeled distribution.

        :param n_samples: The number of samples.
        :param y: The target classes. It can be None for unlabeled and uniform sampling.
        :return: The samples.
        """
        # Sample the target classes uniformly, if not specified
        y = y if y else torch.randint(self.out_classes, [n_samples])

        # Get the first partitions and offset indices pair
        idx_partition, idx_offset = self.root_layer.sample(n_samples, y)

        # Compute in top-down mode through the inner layers
        for layer in reversed(self.layers):
            idx_partition, idx_offset = layer.sample(idx_partition, idx_offset)

        # Sample at the base layer
        return self.base_layer.sample(idx_partition, idx_offset)

    def apply_constraints(self):
        """
        Apply the constraints specified by the model.
        """
        # Apply the scale clipper to the base layer, if specified
        if self.optimize_scale:
            self.scale_clipper(self.base_layer)


class RatSpnFlow(AbstractModel):
    """RAT-SPN base distribution improved with Normalizing Flows."""
    def __init__(self,
                 in_features,
                 rg_depth=2,
                 rg_repetitions=1,
                 n_batch=2,
                 n_sum=2,
                 dropout=None,
                 optimize_scale=True,
                 rand_state=None,
                 flow='nvp',
                 n_flows=5,
                 batch_norm=True,
                 depth=1,
                 units=128,
                 activation=torch.nn.ReLU
                 ):
        """
        Initialize a RAT-SPN base distribution improved with Normalizing Flows.

        :param in_features: The number of input features.
        :param rg_depth: The depth of the region graph.
        :param rg_repetitions: The number of independent repetitions of the region graph.
        :param n_batch: The number of base distributions batches.
        :param n_sum: The number of sum nodes per region.
        :param dropout: The dropout rate for probabilistic dropout at sum layer inputs (can be None).
        :param optimize_scale: Whether to train scale and location jointly.
        :param rand_state: The random state used to generate the random graph.
        :param flow: The normalizing flow kind. At the moment, only 'nvp' and 'maf' are supported.
        :param n_flows: The number of sequential normalizing flows.
        :param batch_norm: Whether to apply batch normalization after each normalizing flow layer.
        :param depth: The number of hidden layers of flows conditioners.
        :param units: The number of hidden units per layer of flows conditioners.
        :param activation: The activation class to use for the flows conditioners hidden layers.
        """
        super(RatSpnFlow, self).__init__()
        self.in_features = in_features
        self.rg_depth = rg_depth
        self.rg_repetitions = rg_repetitions
        self.n_batch = n_batch
        self.n_sum = n_sum
        self.dropout = dropout
        self.optimize_scale = optimize_scale
        self.rand_state = rand_state
        self.flow = flow.lower()
        self.n_flows = n_flows
        self.batch_norm = batch_norm
        self.depth = depth
        self.units = units
        self.activation = activation

        # Build the RAT-SPN that models the base distribution
        self.ratspn = RatSpn(
            self.in_features, 1, self.rg_depth, self.rg_repetitions, self.n_batch, self.n_sum, self.dropout,
            self.optimize_scale, self.rand_state
        )

        # Build the normalizing flow layers
        if self.flow == 'nvp':
            flow_class = RealNVP
        elif self.flow == 'maf':
            flow_class = MAF
        else:
            raise NotImplementedError('Unknown normalizing flow named \'' + self.flow + '\'')
        self.flows = flow_class(
            self.in_features, self.n_flows, self.batch_norm, self.depth, self.units, self.activation, self.ratspn
        )

        # Initialize the scale clipper to apply, if specified
        self.scale_clipper = ScaleClipper() if self.optimize_scale else None

    def forward(self, x):
        """
        Compute the log-likelihood given complete evidence.

        :param x: The inputs tensor.
        :return: The output of the model.
        """
        return self.flows(x)

    @torch.no_grad()
    def mpe(self, x):
        raise NotImplementedError('Maximum at posteriori estimation is not implemented for RatSpnFlows')

    @torch.no_grad()
    def sample(self, n_samples):
        """
        Sample some values from the modeled distribution.

        :param n_samples: The number of samples.
        :return: The samples.
        """
        return self.flows.sample(n_samples)

    def apply_constraints(self):
        """
        Apply the constraints specified by the model.
        """
        # Apply the scale clipper to the base layer, if specified
        if self.optimize_scale:
            self.scale_clipper(self.ratspn.base_layer)


class DgcSpn(AbstractModel):
    """Deep Generalized Convolutional SPN model class."""
    def __init__(self,
                 in_size,
                 n_batch=8,
                 prod_channels=16,
                 sum_channels=8,
                 n_pooling=0,
                 quantiles_loc=True,
                 optimize_scale=True,
                 rand_state=None,
                 ):
        """
        Initialize a SpatialSpn.

        :param in_size: The input size.
        :param n_batch: The number of output channels of the base layer.
        :param prod_channels: The number of output channels of spatial product layers.
        :param sum_channels: The number of output channels of spatial sum layers.
        :param n_pooling: The number of initial pooling product layers.
        :param quantiles_loc: Whether to initialize the base distribution location parameters using data quantiles.
        :param optimize_scale: Whether to train scale and location jointly.
        :param rand_state: The random state used to initialize the spatial product layers weights.
        """
        super(DgcSpn, self).__init__()
        self.in_size = in_size
        self.n_batch = n_batch
        self.prod_channels = prod_channels
        self.sum_channels = sum_channels
        self.n_pooling = n_pooling
        self.quantiles_loc = quantiles_loc
        self.optimize_scale = optimize_scale
        self.rand_state = rand_state

        # Instantiate the base layer
        self.base_layer = SpatialGaussianLayer(self.in_size, self.n_batch, self.quantiles_loc, self.optimize_scale)
        in_size = self.base_layer.output_size()

        # Add the initial pooling layers
        self.layers = torch.nn.ModuleList()
        for _ in range(self.n_pooling):
            # Add a spatial product layer (but with strides in order to reduce the dimensionality)
            self.layers.append(SpatialProductLayer(
                in_size, self.prod_channels, (2, 2), padding='valid',
                stride=(2, 2), dilation=(1, 1),
                rand_state=self.rand_state
            ))
            in_size = self.layers[-1].output_size()

        # Instantiate the inner layers
        depth = int(np.max(np.ceil(np.log2(in_size[1:]))).item())
        for k in range(depth):
            # Add a spatial product layer (with full padding and no strides)
            self.layers.append(SpatialProductLayer(
                in_size, self.prod_channels, (2, 2), padding='full',
                stride=(1, 1), dilation=(2 ** k, 2 ** k),
                rand_state=self.rand_state
            ))
            in_size = self.layers[-1].output_size()

            # Add a spatial sum layer
            self.layers.append(SpatialSumLayer(in_size, self.sum_channels))
            in_size = self.layers[-1].output_size()

        # Add the last product layer
        self.layers.append(SpatialProductLayer(
            in_size, self.prod_channels, (2, 2), padding='final',
            stride=(1, 1), dilation=(2 ** depth, 2 ** depth),
            rand_state=self.rand_state
        ))
        in_size = self.layers[-1].output_size()

        # Instantiate the spatial root layer
        self.root_layer = SpatialRootLayer(in_size, out_channels=1)

        # Initialize the scale clipper to apply, if specified
        self.scale_clipper = ScaleClipper() if self.optimize_scale else None

    def forward(self, x):
        """
        Compute the log-likelihood given some evidence.
        Random variables can be marginalized using NaN values.

        :param x: The inputs tensor.
        :return: The output of the model.
        """
        # Compute the base distributions log-likelihoods
        x = self.base_layer(x)

        # Forward through the inner layers
        for layer in self.layers:
            x = layer(x)

        # Forward through the root layer
        return self.root_layer(x)

    def apply_initializers(self, **kwargs):
        """
        Apply the initializers specified by the model.

        :param kwargs: The arguments to pass to the initializers of the model.
        """
        # Initialize the location parameters of the base layer using some data, if specified
        if self.quantiles_loc:
            quantiles_initializer(self.base_layer.loc, **kwargs)

    def apply_constraints(self):
        """
        Apply the constraints specified by the model.
        """
        # Apply the scale clipper to the base layer, if specified
        if self.optimize_scale:
            self.scale_clipper(self.base_layer)
