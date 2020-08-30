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

    @abc.abstractmethod
    def forward(self, x):
        pass

    def apply_initializers(self, **kwargs):
        pass

    def apply_constraints(self):
        pass


class RealNVP(AbstractModel):
    """Real Non-Volume-Preserving (RealNVP) normalizing flow model."""
    def __init__(self,
                 in_features,
                 in_base=None,
                 n_flows=5,
                 batch_norm=True,
                 depth=1,
                 units=128,
                 activation=torch.nn.ReLU,
                 ):
        """
        Initialize a RealNVP.

        :param in_features: The number of input features.
        :param in_base: The input base distribution to use. If None it is the standard Normal distribution.
        :param n_flows: The number of sequential coupling flows.
        :param batch_norm: Whether to apply batch normalization after each coupling layer.
        :param depth: The number of hidden layers of flows conditioners.
        :param units: The number of hidden units per layer of flows conditioners.
        :param activation: The activation class to use for the flows conditioners hidden layers.

        """
        super(RealNVP, self).__init__()
        self.in_features = in_features
        self.n_flows = n_flows
        self.batch_norm = batch_norm
        self.depth = depth
        self.units = units
        self.activation = activation

        # Build the base distribution, if necessary
        if in_base is None:
            self.in_base = torch.distributions.Normal(0.0, 1.0)
        else:
            self.in_base = in_base

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
            x, ildj = layer(x)
            inv_log_det_jacobian += ildj
        prior = self.in_base.log_prob(x)
        return torch.sum(prior, dim=1) + inv_log_det_jacobian


class MAF(AbstractModel):
    """Masked Autoregressive Flow (MAF) normalizing flow model."""
    def __init__(self,
                 in_features,
                 in_base=None,
                 n_flows=5,
                 batch_norm=True,
                 depth=1,
                 units=128,
                 activation=torch.nn.ReLU,
                 ):
        """
        Initialize a MAF.

        :param in_features: The number of input features.
        :param in_base: The input base distribution to use. If None it is the standard Normal distribution.
        :param n_flows: The number of sequential autoregressive layers.
        :param batch_norm: Whether to apply batch normalization after each autoregressive layer.
        :param depth: The number of hidden layers of flows conditioners.
        :param units: The number of hidden units per layer of flows conditioners.
        :param activation: The activation class to use for the flows conditioners hidden layers.
        """
        super(MAF, self).__init__()
        self.in_features = in_features
        self.n_flows = n_flows
        self.batch_norm = batch_norm
        self.depth = depth
        self.units = units
        self.activation = activation

        # Build the base distribution, if necessary
        if in_base is None:
            self.in_base = torch.distributions.Normal(0.0, 1.0)
        else:
            self.in_base = in_base

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
            x, ildj = layer(x)
            inv_log_det_jacobian += ildj
        prior = self.in_base.log_prob(x)
        return torch.sum(prior, dim=1) + inv_log_det_jacobian


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
            self.in_features,
            self.n_batch,
            self.rg_layers[0],
            self.rg_depth,
            self.optimize_scale
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
            self.in_features, 1,
            self.rg_depth, self.rg_repetitions,
            self.n_batch, self.n_sum,
            self.dropout, self.optimize_scale, self.rand_state
        )

        # Build the normalizing flow layers
        if self.flow == 'nvp':
            self.flows = RealNVP(
                self.in_features, self.ratspn, self.n_flows,
                self.batch_norm, self.depth, self.units, self.activation
            )
        elif self.flow == 'maf':
            self.flows = MAF(
                self.in_features, self.ratspn, self.n_flows,
                self.batch_norm, self.depth, self.units, self.activation
            )
        else:
            raise NotImplementedError('Unknown normalizing flow named \'' + self.flow + '\'')

        # Initialize the scale clipper to apply, if specified
        self.scale_clipper = ScaleClipper() if self.optimize_scale else None

    def forward(self, x):
        """
        Compute the log-likelihood given complete evidence.

        :param x: The inputs tensor.
        :return: The output of the model.
        """
        return self.flows(x)

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
