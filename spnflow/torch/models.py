import torch
import numpy as np
from spnflow.utils.region import RegionGraph
from spnflow.torch.layers.ratspn import GaussianLayer, ProductLayer, SumLayer, RootLayer
from spnflow.torch.layers.flows import CouplingLayer, AutoregressiveLayer, BatchNormLayer
from spnflow.torch.layers.dgcspn import SpatialGaussianLayer, SpatialProductLayer, SpatialSumLayer, SpatialRootLayer


class RatSpn(torch.nn.Module):
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

    def forward(self, x):
        """
        Call the model.

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

    @property
    def constrained_module(self):
        """
        Get the constrained module.

        :return: The base distribution layer.
        """
        return self.base_layer


class RatSpnFlow(torch.nn.Module):
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
            self._build_real_nvp()
        elif self.flow == 'maf':
            self._build_maf()
        else:
            raise NotImplementedError('Unknown normalizing flow named \'' + self.flow + '\'')

    def _build_real_nvp(self):
        # Build the normalizing flows layers
        self.flows = torch.nn.ModuleList()
        reverse = False
        for _ in range(self.n_flows):
            self.flows.append(
                CouplingLayer(self.in_features, self.depth, self.units, self.activation, reverse=reverse)
            )
            if self.batch_norm:
                self.flows.append(
                    BatchNormLayer(self.in_features)
                )
            reverse = not reverse

    def _build_maf(self):
        # Build the normalizing flows layers
        self.flows = torch.nn.ModuleList()
        reverse = False
        for _ in range(self.n_flows):
            self.flows.append(
                AutoregressiveLayer(self.in_features, self.depth, self.units, self.activation, reverse=reverse)
            )
            if self.batch_norm:
                self.flows.append(
                    BatchNormLayer(self.in_features)
                )
            reverse = not reverse

    def forward(self, x):
        """
        Call the model.

        :param x: The inputs tensor.
        :return: The output of the model.
        """
        # Compute the log-likelihood of the model given complete evidence
        inv_log_det_jacobian = 0.0
        for flow in self.flows:
            x, ildj = flow(x)
            inv_log_det_jacobian += ildj
        return self.ratspn(x) + inv_log_det_jacobian

    @property
    def constrained_module(self):
        """
        Get the constrained module.

        :return: The base distribution layer.
        """
        return self.ratspn.base_layer


class SpatialSpn(torch.nn.Module):
    """Deep Generalized Convolutional SPN model class."""
    def __init__(self,
                 in_size,
                 n_batch=8,
                 prod_channels=16,
                 sum_channels=8,
                 optimize_scale=True,
                 rand_state=None,
                 ):
        """
        Initialize a SpatialSpn.

        :param in_size: The input size.
        :param n_batch: The number of output channels of the base layer.
        :param prod_channels: The number of output channels of spatial product layers.
        :param sum_channels: The number of output channels of spatial sum layers.
        :param optimize_scale: Whether to train scale and location jointly.
        :param rand_state: The random state used to initialize the spatial product layers weights.
        """
        super(SpatialSpn, self).__init__()
        self.in_size = in_size
        self.n_batch = n_batch
        self.prod_channels = prod_channels
        self.sum_channels = sum_channels
        self.optimize_scale = optimize_scale
        self.rand_state = rand_state

        # Instantiate the base layer
        self.base_layer = SpatialGaussianLayer(self.in_size, self.n_batch, self.optimize_scale)

        # Instantiate the inner layers
        in_size = self.base_layer.output_size()
        depth = int(np.max(np.ceil(np.log2(self.in_size))).item())
        self.layers = torch.nn.ModuleList()
        for k in range(depth):
            # Add a spatial product layer
            self.layers.append(SpatialProductLayer(
                in_size, self.prod_channels, (2, 2),
                padding='full', dilation=(2 ** k, 2 ** k),
                rand_state=self.rand_state
            ))
            in_size = self.layers[-1].output_size()

            # Add a spatial sum layer
            self.layers.append(SpatialSumLayer(in_size, self.sum_channels))
            in_size = self.layers[-1].output_size()

        # Add the last product layer
        self.layers.append(SpatialProductLayer(
            in_size, self.prod_channels, (2, 2),
            padding='final', dilation=(2 ** depth, 2 ** depth),
            rand_state=self.rand_state
        ))
        in_size = self.layers[-1].output_size()

        # Instantiate the spatial root layer
        self.root_layer = SpatialRootLayer(in_size, out_channels=1)

    def forward(self, x):
        """
        Call the model.

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

    @property
    def constrained_module(self):
        """
        Get the constrained module.

        :return: The base distribution layer.
        """
        return self.base_layer
