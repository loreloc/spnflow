import numpy as np
import torch
from spnflow.utils.region import RegionGraph
from spnflow.torch.layers import GaussianLayer, ProductLayer, SumLayer, RootLayer, CouplingLayer


class RatSpn(torch.nn.Module):
    """RAT-SPN Keras model class."""
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
                 flow='real_nvp',
                 n_flows=5,
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
        :param flow: The normalizing flow kind. At the moment, only 'nvp' is supported.
        :param n_flows: The number of sequential normalizing flows.
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
        else:
            raise NotImplementedError('Unknown normalizing flow named \'' + self.flow + '\'')

    def _build_real_nvp(self):
        # Build the normalizing flows
        self.flows = torch.nn.ModuleList()
        reverse = False
        for _ in range(self.n_flows):
            self.flows.append(
                CouplingLayer(self.in_features, self.depth, self.units, self.activation, reverse=reverse)
            )
            reverse = not reverse

    def forward(self, x):
        """
        Call the model.

        :param x: The inputs tensor.
        :return: The output of the model.
        """
        # Compute the log-likelihood of the model given complete evidence
        inv_log_det_jacobian = torch.zeros(size=(x.size(0), 1), device=x.device)
        for flow in self.flows:
            x, dj = flow(x)
            inv_log_det_jacobian += dj
        return self.ratspn(x) + inv_log_det_jacobian

    @property
    def constrained_module(self):
        """
        Get the constrained module.

        :return: The base distribution layer.
        """
        return self.ratspn.base_layer
