import numpy as np
import torch
from spnflow.utils.region import RegionGraph
from spnflow.torch.layers import GaussianLayer, ProductLayer, SumLayer, RootLayer


class RatSpn(torch.nn.Module):
    """RAT-SPN Keras model class."""
    def __init__(self,
                 n_features,
                 n_classes=1,
                 depth=2,
                 n_batch=2,
                 n_sum=2,
                 n_repetitions=1,
                 dropout=None,
                 optimize_scale=True,
                 rand_state=None,
                 ):
        """
        Initialize a RAT-SPN.

        :param n_features: The number of features.
        :param n_classes: The number of classes. Specify 1 in case of plain density estimation.
        :param depth: The depth of the network.
        :param n_batch: The number of distributions.
        :param n_sum: The number of sum nodes.
        :param n_repetitions: The number of independent repetitions of the region graph.
        :param dropout: The rate of the dropout layers (can be None).
        :param optimize_scale: Whatever to train scale and mean jointly.
        :param rand_state: The random state used to generate the random graph.
        """
        super(RatSpn, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.depth = depth
        self.n_batch = n_batch
        self.n_sum = n_sum
        self.n_repetitions = n_repetitions
        self.dropout = dropout
        self.optimize_scale = optimize_scale
        self.rand_state = rand_state

        # If necessary, instantiate a random state
        if self.rand_state is None:
            self.rand_state = np.random.RandomState(42)

        # Instantiate the region graph
        region_graph = RegionGraph(self.n_features, self.depth, self.rand_state)

        # Generate the layers
        self.rg_layers = region_graph.make_layers(self.n_repetitions)
        self.rg_layers = list(reversed(self.rg_layers))

        # Add the base distributions layer
        self.base_layer = GaussianLayer(
            self.n_features,
            self.depth,
            self.rg_layers[0],
            self.n_batch,
            self.optimize_scale
        )

        # Alternate between product and sum layer
        n_blocks = len(self.rg_layers[0]) // 2
        n_nodes = self.n_batch
        self.layers = torch.nn.ModuleList()
        for i in range(1, len(self.rg_layers) - 1):
            if i % 2 == 1:
                layer = ProductLayer(n_blocks, n_nodes)
                n_nodes = n_nodes ** 2
            else:
                layer = SumLayer(n_blocks, n_nodes, self.n_sum, self.dropout)
                n_blocks = n_blocks // 2
                n_nodes = self.n_sum
            self.layers.append(layer)

        # Add the sum root layer
        self.root_layer = RootLayer(n_classes, n_blocks * n_nodes)

    def forward(self, x):
        """
        Call the model.

        :param x: The inputs tensor.
        :return: The output of the model.
        """
        # Calculate the base log-likelihoods
        x = self.base_layer(x)

        # Forward through the inner layers
        for layer in self.layers:
            x = layer(x)

        # Flatten the result and forward through the sum root layer
        x = torch.flatten(x, start_dim=1)
        x = self.root_layer(x)
        return x
