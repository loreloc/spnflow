import torch
import numpy as np

from spnflow.utils.region import RegionGraph
from spnflow.torch.models.abstract import AbstractModel
from spnflow.torch.layers.ratspn import GaussianLayer, BernoulliLayer, SumLayer, ProductLayer, RootLayer
from spnflow.torch.constraints import ScaleClipper


class AbstractRatSpn(AbstractModel):
    """Abstract RAT-SPN model class"""
    def __init__(self,
                 in_features,
                 out_classes=1,
                 rg_depth=2,
                 rg_repetitions=1,
                 n_batch=2,
                 n_sum=2,
                 in_dropout=None,
                 prod_dropout=None,
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
        :param in_dropout: The dropout rate for probabilistic dropout at distributions layer outputs. It can be None.
        :param prod_dropout: The dropout rate for probabilistic dropout at product layer outputs. It can be None.
        :param rand_state: The random state used to generate the random graph.
        """
        super(AbstractRatSpn, self).__init__()
        assert in_features > 0
        assert out_classes > 0
        assert rg_depth > 0
        assert rg_repetitions > 0
        assert n_batch > 0
        assert n_sum > 0
        assert in_dropout is None or 0.0 < in_dropout < 1.0
        assert prod_dropout is None or 0.0 < prod_dropout < 1.0
        self.in_features = in_features
        self.out_classes = out_classes
        self.rg_depth = rg_depth
        self.rg_repetitions = rg_repetitions
        self.n_batch = n_batch
        self.n_sum = n_sum
        self.in_dropout = in_dropout
        self.prod_dropout = prod_dropout
        self.rand_state = rand_state
        self.base_layer = None
        self.layers = None
        self.root_layer = None

        # If necessary, instantiate a random state
        if self.rand_state is None:
            self.rand_state = np.random.RandomState(42)

        # Instantiate the region graph
        region_graph = RegionGraph(self.in_features, self.rg_depth, self.rand_state)

        # Generate the layers
        self.rg_layers = region_graph.make_layers(self.rg_repetitions)
        self.rg_layers = list(reversed(self.rg_layers))

    def build(self):
        """
        Build the RatSpn Torch model.
        """
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
                layer = SumLayer(in_groups, in_nodes, self.n_sum, self.prod_dropout)
                in_groups = layer.out_regions
                in_nodes = layer.out_nodes
            self.layers.append(layer)

        # Instantiate the root layer
        self.root_layer = RootLayer(in_groups, in_nodes, self.out_classes)

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
        y = y if y else torch.argmax(self.root_layer(x), dim=1)

        # Get the root layer indices
        idx = self.root_layer.mpe(x, y)

        # Compute in top-down mode through the inner layers
        rev_lls = list(reversed(lls))
        rev_layers = list(reversed(self.layers))
        for layer, ll in zip(rev_layers, rev_lls):
            idx = layer.mpe(ll, idx)

        # Compute the maximum at posteriori inference at the base layer
        return self.base_layer.mpe(inputs, idx)

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

        # Get the root layer indices
        idx = self.root_layer.sample(y)

        # Compute in top-down mode through the inner layers
        for layer in reversed(self.layers):
            idx = layer.sample(idx)

        # Sample at the base layer
        return self.base_layer.sample(idx)


class GaussianRatSpn(AbstractRatSpn):
    """Gaussian RAT-SPN model class."""
    def __init__(self,
                 in_features,
                 out_classes=1,
                 rg_depth=2,
                 rg_repetitions=1,
                 n_batch=2,
                 n_sum=2,
                 in_dropout=None,
                 prod_dropout=None,
                 rand_state=None,
                 optimize_scale=True
                 ):
        """
        Initialize a RAT-SPN.

        :param in_features: The number of input features.
        :param out_classes: The number of output classes. Specify 1 in case of plain density estimation.
        :param rg_depth: The depth of the region graph.
        :param rg_repetitions: The number of independent repetitions of the region graph.
        :param n_batch: The number of base distributions batches.
        :param n_sum: The number of sum nodes per region.
        :param in_dropout: The dropout rate for probabilistic dropout at distributions layer outputs. It can be None.
        :param prod_dropout: The dropout rate for probabilistic dropout at product layer outputs. It can be None.
        :param rand_state: The random state used to generate the random graph.
        :param optimize_scale: Whether to train scale and location jointly.
        """
        super(GaussianRatSpn, self).__init__(
            in_features, out_classes, rg_depth, rg_repetitions,
            n_batch, n_sum, in_dropout, prod_dropout, rand_state
        )
        self.optimize_scale = optimize_scale
        self.scale_clipper = ScaleClipper() if self.optimize_scale else None

        # Instantiate the base distributions layer
        self.base_layer = GaussianLayer(
            self.in_features,
            self.n_batch,
            self.rg_layers[0],
            self.rg_depth,
            self.in_dropout,
            self.optimize_scale
        )

        # Build the Torch model
        super(GaussianRatSpn, self).build()

    def apply_constraints(self):
        """
        Apply the constraints specified by the model.
        """
        # Apply the scale clipper to the base layer, if specified
        if self.optimize_scale:
            self.scale_clipper(self.base_layer)


class BernoulliRatSpn(AbstractRatSpn):
    """Bernoulli RAT-SPN model class."""
    def __init__(self,
                 in_features,
                 out_classes=1,
                 rg_depth=2,
                 rg_repetitions=1,
                 n_batch=2,
                 n_sum=2,
                 in_dropout=None,
                 prod_dropout=None,
                 rand_state=None
                 ):
        """
        Initialize a RAT-SPN.

        :param in_features: The number of input features.
        :param out_classes: The number of output classes. Specify 1 in case of plain density estimation.
        :param rg_depth: The depth of the region graph.
        :param rg_repetitions: The number of independent repetitions of the region graph.
        :param n_batch: The number of base distributions batches.
        :param n_sum: The number of sum nodes per region.
        :param in_dropout: The dropout rate for probabilistic dropout at distributions layer outputs. It can be None.
        :param prod_dropout: The dropout rate for probabilistic dropout at product layer outputs. It can be None.
        :param rand_state: The random state used to generate the random graph.
        """
        super(BernoulliRatSpn, self).__init__(
            in_features, out_classes, rg_depth, rg_repetitions,
            n_batch, n_sum, in_dropout, prod_dropout, rand_state
        )

        # Instantiate the base distributions layer
        self.base_layer = BernoulliLayer(
            self.in_features,
            self.n_batch,
            self.rg_layers[0],
            self.rg_depth,
            self.in_dropout
        )

        # Build the Torch model
        super(BernoulliRatSpn, self).build()
