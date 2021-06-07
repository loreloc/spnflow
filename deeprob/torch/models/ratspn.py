import torch
import numpy as np

import deeprob.algorithms.inference
from deeprob.utils.region import RegionGraph
from deeprob.torch.models.abstract import AbstractModel
from deeprob.torch.layers.ratspn import GaussianLayer, BernoulliLayer, SumLayer, ProductLayer, RootLayer
from deeprob.torch.constraints import ScaleClipper


class AbstractRatSpn(AbstractModel):
    """Abstract RAT-SPN model class"""
    def __init__(self,
                 in_features,
                 dequantize=False,
                 logit=None,
                 out_classes=1,
                 rg_depth=2,
                 rg_repetitions=1,
                 n_batch=2,
                 n_sum=2,
                 in_dropout=None,
                 sum_dropout=None,
                 rand_state=None,
                 ):
        """
        Initialize a RAT-SPN.

        :param in_features: The number of input features.
        :param dequantize: Whether to apply the dequantization transformation.
        :param logit: The logit factor to use. Use None to disable the logit transformation.
        :param out_classes: The number of output classes. Specify 1 in case of plain density estimation.
        :param rg_depth: The depth of the region graph.
        :param rg_repetitions: The number of independent repetitions of the region graph.
        :param n_batch: The number of base distributions batches.
        :param n_sum: The number of sum nodes per region.
        :param in_dropout: The dropout rate for probabilistic dropout at distributions layer outputs. It can be None.
        :param sum_dropout: The dropout rate for probabilistic dropout at sum layers. It can be None.
        :param rand_state: The random state used to generate the random graph.
        """
        super(AbstractRatSpn, self).__init__(dequantize=dequantize, logit=logit)
        assert in_features > 0
        assert out_classes > 0
        assert rg_depth > 0
        assert rg_repetitions > 0
        assert n_batch > 0
        assert n_sum > 0
        assert in_dropout is None or 0.0 < in_dropout < 1.0
        assert sum_dropout is None or 0.0 < sum_dropout < 1.0
        self.in_features = in_features
        self.out_classes = out_classes
        self.rg_depth = rg_depth
        self.rg_repetitions = rg_repetitions
        self.n_batch = n_batch
        self.n_sum = n_sum
        self.in_dropout = in_dropout
        self.sum_dropout = sum_dropout
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
                layer = SumLayer(in_groups, in_nodes, self.n_sum, self.sum_dropout)
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
        # Preprocess the data
        x, inv_log_det_jacobian = self.preprocess(x)

        # Compute the base distributions log-likelihoods
        x = self.base_layer(x)

        # Forward through the inner layers
        for layer in self.layers:
            x = layer(x)

        # Forward through the root layer
        log_prob = self.root_layer(x)
        return log_prob + inv_log_det_jacobian

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
        n_samples = x.size(0)

        # Compute the base distributions log-likelihoods
        x = self.base_layer(x)

        # Compute in forward mode and gather the inner log-likelihoods
        for layer in self.layers:
            lls.append(x)
            x = layer(x)

        # Compute in forward mode through the root layer and get the class index,
        # if no target classes are specified
        if self.out_classes == 1:
            y = torch.zeros(n_samples).long()
        elif y is None:
            y = torch.argmax(self.root_layer(x), dim=1)

        # Get the root layer indices
        idx_group, idx_offset = deeprob.algorithms.inference.mpe(x, y)

        # Compute in top-down mode through the inner layers
        for i in range(len(self.layers) - 1, -1, -1):
            idx_group, idx_offset = deeprob.algorithms.inference.mpe(lls[i], idx_group, idx_offset)

        # Compute the maximum at posteriori inference at the base layer
        samples = deeprob.algorithms.inference.mpe(inputs, idx_group, idx_offset)

        # Unpreprocess the samples
        samples, _ = self.unpreprocess(samples)
        return samples

    @torch.no_grad()
    def sample(self, n_samples, y=None):
        """
        Compute some samples from the modeled distribution.

        :param n_samples: The number of samples.
        :param y: The target classes array. It can be None for unsupervised maximum at posteriori estimation.
        :return: The resulting samples.
        """
        # Compute in forward mode through the root layer and get the class index,
        # if no target classes are specified
        if self.out_classes == 1:
            y = torch.zeros(n_samples).long()
        elif y is None:
            y = torch.randint(self.out_classes, [n_samples])

        # Get the root layer indices
        idx_group, idx_offset = self.root_layer.sample(y)

        # Compute in top-down mode through the inner layers
        for i in range(len(self.layers) - 1, -1, -1):
            idx_group, idx_offset = self.layers[i].sample(idx_group, idx_offset)

        # Compute the maximum at posteriori inference at the base layer
        samples = self.base_layer.sample(idx_group, idx_offset)

        # Unpreprocess the samples
        samples, _ = self.unpreprocess(samples)
        return samples


class GaussianRatSpn(AbstractRatSpn):
    """Gaussian RAT-SPN model class."""
    def __init__(self,
                 in_features,
                 dequantize=False,
                 logit=None,
                 out_classes=1,
                 rg_depth=2,
                 rg_repetitions=1,
                 rg_batch=2,
                 rg_sum=2,
                 in_dropout=None,
                 sum_dropout=None,
                 rand_state=None,
                 uniform_loc=None,
                 optimize_scale=True
                 ):
        """
        Initialize a RAT-SPN.

        :param in_features: The number of input features.
        :param dequantize: Whether to apply the dequantization transformation.
        :param logit: Whether to apply logit transformation on the input layer.
        :param out_classes: The number of output classes. Specify 1 in case of plain density estimation.
        :param rg_depth: The depth of the region graph.
        :param rg_repetitions: The number of independent repetitions of the region graph.
        :param rg_batch: The number of base distributions batches.
        :param rg_sum: The number of sum nodes per region.
        :param in_dropout: The dropout rate for probabilistic dropout at distributions layer outputs. It can be None.
        :param sum_dropout: The dropout rate for probabilistic dropout at sum layers. It can be None.
        :param rand_state: The random state used to generate the random graph.
        :param uniform_loc: The optional uniform distribution parameters for location initialization.
        :param optimize_scale: Whether to train scale and location jointly.
        """
        super(GaussianRatSpn, self).__init__(
            in_features, dequantize, logit, out_classes, rg_depth, rg_repetitions,
            rg_batch, rg_sum, in_dropout, sum_dropout, rand_state
        )
        assert uniform_loc is None or uniform_loc[0] < uniform_loc[1], \
            "The first parameter of the uniform distribution most be less than the second one."
        self.optimize_scale = optimize_scale
        self.uniform_loc = uniform_loc
        self.scale_clipper = ScaleClipper() if self.optimize_scale else None

        # Instantiate the base distributions layer
        self.base_layer = GaussianLayer(
            self.in_features,
            self.n_batch,
            self.rg_layers[0],
            self.rg_depth,
            self.in_dropout,
            self.uniform_loc,
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
                 sum_dropout=None,
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
        :param sum_dropout: The dropout rate for probabilistic dropout at product layer outputs. It can be None.
        :param rand_state: The random state used to generate the random graph.
        """
        super(BernoulliRatSpn, self).__init__(
            in_features, False, None, out_classes, rg_depth, rg_repetitions,
            n_batch, n_sum, in_dropout, sum_dropout, rand_state
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
