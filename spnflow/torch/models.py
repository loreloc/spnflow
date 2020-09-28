import abc
import torch
import numpy as np
from spnflow.utils.region import RegionGraph
from spnflow.torch.layers.ratspn import GaussianLayer, ProductLayer, SumLayer, RootLayer
from spnflow.torch.layers.flows import CouplingLayer, AutoregressiveLayer, BatchNormLayer, LogitLayer
from spnflow.torch.layers.dgcspn import SpatialGaussianLayer, SpatialProductLayer, SpatialSumLayer, SpatialRootLayer
from spnflow.torch.constraints import ScaleClipper


class AbstractModel(abc.ABC, torch.nn.Module):
    """Abstract class for deep probabilistic models."""
    def __init__(self):
        super(AbstractModel, self).__init__()

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def mpe(self, x):
        pass

    @abc.abstractmethod
    def sample(self, n_samples):
        pass

    def log_prob(self, x):
        return self(x)

    def apply_constraints(self):
        pass


class NormalizingFlow(AbstractModel):
    """Normalizing Flow abstract model."""
    def __init__(self, in_features, n_flows=5, logit=False, in_base=None):
        """
        Initialize an abstract Normalizing Flow model.

        :param in_features: The number of input features.
        :param n_flows: The number of sequential coupling flows.
        :param logit: Whether to apply logit transformation on the input layer.
        :param in_base: The input base distribution to use. If None, the standard Normal distribution is used.
        """
        super(NormalizingFlow, self).__init__()
        self.in_features = in_features
        self.n_flows = n_flows
        self.logit = logit

        # Build the base distribution, if necessary
        if in_base is None:
            self.in_base_loc = torch.nn.Parameter(torch.zeros([self.in_features], requires_grad=False))
            self.in_base_scale = torch.nn.Parameter(torch.ones([self.in_features], requires_grad=False))
            self.in_base = torch.distributions.Normal(self.in_base_loc, self.in_base_scale)
        else:
            self.in_base = in_base

        # Initialize the normalizing flow layers
        # Moreover, append the logit transformation, if specified
        self.layers = torch.nn.ModuleList()
        if self.logit:
            self.layers.append(LogitLayer())

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
        prior = torch.sum(self.in_base.log_prob(x), dim=1, keepdim=True)
        return prior + inv_log_det_jacobian

    @torch.no_grad()
    def mpe(self, x):
        raise NotImplementedError('Maximum at posteriori estimation is not implemented for Normalizing Flows')

    @torch.no_grad()
    def sample(self, n_samples):
        """
        Sample some values from the modeled distribution.

        :param n_samples: The number of samples.
        :return: The samples.
        """
        x = self.in_base.sample([n_samples])
        for layer in reversed(self.layers):
            x, ldj = layer.forward(x)
        return x


class RealNVP(NormalizingFlow):
    """Real Non-Volume-Preserving (RealNVP) normalizing flow model."""
    def __init__(self,
                 in_features,
                 n_flows=5,
                 batch_norm=True,
                 depth=1,
                 units=128,
                 activation=torch.nn.ReLU,
                 logit=False,
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
        :param logit: Whether to apply logit transformation on the input layer.
        :param in_base: The input base distribution to use. If None, the standard Normal distribution is used.
        """
        super(RealNVP, self).__init__(in_features, n_flows, logit, in_base)
        self.batch_norm = batch_norm
        self.depth = depth
        self.units = units
        self.activation = activation

        # Build the coupling layers
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


class MAF(NormalizingFlow):
    """Masked Autoregressive Flow (MAF) normalizing flow model."""
    def __init__(self,
                 in_features,
                 n_flows=5,
                 batch_norm=True,
                 depth=1,
                 units=128,
                 activation=torch.nn.ReLU,
                 sequential=True,
                 logit=False,
                 in_base=None,
                 rand_state=None
                 ):
        """
        Initialize a MAF.

        :param in_features: The number of input features.
        :param n_flows: The number of sequential autoregressive layers.
        :param batch_norm: Whether to apply batch normalization after each autoregressive layer.
        :param depth: The number of hidden layers of flows conditioners.
        :param units: The number of hidden units per layer of flows conditioners.
        :param activation: The activation class to use for the flows conditioners hidden layers.
        :param sequential: If True build masks degrees sequentially, otherwise randomly.
        :param logit: Whether to apply logit transformation on the input layer.
        :param in_base: The input base distribution to use. If None, the standard Normal distribution is used.
        :param rand_state: The random state used to generate the masks degrees. Used only if sequential is False.
        """
        super(MAF, self).__init__(in_features, n_flows, logit, in_base)
        self.batch_norm = batch_norm
        self.depth = depth
        self.units = units
        self.activation = activation
        self.sequential = sequential
        self.rand_state = rand_state

        # If necessary, instantiate a random state
        if not self.sequential and self.rand_state is None:
            self.rand_state = np.random.RandomState(42)

        # Build the autoregressive layers
        reverse = False
        for _ in range(self.n_flows):
            self.layers.append(
                AutoregressiveLayer(
                    self.in_features, self.depth, self.units, self.activation,
                    sequential=self.sequential, reverse=reverse, rand_state=self.rand_state
                )
            )

            # Append batch normalization after each layer, if specified
            if self.batch_norm:
                self.layers.append(BatchNormLayer(self.in_features))

            # Invert the input ordering
            reverse = not reverse


class RatSpn(AbstractModel):
    """RAT-SPN model class."""
    def __init__(self,
                 in_features,
                 out_classes=1,
                 rg_depth=2,
                 rg_repetitions=1,
                 n_batch=2,
                 n_sum=2,
                 optimize_scale=True,
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
        :param optimize_scale: Whether to train scale and location jointly.
        :param in_dropout: The dropout rate for probabilistic dropout at distributions layer outputs. It can be None.
        :param prod_dropout: The dropout rate for probabilistic dropout at product layer outputs. It can be None.
        :param rand_state: The random state used to generate the random graph.
        """
        super(RatSpn, self).__init__()
        self.in_features = in_features
        self.out_classes = out_classes
        self.rg_depth = rg_depth
        self.rg_repetitions = rg_repetitions
        self.n_batch = n_batch
        self.n_sum = n_sum
        self.optimize_scale = optimize_scale
        self.in_dropout = in_dropout
        self.prod_dropout = prod_dropout
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
            self.optimize_scale,
            self.in_dropout
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
                layer = SumLayer(in_groups, in_nodes, self.n_sum, self.prod_dropout)
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

    def apply_constraints(self):
        """
        Apply the constraints specified by the model.
        """
        # Apply the scale clipper to the base layer, if specified
        if self.optimize_scale:
            self.scale_clipper(self.base_layer)


class DgcSpn(AbstractModel):
    """Deep Generalized Convolutional SPN model class."""
    def __init__(self,
                 in_size,
                 out_classes=1,
                 n_batch=8,
                 sum_channels=8,
                 depthwise=False,
                 n_pooling=0,
                 dropout=None,
                 optimize_scale=True,
                 quantiles_loc=None,
                 uniform_loc=None,
                 rand_state=None,
                 ):
        """
        Initialize a SpatialSpn.

        :param in_size: The input size.
        :param out_classes: The number of output classes. Specify 1 in case of plain density estimation.
        :param n_batch: The number of output channels of the base layer.
        :param sum_channels: The number of output channels of spatial sum layers.
        :param depthwise: Whether to use depthwise convolutions as product layers.
        :param n_pooling: The number of initial pooling product layers.
        :param dropout: The dropout rate for probabilistic dropout at sum layer inputs. It can be None.
        :param optimize_scale: Whether to train scale.
        :param quantiles_loc: The mean quantiles for location initialization. It can be None.
        :param uniform_loc: The uniform range for location initialization. It can be None.
        :param rand_state: The random state used to initialize the spatial product layers weights.
                           Used only if depthwise is False.
        """
        super(DgcSpn, self).__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.n_batch = n_batch
        self.sum_channels = sum_channels
        self.depthwise = depthwise
        self.n_pooling = n_pooling
        self.dropout = dropout
        self.optimize_scale = optimize_scale
        self.quantiles_loc = quantiles_loc
        self.uniform_loc = uniform_loc
        self.rand_state = rand_state

        # If necessary, instantiate a random state
        if not self.depthwise and self.rand_state is None:
            self.rand_state = np.random.RandomState(42)

        # Instantiate the base layer
        self.base_layer = SpatialGaussianLayer(
            self.in_size, self.n_batch, self.optimize_scale, self.quantiles_loc, self.uniform_loc
        )
        in_size = self.base_layer.out_size

        # Add the initial pooling layers, if specified
        self.layers = torch.nn.ModuleList()
        for _ in range(self.n_pooling):
            # Add a spatial product layer (but with strides in order to reduce the dimensionality)
            # Also, use depthwise convolutions for pooling
            pooling = SpatialProductLayer(
                in_size, depthwise=True, kernel_size=(2, 2), padding='valid', stride=(2, 2), dilation=(1, 1)
            )
            self.layers.append(pooling)
            in_size = pooling.out_size

            # Add a spatial sum layer
            spatial_sum = SpatialSumLayer(in_size, self.sum_channels, self.dropout)
            self.layers.append(spatial_sum)
            in_size = spatial_sum.out_size

        # Instantiate the inner layers
        depth = int(np.max(np.ceil(np.log2(in_size[1:]))).item())
        for k in range(depth):
            # Add a spatial product layer (with full padding and no strides)
            spatial_prod = SpatialProductLayer(
                in_size, depthwise=self.depthwise, kernel_size=(2, 2), padding='full',
                stride=(1, 1), dilation=(2 ** k, 2 ** k), rand_state=self.rand_state
            )
            self.layers.append(spatial_prod)
            in_size = spatial_prod.out_size

            # Add a spatial sum layer
            spatial_sum = SpatialSumLayer(in_size, self.sum_channels, self.dropout)
            self.layers.append(spatial_sum)
            in_size = spatial_sum.out_size

        # Add the last product layer
        spatial_prod = SpatialProductLayer(
            in_size, depthwise=self.depthwise, kernel_size=(2, 2), padding='final',
            stride=(1, 1), dilation=(2 ** depth, 2 ** depth), rand_state=self.rand_state
        )
        self.layers.append(spatial_prod)
        in_size = spatial_prod.out_size

        # Instantiate the spatial root layer
        self.root_layer = SpatialRootLayer(in_size, self.out_classes)

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

    def mpe(self, x):
        """
        Compute the maximum at posteriori estimation.
        Random variables can be marginalized using NaN values.

        :param x: The inputs tensor.
        :return: The output of the model.
        """
        # Compute the base distribution log-likelihoods
        z = self.base_layer(x)

        # Forward through the inner layers
        y = z
        for layer in self.layers:
            y = layer(y)

        # Forward through the root layer
        y = self.root_layer(y)

        # Compute the gradients at distribution leaves
        (z_grad,) = torch.autograd.grad(y, z, grad_outputs=torch.ones_like(y))

        with torch.no_grad():
            # Compute the maximum at posteriori estimate using leaves gradients
            mode = self.base_layer.loc
            estimates = torch.sum(torch.unsqueeze(z_grad, dim=2) * mode, dim=1)
            return torch.where(torch.isnan(x), estimates, x)

    def sample(self, n_samples, y=None):
        raise NotImplementedError('Sampling is not implemented for DGC-SPNs')

    def apply_constraints(self):
        """
        Apply the constraints specified by the model.
        """
        # Apply the scale clipper to the base layer, if specified
        if self.optimize_scale:
            self.scale_clipper(self.base_layer)
