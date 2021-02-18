import torch
import numpy as np

from spnflow.torch.models.abstract import AbstractModel
from spnflow.torch.layers.dgcspn import SpatialGaussianLayer, SpatialProductLayer, SpatialSumLayer, SpatialRootLayer
from spnflow.torch.constraints import ScaleClipper


class DgcSpn(AbstractModel):
    """Deep Generalized Convolutional SPN model class."""
    def __init__(self,
                 in_size,
                 out_classes=1,
                 n_batch=8,
                 sum_channels=8,
                 depthwise=False,
                 n_pooling=0,
                 optimize_scale=True,
                 in_dropout=None,
                 prod_dropout=None,
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
        :param optimize_scale: Whether to train scale and location jointly.
        :param in_dropout: The dropout rate for probabilistic dropout at distributions layer outputs. It can be None.
        :param prod_dropout: The dropout rate for probabilistic dropout at product layer outputs. It can be None.
        :param quantiles_loc: The mean quantiles for location initialization. It can be None.
        :param uniform_loc: The uniform range for location initialization. It can be None.
        :param rand_state: The random state used to initialize the spatial product layers weights.
                           Used only if depthwise is False.
        """
        super(DgcSpn, self).__init__()
        assert len(in_size) == 3 and in_size[0] > 0 and in_size[1] > 0 and in_size[2] > 0
        assert out_classes > 0
        assert n_batch > 0
        assert sum_channels > 0
        assert n_pooling >= 0
        assert in_dropout is None or 0.0 < in_dropout < 1.0
        assert prod_dropout is None or 0.0 < prod_dropout < 1.0
        assert quantiles_loc is None or uniform_loc is None,\
            'At least one between quantiles_loc and uniform_loc must be None'
        assert quantiles_loc is None or len(quantiles_loc.shape) == 4
        assert uniform_loc is None or (len(uniform_loc) == 2 and uniform_loc[0] < uniform_loc[1])
        self.in_size = in_size
        self.out_classes = out_classes
        self.n_batch = n_batch
        self.sum_channels = sum_channels
        self.depthwise = depthwise
        self.n_pooling = n_pooling
        self.optimize_scale = optimize_scale
        self.in_dropout = in_dropout
        self.prod_dropout = prod_dropout
        self.uniform_loc = uniform_loc
        self.rand_state = rand_state

        self.quantiles_loc = None
        if quantiles_loc is not None:
            self.quantiles_loc = torch.tensor(quantiles_loc)

        # If necessary, instantiate a random state
        if not self.depthwise and self.rand_state is None:
            self.rand_state = np.random.RandomState(42)

        # Instantiate the base layer
        self.base_layer = SpatialGaussianLayer(
            self.in_size,
            self.n_batch,
            self.optimize_scale,
            self.in_dropout,
            self.quantiles_loc,
            self.uniform_loc
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
            spatial_sum = SpatialSumLayer(in_size, self.sum_channels, self.prod_dropout)
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
            spatial_sum = SpatialSumLayer(in_size, self.sum_channels, self.prod_dropout)
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

    @torch.no_grad()
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

    @torch.no_grad()
    def sample(self, n_samples, y=None):
        raise NotImplementedError('Sampling is not implemented for DGC-SPNs')

    def apply_constraints(self):
        """
        Apply the constraints specified by the model.
        """
        # Apply the scale clipper to the base layer, if specified
        if self.optimize_scale:
            self.scale_clipper(self.base_layer)
