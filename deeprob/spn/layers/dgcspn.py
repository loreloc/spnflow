import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGaussianLayer(nn.Module):
    """Spatial Gaussian input layer."""
    def __init__(self, in_size, out_channels, optimize_scale, dropout=None, quantiles_loc=None, uniform_loc=None):
        """
        Initialize a Spatial Gaussian input layer.

        :param in_size: The size of the input tensor.
        :param out_channels: The number of output channels.
        :param optimize_scale: Whether to optimize scale.
        :param dropout: The leaf nodes dropout rate. It can be None.
        :param quantiles_loc: The mean quantiles for location initialization. It can be None.
        :param uniform_loc: The uniform range for location initialization. It can be None.
        """
        super(SpatialGaussianLayer, self).__init__()
        self.in_size = in_size
        self.out_channels = out_channels
        self.optimize_scale = optimize_scale
        self.dropout = dropout
        self.quantiles_loc = quantiles_loc
        self.uniform_loc = uniform_loc

        # Instantiate the location parameter
        if self.quantiles_loc is not None:
            self.loc = nn.Parameter(self.quantiles_loc, requires_grad=False)
        elif self.uniform_loc is not None:
            low, high = self.uniform_loc
            linspace = torch.linspace(low, high, steps=self.out_channels).view(-1, 1, 1, 1)
            self.loc = nn.Parameter(linspace.repeat(1, *self.in_size), requires_grad=True)
        else:
            self.loc = nn.Parameter(torch.randn(self.out_channels, *self.in_size), requires_grad=True)

        # Instantiate the scale parameter
        if self.optimize_scale:
            self.scale = torch.nn.Parameter(
                0.5 + 0.1 * torch.tanh(torch.randn(self.out_channels, *self.in_size)),
                requires_grad=True
            )
        else:
            self.scale = torch.nn.Parameter(
                torch.ones(self.out_channels, *self.in_size),
                requires_grad=False
            )

        # Instantiate the multi-batch normal distribution
        self.distribution = torch.distributions.Normal(self.loc, self.scale)

        # Initialize some useful constants
        self.register_buffer('zero', torch.zeros(1))
        self.register_buffer('nan', torch.tensor([np.nan]))

    @property
    def in_channels(self):
        return self.in_size[0]

    @property
    def in_height(self):
        return self.in_size[1]

    @property
    def in_width(self):
        return self.in_size[2]

    @property
    def out_size(self):
        return self.out_channels, self.in_height, self.in_width

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Compute the log-likelihoods
        x = torch.unsqueeze(x, dim=1)
        x = self.distribution.log_prob(x)

        # Apply the input dropout, if specified
        if self.training and self.dropout is not None:
            x = torch.where(torch.gt(torch.rand_like(x), self.dropout), x, self.nan)

        # Marginalize missing values (denoted with NaNs)
        x = torch.where(torch.isnan(x), self.zero, x)

        # This implementation assumes independence between channels of the same pixel random variables
        return torch.sum(x, dim=2)


class SpatialProductLayer(nn.Module):
    """Spatial Product layer class."""
    def __init__(self, in_size, depthwise, kernel_size, padding, stride, dilation, rand_state=None):
        """
        Initialize a Spatial Product layer.

        :param in_size: The input tensor size. It should be (in_channels, in_height, in_width).
        :param depthwise: Whether to use depthwise convolutions.
        :param kernel_size: The size of the kernels.
        :param stride: The strides to use.
        :param padding: The padding mode to use. It can be 'valid', 'full' or 'final'.
        :param dilation: The space between the kernel points.
        :param rand_state: The random state used to generate the weight mask. It can be None if depthwise is True.
        """
        super(SpatialProductLayer, self).__init__()
        self.in_size = in_size
        self.depthwise = depthwise
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.rand_state = rand_state
        self.groups = self.in_channels if self.depthwise else 1
        self.out_channels = self.in_channels if self.depthwise else self.in_channels * np.prod(self.kernel_size)

        # Compute the effective kernel size (due to dilation)
        dh, dw = self.dilation
        kh, kw = self.kernel_size
        kah, kaw = (kh - 1) * dh + 1, (kw - 1) * dw + 1
        self.effective_kernel_size = (kah, kaw)

        # Compute the padding to apply
        if self.padding == 'valid':
            self.pad = [0, 0, 0, 0]
        elif self.padding == 'full':
            self.pad = [kah - 1, kah - 1, kaw - 1, kaw - 1]
        elif self.padding == 'final':
            self.pad = [(kah - 1) * 2 - self.in_height, 0, (kaw - 1) * 2 - self.in_width, 0]
        else:
            raise NotImplementedError('Unknown padding mode named ' + self.padding)

        # Build the convolution kernels
        if self.depthwise:
            weight = self._build_depthwise_kernels()
        else:
            weight = self._build_sparse_kernels(rand_state)

        # Initialize the weight buffer
        self.register_buffer('weight', torch.tensor(weight))

    @property
    def in_channels(self):
        return self.in_size[0]

    @property
    def in_height(self):
        return self.in_size[1]

    @property
    def in_width(self):
        return self.in_size[2]

    @property
    def out_size(self):
        kah, kaw = self.effective_kernel_size
        out_height = self.pad[0] + self.pad[1] + self.in_height - kah + 1
        out_width = self.pad[2] + self.pad[3] + self.in_width - kaw + 1
        out_height = int(np.ceil(out_height / self.stride[0]).item())
        out_width = int(np.ceil(out_width / self.stride[1]).item())
        return self.out_channels, out_height, out_width

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Pad the input and compute the log-likelihoods
        x = F.pad(x, self.pad)
        return F.conv2d(
            x, self.weight, stride=self.stride, dilation=self.dilation, groups=self.groups
        )

    def _build_sparse_kernels(self, rand_state):
        """
        Generate sparse (and non-depthwise) convolution kernels randomly.

        :param rand_state: The random state used to generate the weight mask.
        :return: The convolution kernels.
        """
        kernels_idx = []
        kh, kw = self.kernel_size

        # Generate a sparse representation of weight
        for _ in range(kh * kw):
            idx = np.arange(self.out_channels) % self.in_channels
            rand_state.shuffle(idx)
            kernels_idx.append(idx)
        kernels_idx = np.stack(kernels_idx, axis=1)
        kernels_idx = np.reshape(kernels_idx, (self.out_channels, 1, kh, kw))

        # Generate a dense representation of weight, given the sparse representation
        weight = np.ones((self.out_channels, self.in_channels, kh, kw))
        weight = weight * np.arange(self.in_channels).reshape((1, self.in_channels, 1, 1))
        weight = np.equal(weight, kernels_idx)
        return weight.astype(np.float32)

    def _build_depthwise_kernels(self):
        """
        Generate depthwise convolution kernels.

        :return: The convolution kernels.
        """
        kh, kw = self.kernel_size
        return np.ones((self.out_channels, 1, kh, kw), dtype=np.float32)


class SpatialSumLayer(nn.Module):
    """Spatial Sum layer class."""
    def __init__(self, in_size, out_channels, dropout=None):
        """
        Initialize a Spatial Sum layer.

        :param in_size: The input tensor size. It should be (in_channels, in_height, in_width).
        :param out_channels: The number of output channels.
        :param dropout: The input nodes dropout rate. It can be None.
        """
        super(SpatialSumLayer, self).__init__()
        self.in_size = in_size
        self.out_channels = out_channels
        self.dropout = dropout

        # Initialize the weight tensor
        self.weight = nn.Parameter(
            torch.normal(0.0, 1e-1, [self.out_channels, *self.in_size]),
            requires_grad=True
        )

    @property
    def in_channels(self):
        return self.in_size[0]

    @property
    def in_height(self):
        return self.in_size[1]

    @property
    def in_width(self):
        return self.in_size[2]

    @property
    def out_size(self):
        return self.out_channels, self.in_height, self.in_width

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Apply the dropout, if specified
        if self.training and self.dropout is not None:
            x = x + torch.log(torch.floor(1.0 - self.dropout + torch.rand_like(x)))

        # Normalize the weight using log softmax
        w = torch.log_softmax(self.weight, dim=1)

        # Apply the logsumexp trick
        x = torch.unsqueeze(x, dim=1)
        return torch.logsumexp(x + w, dim=2)


class SpatialRootLayer(nn.Module):
    """Spatial Root layer class."""
    def __init__(self, in_size, out_channels):
        """
        Initialize a Spatial Root layer.

        :param in_size: The input tensor size. It should be (in_channels, in_height, in_width).
        :param out_channels: The number of output channels.
        """
        super(SpatialRootLayer, self).__init__()
        self.in_size = in_size
        self.out_channels = out_channels

        # Initialize the weight tensor
        in_flatten_size = np.prod(self.in_size).item()
        self.weight = nn.Parameter(
            torch.normal(0.0, 1e-1, [self.out_channels, in_flatten_size]),
            requires_grad=True
        )

    @property
    def in_channels(self):
        return self.in_size[0]

    @property
    def in_height(self):
        return self.in_size[1]

    @property
    def in_width(self):
        return self.in_size[2]

    @property
    def out_size(self):
        return self.out_channels,

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Calculate the log likelihood using the "logsumexp" trick
        x = torch.flatten(x, start_dim=1)
        x = torch.unsqueeze(x, dim=1)              # (-1, 1, in_flatten_size)
        w = torch.log_softmax(self.weight, dim=1)  # (out_channels, in_flatten_size)
        x = torch.logsumexp(x + w, dim=-1)         # (-1, out_channels)
        return x
