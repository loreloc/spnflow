import torch
import numpy as np


class SpatialGaussianLayer(torch.nn.Module):
    """Spatial Gaussian input layer."""
    def __init__(self, in_size, out_channels, optimize_loc, optimize_scale):
        """
        Initialize a Spatial Gaussian input layer.

        :param in_size: The size of the input tensor.
        :param out_channels: The number of output channels.
        :param optimize_loc: Whether to optimize location.
        :param optimize_scale: Whether to optimize scale.
        """
        super(SpatialGaussianLayer, self).__init__()
        self.in_size = in_size
        self.out_channels = out_channels
        self.optimize_loc = optimize_loc
        self.optimize_scale = optimize_scale

        # Instantiate the location parameter
        if self.optimize_loc:
            self.loc = torch.nn.Parameter(torch.randn(self.out_channels, *self.in_size), requires_grad=True)
        else:
            self.loc = torch.nn.Parameter(torch.zeros(self.out_channels, *self.in_size), requires_grad=False)

        # Instantiate the scale parameter
        if self.optimize_scale:
            sigma = torch.sigmoid(torch.randn(self.out_channels, *self.in_size))
            self.scale = torch.nn.Parameter(0.25 + sigma * 0.5, requires_grad=True)
        else:
            self.scale = torch.nn.Parameter(torch.ones(self.out_channels, *self.in_size), requires_grad=False)

        # Instantiate the multi-batch normal distribution
        self.distribution = torch.distributions.Normal(self.loc, self.scale)

        # Initialize the marginalization constant
        self.register_buffer('zero', torch.zeros(1))

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
        # This implementation assumes independence between channels of the same pixel random variables
        # Also, marginalize random variables (denoted with NaNs)
        x = torch.unsqueeze(x, dim=1)
        x = self.distribution.log_prob(x)
        x = torch.where(torch.isnan(x), self.zero, x)
        return torch.sum(x, dim=2)


class SpatialProductLayer(torch.nn.Module):
    """Spatial Product layer class."""
    def __init__(self, in_size, out_channels, kernel_size, padding, stride, dilation, rand_state=None):
        """
        Initialize a Spatial Product layer.

        :param in_size: The input tensor size. It should be (in_channels, in_height, in_width).
        :param out_channels: The number of output channels. It can be None for depthwise convolutions.
        :param kernel_size: The size of the kernels.
        :param stride: The strides to use.
        :param padding: The padding mode to use. It can be 'valid', 'full' or 'final'.
        :param dilation: The space between the kernel points.
        :param rand_state: The random state used to generate the weight mask. It can be None if out_channels is None.
        """
        super(SpatialProductLayer, self).__init__()
        self.in_size = in_size
        self.depthwise = out_channels is None
        self.out_channels = self.in_channels if self.depthwise else out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.rand_state = rand_state
        self.groups = self.in_channels if self.depthwise else 1

        # Compute the effective kernel size (due to dilation)
        dh, dw = self.dilation
        kh, kw = self.kernel_size
        kah, kaw = (kh - 1) * dh + 1, (kw - 1) * dw + 1
        self.effective_kernel_size = (kah, kaw)

        # Compute the padding to apply
        if self.padding == 'valid':
            self.pad = (0, 0, 0, 0)
        elif self.padding == 'full':
            self.pad = (kaw - 1, kaw - 1, kah - 1, kah - 1)
        elif self.padding == 'final':
            self.pad = ((kaw - 1) * 2 - self.in_width, 0, (kah - 1) * 2 - self.in_height, 0)
        else:
            raise NotImplementedError('Unknown padding mode named ' + self.padding)

        # Build the convolution kernels
        if self.depthwise:
            weight = self._build_depthwise_kernels()
        else:
            weight = self._build_sparse_kernels(rand_state)

        # Initialize the weight tensor
        self.weight = torch.nn.Parameter(torch.tensor(weight), requires_grad=False)

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
        out_height = self.pad[2] + self.pad[3] + self.in_height - kah + 1
        out_width = self.pad[0] + self.pad[1] + self.in_width - kaw + 1
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
        x = torch.nn.functional.pad(x, self.pad)
        return torch.nn.functional.conv2d(
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
        return np.ones((self.out_channels, 1, kh, kw)).astype(np.float32)


class SpatialSumLayer(torch.nn.Module):
    """Spatial Sum layer class."""
    def __init__(self, in_size, out_channels):
        """
        Initialize a Spatial Sum layer.

        :param in_size: The input tensor size. It should be (in_channels, in_height, in_width).
        :param out_channels: The number of output channels.
        """
        super(SpatialSumLayer, self).__init__()
        self.in_size = in_size
        self.out_channels = out_channels

        # Initialize the weight tensor
        self.weight = torch.nn.Parameter(torch.randn(self.out_channels, self.in_channels, 1, 1), requires_grad=True)

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
        # Normalize the weight using softmax
        w = torch.softmax(self.weight, dim=1)

        # Subtract the max of the inputs (this is used for numerical stability)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_max = torch.detach(x_max)
        x = x - x_max

        # Apply plain convolution on exp(x) and return back to log space
        y = torch.nn.functional.conv2d(torch.exp(x), w)
        y = torch.log(y) + x_max
        return y


class SpatialRootLayer(torch.nn.Module):
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
        self.weight = torch.nn.Parameter(torch.randn(self.out_channels, in_flatten_size), requires_grad=True)

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
