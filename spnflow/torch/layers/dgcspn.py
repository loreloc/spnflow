import torch
import numpy as np


class SpatialGaussianLayer(torch.nn.Module):
    """Spatial Gaussian input layer."""
    def __init__(self, in_size, out_channels, zeros_loc, optimize_scale):
        """
        Initialize a Spatial Gaussian input layer.

        :param in_size: The size of the input tensor.
        :param out_channels: The number of output channels.
        :param optimize_scale: Whether to optimize scale and location jointly.
        :param zeros_loc: Whether to initialize the location parameter with zeros.
        """
        super(SpatialGaussianLayer, self).__init__()
        self.in_size = in_size
        self.out_channels = out_channels
        self.zeros_loc = zeros_loc
        self.optimize_scale = optimize_scale

        # Instantiate the location variable
        params_size = (self.out_channels, *self.in_size)
        if self.zeros_loc:
            self.loc = torch.nn.Parameter(
                torch.zeros(size=params_size), requires_grad=True
            )
        else:
            self.loc = torch.nn.Parameter(
                torch.normal(0.0, 1e-1, size=params_size), requires_grad=True
            )

        # Instantiate the scale variable
        if self.optimize_scale:
            self.scale = torch.nn.Parameter(
                torch.normal(0.5, 5e-2, size=params_size), requires_grad=True
            )
        else:
            self.scale = torch.nn.Parameter(
                torch.ones(size=params_size), requires_grad=False
            )

        # Instantiate the multi-batch normal distribution
        self.distribution = torch.distributions.Normal(self.loc, self.scale)

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Compute the log-likelihoods (assume independence between channels of the same pixel random variables)
        x = torch.unsqueeze(x, dim=1)
        return torch.sum(self.distribution.log_prob(x), dim=2)

    @property
    def in_channels(self):
        return self.in_size[0]

    @property
    def in_height(self):
        return self.in_size[1]

    @property
    def in_width(self):
        return self.in_size[2]

    def output_size(self):
        """
        Get the output size of the layer.

        :return: The output size tuple.
        """
        return self.out_channels, self.in_height, self.in_width


class SpatialProductLayer(torch.nn.Module):
    """Spatial Product layer class."""
    def __init__(self, in_size, out_channels, kernel_size, padding, dilation, rand_state):
        """
        Initialize a Spatial Product layer.

        :param in_size: The input tensor size. It should be (in_channels, in_height, in_width).
        :param out_channels: The number of output channels.
        :param kernel_size: The size of the kernels.
        :param padding: The padding mode do use. It can be 'full' or 'final'.
        :param dilation: The space between the kernel points.
        :param rand_state: The random state used to generate the weight mask.
        """
        super(SpatialProductLayer, self).__init__()
        self.in_size = in_size
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation

        # Compute the effective kernel size (due to dilation)
        dh, dw = self.dilation
        kh, kw = self.kernel_size
        kah, kaw = (kh - 1) * dh + 1, (kw - 1) * dw + 1
        self.effective_kernel_size = (kah, kaw)

        # Compute the padding to apply
        if self.padding == 'full':
            self.pad = (kaw - 1, kaw - 1, kah - 1, kah - 1)
        elif self.padding == 'final':
            self.pad = ((kaw - 1) * 2 - self.in_width, 0, (kah - 1) * 2 - self.in_height, 0)
        else:
            raise NotImplementedError('Unknown padding mode named ' + self.padding)

        # Generate a sparse representation of weight
        kernels_idx = []
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
        weight = weight.astype(np.float32)

        # Initialize the weight tensor
        self.weight = torch.nn.Parameter(torch.tensor(weight), requires_grad=False)

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Pad the input
        x = torch.nn.functional.pad(x, self.pad)

        # Compute the log-likelihoods
        return torch.nn.functional.conv2d(x, self.weight, dilation=self.dilation)

    @property
    def in_channels(self):
        return self.in_size[0]

    @property
    def in_height(self):
        return self.in_size[1]

    @property
    def in_width(self):
        return self.in_size[2]

    def output_size(self):
        """
        Get the output size of the layer.

        :return: The output size tuple.
        """
        kah, kaw = self.effective_kernel_size
        out_height = self.pad[2] + self.pad[3] + self.in_height - kah + 1
        out_width = self.pad[0] + self.pad[1] + self.in_width - kaw + 1
        return self.out_channels, out_height, out_width


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
        self.weight = torch.nn.Parameter(
            torch.normal(0.0, 1e-1, size=(self.out_channels, self.in_channels, 1, 1)),
            requires_grad=True
        )

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Normalize the weight
        w = torch.log_softmax(self.weight, dim=1)

        # Subtract the max of the inputs (this is used for numerical stability during backward)
        w_max, _ = torch.max(w, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        w_max = torch.detach(w_max)
        x_max = torch.detach(x_max)
        w = w - w_max
        x = x - x_max

        # Apply plain convolution on exp(x) and return back to log space
        y = torch.nn.functional.conv2d(torch.exp(x), torch.exp(w))
        y = torch.log(y) + x_max + torch.transpose(w_max, 0, 1)
        return y

    @property
    def in_channels(self):
        return self.in_size[0]

    @property
    def in_height(self):
        return self.in_size[1]

    @property
    def in_width(self):
        return self.in_size[2]

    def output_size(self):
        """
        Get the output size of the layer.

        :return: The output size tuple.
        """
        return self.out_channels, self.in_height, self.in_width

    @staticmethod
    def _replace_infs_with_zero(x):
        return torch.where(torch.isinf(x), torch.zeros_like(x), x)


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
        self.weight = torch.nn.Parameter(
            torch.normal(0.0, 1e-1, size=(self.out_channels, in_flatten_size)),
            requires_grad=True
        )

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Flatten the input
        x = torch.flatten(x, start_dim=1)

        # Calculate the log likelihood using the "logsumexp" trick
        w = torch.log_softmax(self.weight, dim=1)  # (out_channels, in_flatten_size)
        x = torch.unsqueeze(x, dim=1)              # (-1, 1, in_flatten_size)
        x = torch.logsumexp(x + w, dim=-1)         # (-1, out_channels)
        return x

    @property
    def in_channels(self):
        return self.in_size[0]

    @property
    def in_height(self):
        return self.in_size[1]

    @property
    def in_width(self):
        return self.in_size[2]

    def output_size(self):
        """
        Get the output size of the layer.

        :return: The output size tuple.
        """
        return self.out_channels,
