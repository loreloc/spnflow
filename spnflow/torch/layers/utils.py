import numpy as np
import torch

from spnflow.torch.utils import unsqueeze_depth2d, squeeze_depth2d


class ScaledTanh(torch.nn.Module):
    """Scaled Tanh activation module."""
    def __init__(self, n_weights=None):
        """
        Build the module.

        :param n_weights: The number of weights. It can be None in order to get only one scale parameter.
        """
        super(ScaledTanh, self).__init__()
        if n_weights is None:
            n_weights = 1
        self.weight = torch.nn.Parameter(torch.ones(n_weights), requires_grad=True)

    def forward(self, x):
        """
        Apply the scaled tanh function.

        :return: The result of the module.
        """
        return self.weight * torch.tanh(x)


class MaskedLinear(torch.nn.Linear):
    """Masked version of linear layer."""
    def __init__(self, in_features, out_features, mask):
        """
        Build a masked linear layer.

        :param in_features: The number of input features.
        :param out_features: The number of output_features.
        :param mask: The mask to apply to the weights of the layer.
        """
        super(MaskedLinear, self).__init__(in_features, out_features)
        self.register_buffer('mask', torch.tensor(mask))

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        return torch.nn.functional.linear(x, self.mask * self.weight, self.bias)


class BatchNormLayer(torch.nn.Module):
    """Batch Normalization layer."""
    def __init__(self, in_features, momentum=0.1, epsilon=1e-5):
        """
        Build a Batch Normalization layer.

        :param in_features: The number of input features.
        :param momentum: The momentum used to update the running parameters.
        :param epsilon: An arbitrarily small value.
        """
        super(BatchNormLayer, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.momentum = momentum
        self.epsilon = epsilon

        # Initialize the learnable parameters (used for training)
        self.weight = torch.nn.Parameter(torch.zeros(self.in_features), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros(self.in_features), requires_grad=True)

        # Initialize the running parameters (used for inference)
        self.register_buffer('running_var', torch.ones(self.in_features))
        self.register_buffer('running_mean', torch.zeros(self.in_features))

    def inverse(self, x):
        """
        Evaluate the layer given some inputs (backward mode).

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Check if the module is training
        if self.training:
            # Get the minibatch statistics
            var, mean = torch.var_mean(x, dim=0)

            # Update the running parameters
            self.running_var = self.momentum * var + (1.0 - self.momentum) * self.running_var
            self.running_mean = self.momentum * mean + (1.0 - self.momentum) * self.running_mean
        else:
            # Get the running parameters as batch mean and variance
            mean = self.running_mean
            var = self.running_var

        # Apply the transformation
        var = var + self.epsilon
        u = (x - mean) / torch.sqrt(var)
        u = u * torch.exp(self.weight) + self.bias
        dj = torch.flatten(self.weight - 0.5 * torch.log(var))
        inv_log_det_jacobian = torch.sum(dj, dim=0, keepdim=True)
        return u, inv_log_det_jacobian

    def forward(self, u):
        """
        Evaluate the layer given some inputs (forward mode).

        :param u: The inputs.
        :return: The tensor result of the layer.
        """
        # Get the running parameters as batch mean and variance
        mean = self.running_mean
        var = self.running_var

        # Apply the transformation
        var = var + self.epsilon
        x = (u - self.bias) * torch.exp(-self.weight)
        x = x * torch.sqrt(var) + mean
        dj = torch.flatten(-self.weight + 0.5 * torch.log(var))
        log_det_jacobian = torch.sum(dj, dim=0, keepdim=True)
        return x, log_det_jacobian


class LogitLayer(torch.nn.Module):
    """Logit transformation layer."""
    def __init__(self, alpha=0.05):
        """
        Build a Logit layer.

        :param alpha: The alpha parameter for logit transformation.
        """
        super(LogitLayer, self).__init__()
        self.alpha = alpha

    def inverse(self, x):
        """
        Evaluate the layer given some inputs (backward mode).

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Apply logit transformation
        x = self.alpha + (1.0 - 2.0 * self.alpha) * x
        lx = torch.log(x)
        rx = torch.log(1.0 - x)
        u = lx - rx
        inv_log_det_jacobian = -torch.sum(
            torch.flatten(lx + rx, start_dim=1), dim=1, keepdim=True
        ) + np.prod(x.size()[1:]) * (np.log(1.0 - 2.0 * self.alpha) - np.log(256.0))
        return u, inv_log_det_jacobian

    def forward(self, u):
        """
        Evaluate the layer given some inputs (forward mode).

        :param u: The inputs.
        :return: The tensor result of the layer.
        """
        # Apply de-logit transformation
        u = torch.sigmoid(u)
        x = (u - self.alpha) / (1.0 - 2.0 * self.alpha)
        log_det_jacobian = torch.sum(
            torch.flatten(torch.log(u) + torch.log(-u), start_dim=1), dim=1, keepdim=True
        ) - np.prod(x.size()[1:]) * (np.log(1.0 - 2.0 * self.alpha) - np.log(256.0))
        return x, log_det_jacobian


class SqueezeLayer2d(torch.nn.Module):
    """Squeeze 2x2 operation as in RealNVP based on ResNets."""
    def __init__(self):
        """Initialize the layer."""
        super(SqueezeLayer2d, self).__init__()

    def forward(self, x):
        """
        Evaluate the layer given some inputs (depth-to-space transformation).

        :param x: The inputs of size [N, C * 4, H // 2, W // 2].
        :return: The outputs of size [N, C, H, W].
        """
        return unsqueeze_depth2d(x), 0.0

    def inverse(self, x):
        """
        Evaluate the layer given some inputs (space-to-depth transformation).

        :param x: The inputs of size [N, C, H, W].
        :return: The outputs of size [N, C * 4, H // 2, W // 2].
        """
        return squeeze_depth2d(x), 0.0


class UnsqueezeLayer2d(torch.nn.Module):
    """Unsqueeze 2x2 operation as in RealNVP based on ResNets."""
    def __init__(self):
        """Initialize the layer."""
        super(UnsqueezeLayer2d, self).__init__()

    def forward(self, x):
        """
        Evaluate the layer given some inputs (space-to-depth transformation).

        :param x: The inputs of size [N, C, H, W].
        :return: The outputs of size [N, C * 4, H // 2, W // 2].
        """
        return squeeze_depth2d(x), 0.0

    def inverse(self, x):
        """
        Evaluate the layer given some inputs (depth-to-space transformation).

        :param x: The inputs of size [N, C * 4, H // 2, W // 2].
        :return: The outputs of size [N, C, H, W].
        """
        return unsqueeze_depth2d(x), 0.0


class WeightNormConv2d(torch.nn.Module):
    """Conv2D with weight normalization."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        """
        Initialize a Conv2d layer with weight normalization.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param kernel_size: The convolving kernel size.
        :param stride: The stride of convolution.
        :param padding: The padding to apply.
        :param bias: Whether to use bias parameters.
        """
        super(WeightNormConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.conv = torch.nn.utils.weight_norm(torch.nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        ))

    def forward(self, x):
        """
        Evaluate the convolutional layer.

        :param x: The inputs.
        :return: The outputs of convolution.
        """
        return self.conv(x)
