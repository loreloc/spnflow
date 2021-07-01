import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def get_activation_class(activation):
    """
    Get the activation function class by its name.

    :param activation: The activation function's name.
                       It can be one of: 'relu', 'leaky-relu', 'softplus', 'tanh', 'sigmoid'.
    :return: The activation function class.
    """
    return {
        'relu': nn.ReLU,
        'leaky-relu': nn.LeakyReLU,
        'softplus': nn.Softplus,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
    }[activation]


def get_optimizer_class(optimizer):
    """
    Get the optimizer class by its name.

    :param optimizer: The optimizer's name. It can be 'sgd', 'rmsprop', 'adagrad', 'adam'.
    :return: The optimizer class.
    """
    return {
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
        'adagrad': optim.Adagrad,
        'adam': optim.Adam
    }[optimizer]


class ScaledTanh(nn.Module):
    """Scaled Tanh activation module."""
    def __init__(self, n_weights=None):
        """
        Build the module.

        :param n_weights: The number of weights. It can be None in order to get only one scale parameter.
        """
        super(ScaledTanh, self).__init__()
        if n_weights is None:
            n_weights = 1
        self.weight = nn.Parameter(torch.ones(n_weights), requires_grad=True)

    def forward(self, x):
        """
        Apply the scaled tanh function.

        :return: The result of the module.
        """
        return self.weight * torch.tanh(x)


class MaskedLinear(nn.Linear):
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
        return F.linear(x, self.mask * self.weight, self.bias)


class WeightNormConv2d(nn.Module):
    """Conv2D with weight normalization."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), bias=True):
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
        self.conv = nn.utils.weight_norm(nn.Conv2d(
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
