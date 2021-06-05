import torch
import itertools


def get_activation_class(activation):
    """
    Get the activation function class by its name.

    :param activation: The activation function's name. It can be: 'relu', 'tanh', 'sigmoid'.
    :return: The activation function class.
    """
    return {
        'relu': torch.nn.ReLU,
        'tanh': torch.nn.Tanh,
        'sigmoid': torch.nn.Sigmoid
    }[activation]


def get_optimizer_class(optimizer):
    """
    Get the optimizer class by its name.

    :param optimizer: The optimizer's name. It can be 'sgd', 'rmsprop', 'adagrad', 'adam'.
    :return: The optimizer class.
    """
    return {
        'sgd': torch.optim.SGD,
        'rmsprop': torch.optim.RMSprop,
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam
    }[optimizer]


def squeeze_depth2d(x, interleave=False):
    """
    Squeeze operation (as in RealNVP).

    :param x: The input tensor of size [N, C, H, W].
    :param interleave: Whether to interleave the channels ordering.
    :return: The output tensor of size [N, C * 4, H // 2, W // 2].
    """
    n, c, h, w = x.size()
    x = x.reshape(n, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 3, 5, 1, 2, 4)
    x = x.reshape(n, c * 4, h // 2, w // 2)
    if interleave:
        shuffle = torch.tensor([k * c + i for k, i in itertools.product([0, 3, 1, 2], range(c))])
        x = x[:, shuffle]
    return x


def unsqueeze_depth2d(x, interleave=False):
    """
    Un-squeeze operation (as in RealNVP).

    :param x: The input tensor of size [N, C * 4, H // 2, W // 2].
    :param interleave: Whether to interleave the channels ordering.
    :return: The output tensor of size [N, C, H, W].
    """
    n, c, h, w = x.size()
    if interleave:
        unsq_c = c // 4
        shuffle = torch.tensor([k * unsq_c + i for k, i in itertools.product([0, 2, 3, 1], range(unsq_c))])
        x = x[:, shuffle]
    x = x.reshape(n, 2, 2, c // 4, h, w)
    x = x.permute(0, 3, 4, 1, 5, 2)
    x = x.reshape(n, c // 4, h * 2, w * 2)
    return x
