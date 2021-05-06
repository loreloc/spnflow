import torch


def get_activation_class(activation):
    """
    Get the activation function class by its name.

    :param activation: The activation function's name. It can be: 'relu', 'tanh', 'sigmoid'.
    :return: The activation function class.
    """
    dict_acts = {
        'relu': torch.nn.ReLU,
        'tanh': torch.nn.Tanh,
        'sigmoid': torch.nn.Sigmoid
    }
    return dict_acts[activation]


def squeeze_depth2d(x):
    """
    Squeeze operation (as in RealNVP).

    :param x: The input tensor of size [N, C, H, W].
    :return: The output tensor of size [N, C * 4, H // 2, W // 2].
    """
    n, c, h, w = x.size()
    x = x.reshape(n, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 3, 5, 1, 2, 4)
    x = x.reshape(n, c * 4, h // 2, w // 2)
    return x


def unsqueeze_depth2d(x):
    """
    Un-squeeze operation (as in RealNVP).

    :param x: The input tensor of size [N, C * 4, H // 2, W // 2].
    :return: The output tensor of size [N, C, H, W].
    """
    n, c, h, w = x.size()
    x = x.reshape(n, 2, 2, c // 4, h, w)
    x = x.permute(0, 3, 4, 1, 5, 2)
    x = x.reshape(n, c // 4, h * 2, w * 2)
    return x
