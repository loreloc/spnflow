import torch


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


def squeeze_depth2d(x):
    """
    Squeeze operation (as in RealNVP).

    :param x: The input tensor of size [N, C, H, W].
    :return: The output tensor of size [N, C * 4, H // 2, W // 2].
    """
    # This is literally 6D tensor black magic
    n, c, h, w = x.size()
    x = x.reshape(n, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.reshape(n, c * 4, h // 2, w // 2)
    return x


def unsqueeze_depth2d(x):
    """
    Un-squeeze operation (as in RealNVP).

    :param x: The input tensor of size [N, C * 4, H // 2, W // 2].
    :return: The output tensor of size [N, C, H, W].
    """
    # This is literally 6D tensor black magic
    n, c, h, w = x.size()
    x = x.reshape(n, c // 4, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3)
    x = x.reshape(n, c // 4, h * 2, w * 2)
    return x
