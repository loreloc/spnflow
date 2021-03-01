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
    Space to depth transformation utility for tensors.

    :param x: The input tensor of size [N, C, H, W].
    :return: The output tensor of size [N, C * 4, H // 2, W // 2].
    """
    n, c, h, w = x.size()
    unfolded = torch.nn.functional.unfold(x, kernel_size=2, stride=2)
    return unfolded.view(n, c * 4, h // 2, w // 2)


def unsqueeze_depth2d(x):
    """
    Depth to space transformation utility for tensors.

    :param x: The input tensor of size [N, C * 4, H // 2, W // 2].
    :return: The output tensor of size [N, C, H, W].
    """
    return torch.nn.functional.pixel_shuffle(x, upscale_factor=2)
