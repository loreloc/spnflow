import torch


class Dequantize:
    """Dequantize transformation"""
    def __init__(self):
        pass

    def __call__(self, x):
        return (x * 255.0 + torch.rand(x.size())) / 256.0


class Flatten:
    """Flatten transformation"""
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.flatten(x)


class Reshape:
    """Reshape transformation"""
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        return torch.reshape(x, self.size)
