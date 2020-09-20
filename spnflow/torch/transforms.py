import torch


class Dequantize:
    """Dequantize transformation"""
    def __init__(self, factor=1.0):
        self.factor = factor

    def __call__(self, tensor):
        return tensor + torch.rand(tensor.size()) * self.factor


class Flatten:
    """Flatten transformation"""
    def __init__(self):
        pass

    def __call__(self, tensor):
        return torch.flatten(tensor)


class Reshape:
    """Reshape transformation"""
    def __init__(self, *size):
        self.size = size

    def __call__(self, tensor):
        return torch.reshape(tensor, self.size)
