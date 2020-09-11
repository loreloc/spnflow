import torch


class Logit:
    """Logit transformation"""
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def __call__(self, tensor):
        x = self.alpha + (1.0 - 2.0 * self.alpha) * tensor
        return torch.log(x / (1.0 - x))


class Delogit:
    """Logit inverse transformation"""
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def __call__(self, tensor):
        x = 1.0 / (1.0 + torch.exp(-tensor))
        return (x - self.alpha) / (1.0 - 2.0 * self.alpha)


class Dequantize:
    """Dequantize transformation"""
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor + torch.rand(tensor.size())


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


class Normalize:
    """Normalize transformation"""
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, tensor):
        return tensor / self.factor
