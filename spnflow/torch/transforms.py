import abc
import torch


class Transform(abc.ABC):
    """Generic data transform function."""
    @abc.abstractmethod
    def __call__(self, x):
        pass

    def inverse(self, x):
        pass


class TransformList(Transform, list):
    """A list of transformations."""
    def __call__(self, x):
        for transform in self:
            x = transform(x)
        return x

    def inverse(self, x):
        for transform in reversed(self):
            x = transform.inverse(x)
        return x


class Standardize(Transform):
    """Standardize transformation."""
    def __init__(self, mean, std, eps=1e-7):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def inverse(self, x):
        return x * (self.std + self.eps) + self.mean


class Quantize(Transform):
    """Quantize and Dequantize transformations."""
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.quantization_bins = 2 ** self.num_bits

    def __call__(self, x):
        x = torch.floor(x * self.quantization_bins)
        x = torch.clamp(x, min=0, max=self.quantization_bins - 1).long()
        return x

    def inverse(self, x):
        x = x * (self.quantization_bins - 1)
        x = (x + torch.rand(x.size())) / self.quantization_bins
        return x


class Flatten(Transform):
    """Flatten transformation."""
    def __init__(self, shape=None):
        self.shape = shape

    def __call__(self, x):
        return torch.flatten(x)

    def inverse(self, x):
        if self.shape is not None:
            return torch.reshape(x, self.shape)


class Reshape(Transform):
    """Reshape transformation."""
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        return torch.reshape(x, self.size)
