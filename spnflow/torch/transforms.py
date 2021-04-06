import torch


class Quantize:
    """Quantize transformation."""
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.quantization_bins = 2 ** self.num_bits

    def __call__(self, x):
        x = torch.floor(x * self.quantization_bins)
        x = torch.clamp(x, min=0, max=self.quantization_bins - 1).long()
        return x


class Dequantize:
    """Dequantize transformation."""
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.quantization_bins = 2 ** self.num_bits

    def __call__(self, x):
        return (x + torch.rand(x.size())) / self.quantization_bins


class Flatten:
    """Flatten transformation."""
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.flatten(x)


class Reshape:
    """Reshape transformation."""
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        return torch.reshape(x, self.size)
