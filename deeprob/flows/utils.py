import numpy as np
import torch
import torch.nn as nn


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


class BatchNormLayer(nn.Module):
    """Batch Normalization layer."""
    def __init__(self, in_features, momentum=0.9, epsilon=1e-5):
        """
        Build a Batch Normalization layer.

        :param in_features: The number of input features.
        :param momentum: The momentum used to update the running parameters.
        :param epsilon: An arbitrarily small value.
        """
        super(BatchNormLayer, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.momentum = momentum
        self.epsilon = epsilon

        # Initialize the learnable parameters (used for training)
        self.weight = nn.Parameter(torch.zeros(self.in_features), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(self.in_features), requires_grad=True)

        # Initialize the running parameters (used for inference)
        self.register_buffer('running_var', torch.ones(self.in_features))
        self.register_buffer('running_mean', torch.zeros(self.in_features))

    def inverse(self, x):
        """
        Evaluate the layer given some inputs (backward mode).

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        batch_size = x.shape[0]

        # Check if the module is training
        if self.training:
            # Get the mini batch statistics
            var, mean = torch.var_mean(x, dim=0)

            # Update the running parameters
            self.running_var.mul_(self.momentum).add_(var.data * (1.0 - self.momentum))
            self.running_mean.mul_(self.momentum).add_(mean.data * (1.0 - self.momentum))
        else:
            # Get the running parameters as batch mean and variance
            mean = self.running_mean
            var = self.running_var

        # Apply the transformation
        var = var + self.epsilon
        u = (x - mean) / torch.sqrt(var)
        u = u * torch.exp(self.weight) + self.bias
        inv_log_det_jacobian = torch.sum(self.weight - 0.5 * torch.log(var)).expand(batch_size, 1)
        return u, inv_log_det_jacobian

    def forward(self, u):
        """
        Evaluate the layer given some inputs (forward mode).

        :param u: The inputs.
        :return: The tensor result of the layer.
        """
        batch_size = u.shape[0]

        # Get the running parameters as batch mean and variance
        mean = self.running_mean
        var = self.running_var

        # Apply the transformation
        var = var + self.epsilon
        x = (u - self.bias) * torch.exp(-self.weight)
        x = x * torch.sqrt(var) + mean
        log_det_jacobian = torch.sum(-self.weight + 0.5 * torch.log(var)).expand(batch_size, 1)
        return x, log_det_jacobian


class DequantizeLayer(nn.Module):
    """Dequantization transformation layer."""
    def __init__(self, num_bits=8):
        """
        Build a Dequantization layer.

        :param num_bits: The number of bits to use.
        """
        super(DequantizeLayer, self).__init__()
        self.num_bits = num_bits
        self.quantization_bins = 2 ** self.num_bits
        self.register_buffer(
            'ildj_dim', torch.tensor(-np.log(self.quantization_bins), dtype=torch.float32)
        )

    def inverse(self, x):
        """
        Evaluate the layer given some inputs (backward mode).

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        batch_size = x.shape[0]
        num_dims = x.shape[1:].numel()
        u = x + torch.rand(x.shape, device=self.ildj_dim.device)
        u = u / self.quantization_bins
        ildj_dim = num_dims * self.ildj_dim
        inv_log_det_jacobian = ildj_dim.expand(batch_size, 1)
        return u, inv_log_det_jacobian

    def forward(self, u):
        """
        Evaluate the layer given some inputs (forward mode).

        :param u: The inputs.
        :return: The tensor result of the layer.
        """
        batch_size = u.shape[0]
        num_dims = u.shape[1:].numel()
        x = torch.floor(u * self.quantization_bins)
        x = torch.clamp(x, min=0, max=self.quantization_bins - 1).long()
        ldj_dim = -num_dims * self.ildj_dim
        log_det_jacobian = ldj_dim.expand(batch_size, 1)
        return x, log_det_jacobian


class LogitLayer(nn.Module):
    """Logit transformation layer."""
    def __init__(self, alpha=0.05):
        """
        Build a Logit layer.

        :param alpha: The alpha parameter for logit transformation.
        """
        super(LogitLayer, self).__init__()
        self.alpha = alpha
        self.register_buffer(
            'ildj_dim', torch.tensor(np.log(1.0 - 2.0 * self.alpha), dtype=torch.float32)
        )

    def inverse(self, x):
        """
        Evaluate the layer given some inputs (backward mode).

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        batch_size = x.shape[0]
        num_dims = x.shape[1:].numel()

        # Apply logit transformation
        x = self.alpha + (1.0 - 2.0 * self.alpha) * x
        lx = torch.log(x)
        rx = torch.log(1.0 - x)
        u = lx - rx
        v = lx + rx
        ildj_dim = num_dims * self.ildj_dim
        inv_log_det_jacobian = -torch.sum(v.view(batch_size, num_dims), dim=1, keepdim=True) + ildj_dim
        return u, inv_log_det_jacobian

    def forward(self, u):
        """
        Evaluate the layer given some inputs (forward mode).

        :param u: The inputs.
        :return: The tensor result of the layer.
        """
        batch_size = u.shape[0]
        num_dims = u.shape[1:].numel()

        # Apply de-logit transformation
        u = torch.sigmoid(u)
        x = (u - self.alpha) / (1.0 - 2.0 * self.alpha)
        lu = torch.log(u)
        ru = torch.log(-u)
        v = lu + ru
        ldj_dim = -num_dims * self.ildj_dim
        log_det_jacobian = torch.sum(v.view(batch_size, num_dims), dim=1, keepdim=True) + ldj_dim
        return x, log_det_jacobian
