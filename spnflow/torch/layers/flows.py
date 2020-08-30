import torch
import numpy as np


class MaskedLinear(torch.nn.Linear):
    """Masked version of linear layer."""
    def __init__(self, in_features, out_features, mask):
        """
        Build a masked linear layer.

        :param in_features: The number of input features.
        :param out_features: The number of output_features.
        :param mask: The mask to apply to the weights of the layer.
        """
        super(MaskedLinear, self).__init__(in_features, out_features)
        self.register_buffer('mask', torch.tensor(mask))

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        return torch.nn.functional.linear(x, self.mask * self.weight, self.bias)


class CouplingLayer(torch.nn.Module):
    """Real-NVP coupling layer."""
    def __init__(self, in_features, depth, units, activation, reverse=False):
        """
        Build a coupling layer as specified in Real-NVP paper.

        :param in_features: The number of input features.
        :param depth: The number of hidden layers of the conditioner.
        :param units: The number of units of each hidden layer of the conditioner.
        :param activation: The activation class used for inner layers of the conditioner.
        :param reverse: Whether to reverse the mask used in the coupling layer. Useful for alternating masks.
        """
        super(CouplingLayer, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.depth = depth
        self.units = units
        self.activation = activation
        self.reverse = reverse
        self.layers = torch.nn.ModuleList()

        # Build the conditioner neural network
        in_cond_features = self.in_features // 2
        if self.reverse and self.in_features % 2 != 0:
            in_cond_features += 1
        in_features = in_cond_features
        out_features = self.units
        for _ in range(depth):
            self.layers.append(torch.nn.Linear(in_features, out_features))
            self.layers.append(self.activation())
            in_features = out_features
        out_features = (self.in_features - in_cond_features) * 2
        self.layers.append(torch.nn.Linear(in_features, out_features))

    def inverse(self, x):
        """
        Evaluate the layer given some inputs (backward mode).

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Split the input
        if self.reverse:
            x1, x0 = torch.chunk(x, chunks=2, dim=1)
        else:
            x0, x1 = torch.chunk(x, chunks=2, dim=1)

        # Get the parameters and apply the affine transformation
        z = x1
        for layer in self.layers:
            z = layer(z)
        mu, sigma = torch.chunk(z, chunks=2, dim=1)
        x0 = (x0 - mu) * torch.exp(-sigma)
        inv_log_det_jacobian = -torch.sum(sigma, dim=1, keepdim=True)

        # Concatenate the data
        if self.reverse:
            u = torch.cat((x1, x0), dim=1)
        else:
            u = torch.cat((x0, x1), dim=1)
        return u, inv_log_det_jacobian

    def forward(self, u):
        """
        Evaluate the layer given some inputs (forward mode).

        :param u: The inputs.
        :return: The tensor result of the layer.
        """
        # Split the input
        if self.reverse:
            u1, u0 = torch.chunk(u, chunks=2, dim=1)
        else:
            u0, u1 = torch.chunk(u, chunks=2, dim=1)

        # Get the parameters and apply the affine transformation (inverse mode)
        z = u1
        for layer in self.layers:
            z = layer(z)
        mu, sigma = torch.chunk(z, chunks=2, dim=1)
        u0 = u0 * torch.exp(sigma) + mu
        log_det_jacobian = torch.sum(sigma, dim=1, keepdim=True)

        # Concatenate the data
        if self.reverse:
            x = torch.cat((u1, u0), dim=1)
        else:
            x = torch.cat((u0, u1), dim=1)
        return x, log_det_jacobian


class AutoregressiveLayer(torch.nn.Module):
    """Masked Autoregressive Flow autoregressive layer."""
    def __init__(self, in_features, depth, units, activation, reverse=False):
        """
        Build an autoregressive layer as specified in Masked Autoregressive Flow paper.

        :param in_features: The number of input features.
        :param depth: The number of hidden layers of the conditioner.
        :param units: The number of units of each hidden layer of the conditioner.
        :param activation: The activation class used for inner layers of the conditioner.
        :param reverse: Whether to reverse the mask used in the autoregressive layer. Useful for alternating orders.
        """
        super(AutoregressiveLayer, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.depth = depth
        self.units = units
        self.activation = activation
        self.reverse = reverse
        self.layers = torch.nn.ModuleList()

        # Create the masks of the masked linear layers
        masks = self._create_masks()

        # Initialize the conditioner neural network
        out_features = self.units
        for mask in masks[:-1]:
            self.layers.append(MaskedLinear(in_features, out_features, mask))
            self.layers.append(self.activation())
            in_features = out_features
        out_features = self.in_features * 2
        self.layers.append(MaskedLinear(in_features, out_features, np.tile(masks[-1], reps=(2, 1))))

    def inverse(self, x):
        """
        Evaluate the layer given some inputs (backward mode).

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Get the parameters and apply the affine transformation
        z = x
        for layer in self.layers:
            z = layer(z)
        mu, sigma = torch.chunk(z, chunks=2, dim=1)
        u = (x - mu) * torch.exp(-sigma)
        inv_log_det_jacobian = -torch.sum(sigma, dim=1, keepdim=True)
        return u, inv_log_det_jacobian

    def forward(self, u):
        """
        Evaluate the layer given some inputs (forward mode).

        :param u: The inputs.
        :return: The tensor result of the layer.
        """
        # Initialize z arbitrarily
        x = torch.zeros_like(u)
        log_det_jacobian = torch.zeros_like(u)

        # This requires K iterations where K is the number of features
        ordering = range(self.in_features)
        if self.reverse:
            ordering = reversed(ordering)

        # Get the parameters and apply the affine transformation (inverse mode)
        for i in ordering:
            z = x
            for layer in self.layers:
                z = layer(z)
            mu, sigma = torch.chunk(z, chunks=2, dim=1)
            x[:, i] = u[:, i] * torch.exp(sigma[:, i]) + mu[:, i]
            log_det_jacobian[:, i] = sigma[:, i]
        log_det_jacobian = torch.sum(log_det_jacobian, dim=1, keepdim=True)
        return x, log_det_jacobian

    def _create_masks(self):
        """
        Create the masks for the linear layers of the autoregressive network.

        :return: The masks to use for each hidden layer of the autoregressive network.
        """
        # Initialize the input degrees sequentially
        degrees = [np.arange(self.in_features)]
        if self.reverse:
            degrees[0] = np.flip(degrees[0])

        # Add the degrees of the hidden layers
        for _ in range(self.depth):
            degrees += [np.arange(self.units) % (self.in_features - 1)]

        # Add the degrees of the output layer
        degrees += [degrees[0] % self.in_features - 1]

        # Construct the masks
        masks = []
        for (d1, d2) in zip(degrees[:-1], degrees[1:]):
            d1 = np.expand_dims(d1, axis=0)
            d2 = np.expand_dims(d2, axis=1)
            masks += [np.greater_equal(d2, d1).astype('float32')]
        return masks


class BatchNormLayer(torch.nn.Module):
    """Batch Normalization layer."""
    def __init__(self, in_features, momentum=0.1, epsilon=1e-5):
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
        self.weight = torch.nn.Parameter(torch.zeros(self.in_features), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros(self.in_features), requires_grad=True)

        # Initialize the running parameters (used for inference)
        self.register_buffer('running_var', torch.ones(self.in_features))
        self.register_buffer('running_mean', torch.zeros(self.in_features))

    def inverse(self, x):
        """
        Evaluate the layer given some inputs (backward mode).

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Check if the module is training
        if self.training:
            # Get the minibatch statistics
            var, mean = torch.var_mean(x, dim=0)

            # Update the running parameters
            self.running_var = self.momentum * var + (1.0 - self.momentum) * self.running_var
            self.running_mean = self.momentum * mean + (1.0 - self.momentum) * self.running_mean
        else:
            # Get the running parameters as batch mean and variance
            mean = self.running_mean
            var = self.running_var

        # Apply the transformation
        var = var + self.epsilon
        u = (x - mean) / torch.sqrt(var)
        u = u * torch.exp(self.weight) + self.bias
        inv_log_det_jacobian = torch.sum(self.weight - 0.5 * torch.log(var), dim=0, keepdim=True)

        return u, inv_log_det_jacobian

    def forward(self, u):
        """
        Evaluate the layer given some inputs (forward mode).

        :param u: The inputs.
        :return: The tensor result of the layer.
        """
        # Get the running parameters as batch mean and variance
        mean = self.running_mean
        var = self.running_var

        # Apply the transformation
        var = var + self.epsilon
        x = (u - self.bias) * torch.exp(-self.weight)
        x = x * torch.sqrt(var) + mean
        log_det_jacobian = torch.sum(-self.weight + 0.5 * torch.log(var), dim=0, keepdim=True)

        return x, log_det_jacobian
