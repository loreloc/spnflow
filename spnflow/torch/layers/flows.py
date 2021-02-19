import abc
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


class AbstractCouplingLayer(abc.ABC, torch.nn.Module):
    """Abstract RealNVP coupling layer."""
    def __init__(self, in_features, depth, activation, reverse):
        super(AbstractCouplingLayer, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.depth = depth
        self.activation = activation
        self.reverse = reverse
        self.layers = torch.nn.ModuleList()

    def register_mask(self, mask):
        """
        Register the coupling mask.

        :param mask: The coupling mask numpy array to register.
        """
        inv_mask = 1.0 - mask
        if self.reverse:
            self.register_buffer('mask', torch.tensor(inv_mask))
            self.register_buffer('inv_mask', torch.tensor(mask))
        else:
            self.register_buffer('mask', torch.tensor(mask))
            self.register_buffer('inv_mask', torch.tensor(inv_mask))

    def inverse(self, x):
        """
        Evaluate the layer given some inputs (backward mode).

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Get the parameters and apply the affine transformation
        mx = self.mask * x
        y = mx
        for layer in self.layers:
            y = layer(y)
        t, s = torch.chunk(y, chunks=2, dim=1)
        u = mx + self.inv_mask * ((x - t) * torch.exp(-s))
        dj = torch.flatten(self.inv_mask * s, start_dim=1)
        inv_log_det_jacobian = -torch.sum(dj, dim=1, keepdim=True)
        return u, inv_log_det_jacobian

    def forward(self, u):
        """
        Evaluate the layer given some inputs (forward mode).

        :param u: The inputs.
        :return: The tensor result of the layer.
        """
        # Get the parameters and apply the affine transformation (inverse mode)
        mu = self.mask * u
        y = mu
        for layer in self.layers:
            y = layer(y)
        t, s = torch.chunk(y, chunks=2, dim=1)
        x = mu + self.inv_mask * (u * torch.exp(s) + t)
        dj = torch.flatten(self.inv_mask * s, start_dim=1)
        log_det_jacobian = torch.sum(dj, dim=1, keepdim=True)
        return x, log_det_jacobian


class CouplingLayer1d(AbstractCouplingLayer):
    """RealNVP 1D coupling layer."""
    def __init__(self, in_features, depth, activation, reverse, units):
        """
        Build a coupling layer as specified in RealNVP paper.

        :param in_features: The number of input features.
        :param depth: The number of hidden layers of the conditioner.
        :param units: The number of units of each hidden layer of the conditioner.
        :param activation: The activation class used for inner layers of the conditioner.
        :param reverse: Whether to reverse the mask used in the coupling layer. Useful for alternating masks.
        """
        super(CouplingLayer1d, self).__init__(in_features, depth, activation, reverse)
        self.units = units

        # Register the coupling mask
        self.register_mask(self.build_mask_alternating(self.in_features))

        # Build the conditioner neural network
        in_features = self.in_features
        out_features = self.units
        for _ in range(self.depth):
            self.layers.append(torch.nn.Linear(in_features, out_features))
            self.layers.append(self.activation())
            in_features = out_features
        out_features = self.in_features * 2
        self.layers.append(torch.nn.Linear(in_features, out_features))

    @staticmethod
    def build_mask_alternating(in_features):
        # Build the alternating coupling mask
        mask = np.arange(in_features) % 2
        return mask.astype(np.float32)


class CouplingLayer2d(AbstractCouplingLayer):
    """RealNVP 2D coupling layer."""
    def __init__(self, in_features, depth, activation, reverse, channels, kernel_size):
        """
        Build a 2D-convolutional coupling layer as specified in RealNVP paper.

        :param in_features: The size of the input.
        :param depth: The number of hidden layers of the conditioner.
        :param activation: The activation class used for inner layers of the conditioner.
        :param reverse: Whether to reverse the mask used in the coupling layer. Useful for alternating masks.
        :param channels: The number of output channels of each convolutional layer.
        :param kernel_size: The kernel size of each convolutional layer.
        """
        super(CouplingLayer2d, self).__init__(in_features, depth, activation, reverse)
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Register the coupling mask
        self.register_mask(self.build_mask_chessboard(self.in_features))

        # Build the conditioner convolutional neural network
        in_channels = self.in_channels
        for _ in range(self.depth):
            self.layers.append(torch.nn.Conv2d(in_channels, channels, self.kernel_size, padding=self.padding))
            self.layers.append(self.activation())
            in_channels = channels
        channels = self.in_channels * 2
        self.layers.append(torch.nn.Conv2d(in_channels, channels, self.kernel_size, padding=self.padding))

    @property
    def in_channels(self):
        return self.in_features[0]

    @property
    def in_height(self):
        return self.in_features[1]

    @property
    def in_width(self):
        return self.in_features[2]

    @staticmethod
    def build_mask_chessboard(in_features):
        # Build the chessboard coupling mask
        in_channels, in_width, in_height = in_features
        mask = np.sum(np.indices([1, in_width, in_height]), axis=0) % 2
        return mask.astype(np.float32)


class AutoregressiveLayer(torch.nn.Module):
    """Masked Autoregressive Flow autoregressive layer."""
    def __init__(self, in_features, depth, units, activation, sequential=True, reverse=False, rand_state=None):
        """
        Build an autoregressive layer as specified in Masked Autoregressive Flow paper.

        :param in_features: The number of input features.
        :param depth: The number of hidden layers of the conditioner.
        :param units: The number of units of each hidden layer of the conditioner.
        :param activation: The activation class used for inner layers of the conditioner.
        :param sequential: Whether to use sequential degrees for inner layers masks.
        :param reverse: Whether to reverse the mask used in the autoregressive layer. Used only if sequential is True.
        :param rand_state: The random state used to generate the masks degrees. Used only if sequential is False.
        """
        super(AutoregressiveLayer, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.depth = depth
        self.units = units
        self.activation = activation
        self.sequential = sequential
        self.reverse = reverse
        self.rand_state = rand_state
        self.layers = torch.nn.ModuleList()

        # Create the masks of the masked linear layers
        degrees = self._build_degrees_sequential() if sequential else self._build_degrees_random()
        masks = self._build_masks(degrees)

        # Preserve the input ordering
        self.ordering = degrees[0]

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

        # This requires D iterations where D is the number of features
        # Get the parameters and apply the affine transformation (inverse mode)
        for i in range(0, self.in_features):
            z = x
            for layer in self.layers:
                z = layer(z)
            mu, sigma = torch.chunk(z, chunks=2, dim=1)
            idx = np.argwhere(self.ordering == i).item()
            x[:, idx] = u[:, idx] * torch.exp(sigma[:, idx]) + mu[:, idx]
            log_det_jacobian[:, idx] = sigma[:, idx]
        log_det_jacobian = torch.sum(log_det_jacobian, dim=1, keepdim=True)
        return x, log_det_jacobian

    def _build_degrees_sequential(self):
        """
        Build sequential degrees for the linear layers of the autoregressive network.

        :return: The masks to use for each hidden layer of the autoregressive network.
        """
        # Initialize the input degrees sequentially
        degrees = []
        if self.reverse:
            degrees.append(np.arange(self.in_features - 1, -1, -1))
        else:
            degrees.append(np.arange(self.in_features))

        # Add the degrees of the hidden layers
        for _ in range(self.depth):
            degrees.append(np.arange(self.units) % (self.in_features - 1))
        return degrees

    def _build_degrees_random(self):
        """
        Create random degrees for the linear layers of the autoregressive network.

        :return: The masks to use for each hidden layer of the autoregressive network.
        """
        # Initialize the input degrees randomly
        degrees = []
        ordering = np.arange(self.in_features)
        self.rand_state.shuffle(ordering)
        degrees.append(ordering)

        # Add the degrees of the hidden layers
        for _ in range(self.depth):
            min_prev_degree = np.min(degrees[-1])
            degrees.append(self.rand_state.randint(min_prev_degree, self.in_features - 1, self.units))
        return degrees

    @staticmethod
    def _build_masks(degrees):
        """
        Build masks from degrees.

        :return: The masks to use for each hidden layer of the autoregressive network.
        """
        masks = []
        for (d1, d2) in zip(degrees[:-1], degrees[1:]):
            d1 = np.expand_dims(d1, axis=0)
            d2 = np.expand_dims(d2, axis=1)
            masks.append(np.less_equal(d1, d2).astype(np.float32))
        d1 = np.expand_dims(degrees[-1], axis=0)
        d2 = np.expand_dims(degrees[0], axis=1)
        masks.append(np.less(d1, d2).astype(np.float32))
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
        dj = torch.flatten(self.weight - 0.5 * torch.log(var))
        inv_log_det_jacobian = torch.sum(dj, dim=0, keepdim=True)
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
        dj = torch.flatten(-self.weight + 0.5 * torch.log(var))
        log_det_jacobian = torch.sum(dj, dim=0, keepdim=True)
        return x, log_det_jacobian


class LogitLayer(torch.nn.Module):
    """Logit transformation layer."""
    def __init__(self, alpha=0.05):
        """
        Build a Logit layer.

        :param alpha: The alpha parameter for logit transformation.
        """
        super(LogitLayer, self).__init__()
        self.alpha = alpha

    def inverse(self, x):
        """
        Evaluate the layer given some inputs (backward mode).

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Apply logit transformation
        x = self.alpha + (1.0 - 2.0 * self.alpha) * x
        u = torch.log(x) - torch.log(1.0 - x)
        dj = torch.flatten(
            torch.log(torch.sigmoid(u)) + torch.log(torch.sigmoid(-u)) - np.log(1.0 - 2.0 * self.alpha),
            start_dim=1
        )
        inv_log_det_jacobian = -torch.sum(dj, dim=1, keepdim=True)
        return u, inv_log_det_jacobian

    def forward(self, u):
        """
        Evaluate the layer given some inputs (forward mode).

        :param u: The inputs.
        :return: The tensor result of the layer.
        """
        # Apply de-logit transformation
        u = torch.sigmoid(u)
        x = (u - self.alpha) / (1.0 - 2.0 * self.alpha)
        dj = torch.flatten(
            torch.log(u) + torch.log(1.0 - u) + np.log(1.0 - 2.0 * self.alpha),
            start_dim=1
        )
        log_det_jacobian = -torch.sum(dj, dim=1, keepdim=True)
        return x, log_det_jacobian
