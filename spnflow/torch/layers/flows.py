import torch
import numpy as np

from spnflow.torch.utils import unsqueeze_depth2d, squeeze_depth2d
from spnflow.torch.layers.resnet import ResidualNetwork
from spnflow.torch.layers.densenet import DenseNetwork
from spnflow.torch.layers.utils import ScaledTanh, MaskedLinear, BatchNormLayer


class CouplingLayer1d(torch.nn.Module):
    """RealNVP 1D coupling layer."""
    def __init__(self, in_features, depth, units, reverse):
        """
        Build a coupling layer as specified in RealNVP paper.

        :param in_features: The number of input features.
        :param depth: The number of hidden layers of the conditioner.
        :param units: The number of units of each hidden layer of the conditioner.
        :param reverse: Whether to reverse the mask used in the coupling layer. Useful for alternating masks.
        """
        super(CouplingLayer1d, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.depth = depth
        self.units = units
        self.reverse = reverse
        self.layers = torch.nn.ModuleList()
        self.scale_activation = ScaledTanh()

        # Register the coupling mask
        mask = self.build_mask_alternating(self.in_features)
        inv_mask = 1.0 - mask
        if self.reverse:
            self.register_buffer('mask', torch.tensor(inv_mask))
            self.register_buffer('inv_mask', torch.tensor(mask))
        else:
            self.register_buffer('mask', torch.tensor(mask))
            self.register_buffer('inv_mask', torch.tensor(inv_mask))

        # Build the conditioner neural network
        in_features = self.in_features
        out_features = self.units
        for _ in range(self.depth):
            self.layers.append(torch.nn.Linear(in_features, out_features))
            self.layers.append(torch.nn.ReLU(inplace=True))
            in_features = out_features
        out_features = self.in_features * 2
        self.layers.append(torch.nn.Linear(in_features, out_features))

    @staticmethod
    def build_mask_alternating(in_features):
        # Build the alternating coupling mask
        mask = np.arange(in_features) % 2
        return mask.astype(np.float32)

    def inverse(self, x):
        """
        Evaluate the layer given some inputs (backward mode).

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Get the parameters
        z = self.mask * x
        for layer in self.layers:
            z = layer(z)
        t, s = torch.chunk(z, chunks=2, dim=1)
        s = self.scale_activation(s)
        t = self.inv_mask * t
        s = self.inv_mask * s

        # Apply the affine transformation (backward mode)
        u = (x - t) * torch.exp(-s)
        inv_log_det_jacobian = -torch.sum(s, dim=1, keepdim=True)
        return u, inv_log_det_jacobian

    def forward(self, u):
        """
        Evaluate the layer given some inputs (forward mode).

        :param u: The inputs.
        :return: The tensor result of the layer.
        """
        # Get the parameters
        z = self.mask * u
        for layer in self.layers:
            z = layer(z)
        t, s = torch.chunk(z, chunks=2, dim=1)
        s = self.scale_activation(s)
        t = self.inv_mask * t
        s = self.inv_mask * s

        # Apply the affine transformation (forward mode).
        x = u * torch.exp(s) + t
        log_det_jacobian = torch.sum(s, dim=1, keepdim=True)
        return x, log_det_jacobian


class CouplingLayer2d(torch.nn.Module):
    """RealNVP 2D coupling layer."""
    def __init__(self, in_features, network, n_blocks, channels, reverse, channel_wise=False):
        """
        Build a ResNet-based coupling layer as specified in RealNVP paper.

        :param in_features: The size of the input.
        :param network: The network conditioner to use. It can be either 'resnet' or 'densenet'.
        :param n_blocks: The number of residual blocks or dense blocks.
        :param channels: The number of output channels of each convolutional layer.
        :param reverse: Whether to reverse the mask used in the coupling layer. Useful for alternating masks.
        :param channel_wise: Whether to use channel_wise coupling mask.
                             Defaults to False, i.e. chessboard coupling mask.
        """
        super(CouplingLayer2d, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.n_blocks = n_blocks
        self.channels = channels
        self.reverse = reverse
        self.channel_wise = channel_wise

        # Register the chessboard coupling mask, if specified
        if not self.channel_wise:
            mask = self.build_mask_chessboard(self.in_features)
            inv_mask = 1.0 - mask
            if self.reverse:
                self.register_buffer('mask', torch.tensor(inv_mask))
                self.register_buffer('inv_mask', torch.tensor(mask))
            else:
                self.register_buffer('mask', torch.tensor(mask))
                self.register_buffer('inv_mask', torch.tensor(inv_mask))

        # Build the conditioner neural network
        in_channels = self.in_channels
        if self.channel_wise:
            in_channels //= 2
        if network == 'resnet':
            self.network = ResidualNetwork(in_channels, self.channels, in_channels * 2, self.n_blocks)
        elif network == 'densenet':
            self.network = DenseNetwork(in_channels, self.channels, in_channels * 2, self.n_blocks)
        else:
            raise NotImplementedError('Unknown network conditioner {}'.format(network))

        # Build the activation function for the scale of affine transformation
        self.scale_activation = ScaledTanh([in_channels, 1, 1])

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
        in_channels, in_height, in_width = in_features
        mask = np.sum(np.indices([1, in_height, in_width]), axis=0) % 2
        return mask.astype(np.float32)

    def inverse(self, x):
        """
        Evaluate the layer given some inputs (backward mode).

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        batch_size = x.shape[0]

        if self.channel_wise:
            # Get the parameters
            if self.reverse:
                mx, my = torch.chunk(x, chunks=2, dim=1)
            else:
                my, mx = torch.chunk(x, chunks=2, dim=1)
            ts = self.network(mx)
            t, s = torch.chunk(ts, chunks=2, dim=1)
            s = self.scale_activation(s)

            # Apply the affine transformation (backward mode)
            my = (my - t) * torch.exp(-s)
            if self.reverse:
                u = torch.cat([mx, my], dim=1)
            else:
                u = torch.cat([my, mx], dim=1)
        else:
            # Get the parameters
            mx = self.mask * x
            ts = self.network(mx)
            t, s = torch.chunk(ts, chunks=2, dim=1)
            s = self.scale_activation(s)
            t = self.inv_mask * t
            s = self.inv_mask * s

            # Apply the affine transformation (backward mode)
            u = (x - t) * torch.exp(-s)
        inv_log_det_jacobian = -torch.sum(s.view(batch_size, -1), dim=1, keepdim=True)
        return u, inv_log_det_jacobian

    def forward(self, u):
        """
        Evaluate the layer given some inputs (forward mode).

        :param u: The inputs.
        :return: The tensor result of the layer.
        """
        batch_size = u.shape[0]

        if self.channel_wise:
            # Get the parameters
            if self.reverse:
                mu, mv = torch.chunk(u, chunks=2, dim=1)
            else:
                mv, mu = torch.chunk(u, chunks=2, dim=1)
            ts = self.network(mu)
            t, s = torch.chunk(ts, chunks=2, dim=1)
            s = self.scale_activation(s)

            # Apply the affine transformation (backward mode)
            mv = mv * torch.exp(s) + t
            if self.reverse:
                x = torch.cat([mu, mv], dim=1)
            else:
                x = torch.cat([mv, mu], dim=1)
        else:
            # Get the parameters
            mu = self.mask * u
            ts = self.network(mu)
            t, s = torch.chunk(ts, chunks=2, dim=1)
            s = self.scale_activation(s)
            t = self.inv_mask * t
            s = self.inv_mask * s

            # Apply the affine transformation (backward mode)
            x = u * torch.exp(s) + t
        log_det_jacobian = torch.sum(s.view(batch_size, -1), dim=1, keepdim=True)
        return x, log_det_jacobian


class CouplingBlock2d(torch.nn.Module):
    """RealNVP 2D coupling block, consisting of checkboard/channelwise couplings and squeeze operation."""
    def __init__(self, in_features, network, n_blocks, channels, last_block=False):
        """
        Build a RealNVP 2d coupling block.

        :param in_features: The size of the input.
        :param network: The network conditioner to use. It can be either 'resnet' or 'densenet'.
        :param n_blocks: The number of residual blocks or dense blocks.
        :param channels: The number of output channels of each convolutional layer.
        :param last_block: Whether it is the last block (i.e. no channelwise-masked couplings) or not.
        """
        super(CouplingBlock2d, self).__init__()
        self.in_features = in_features
        self.network = network
        self.n_blocks = n_blocks
        self.channels = channels
        self.last_block = last_block

        if self.last_block:
            # Get the size of the output (no squeeze operation if last_block=True)
            self.out_features = self.in_features

            # Build the input couplings (consisting of 4 checkboard-masked couplings)
            self.in_couplings = torch.nn.ModuleList([
                CouplingLayer2d(
                    self.in_features, self.network, self.n_blocks, self.channels,
                    reverse=False, channel_wise=False
                ),
                BatchNormLayer(self.in_features),
                CouplingLayer2d(
                    self.in_features, self.network, self.n_blocks, self.channels,
                    reverse=True, channel_wise=False
                ),
                BatchNormLayer(self.in_features),
                CouplingLayer2d(
                    self.in_features, self.network, self.n_blocks, self.channels,
                    reverse=False, channel_wise=False
                ),
                BatchNormLayer(self.in_features),
                CouplingLayer2d(
                    self.in_features, self.network, self.n_blocks, self.channels,
                    reverse=True, channel_wise=False
                ),
                BatchNormLayer(self.in_features)
            ])
        else:
            # Compute the size of the output (after squeezing operation)
            self.out_features = (self.in_channels * 4, self.in_height // 2, self.in_width // 2)

            # Build the input couplings (consisting of 3 checkboard-masked couplings)
            self.in_couplings = torch.nn.ModuleList([
                CouplingLayer2d(
                    self.in_features, self.network, self.n_blocks, self.channels,
                    reverse=False, channel_wise=False
                ),
                BatchNormLayer(self.in_features),
                CouplingLayer2d(
                    self.in_features, self.network, self.n_blocks, self.channels,
                    reverse=True, channel_wise=False
                ),
                BatchNormLayer(self.in_features),
                CouplingLayer2d(
                    self.in_features, self.network, self.n_blocks, self.channels,
                    reverse=False, channel_wise=False
                ),
                BatchNormLayer(self.in_features)
            ])

            # Build the output couplings (consisting of 3 channelwise-masked couplings)
            self.out_couplings = torch.nn.ModuleList([
                CouplingLayer2d(
                    self.out_features, self.network, self.n_blocks, self.channels,
                    reverse=False, channel_wise=True
                ),
                BatchNormLayer(self.out_features),
                CouplingLayer2d(
                    self.out_features, self.network, self.n_blocks, self.channels,
                    reverse=True, channel_wise=True
                ),
                BatchNormLayer(self.out_features),
                CouplingLayer2d(
                    self.out_features, self.network, self.n_blocks, self.channels,
                    reverse=False, channel_wise=True
                ),
                BatchNormLayer(self.out_features)
            ])

    @property
    def in_channels(self):
        return self.in_features[0]

    @property
    def in_height(self):
        return self.in_features[1]

    @property
    def in_width(self):
        return self.in_features[2]

    def inverse(self, x):
        """
        Evaluate the coupling block (backward mode).

        :param x: The inputs.
        :return: The outputs of checkboard-masked only couplings if last_block=True
                 and the outputs of checkboard-masked + squeeze + channelwise-masked couplings if last_block=False.
        """
        inv_log_det_jacobian = 0.0

        # Pass through the checkboard-masked couplings
        for layer in self.in_couplings:
            x, ildj = layer.inverse(x)
            inv_log_det_jacobian += ildj

        if not self.last_block:
            # Squeeze the inputs
            x = squeeze_depth2d(x)

            # Pass through the channelwise-masked couplings
            for layer in self.out_couplings:
                x, ildj = layer.inverse(x)
                inv_log_det_jacobian += ildj

        return x, inv_log_det_jacobian

    def forward(self, x):
        """
        Evaluate the coupling block (forward mode).

        :param x: The inputs.
        :return: The outputs of checkboard-masked only couplings if last_block=True
                 and the outputs of checkboard-masked + squeeze + channelwise-masked couplings if last_block=False.
        """
        log_det_jacobian = 0.0

        if not self.last_block:
            # Pass through the channelwise-masked couplings
            for layer in reversed(self.out_couplings):
                x, ldj = layer.forward(x)
                log_det_jacobian += ldj

            # Un-squeeze the inputs
            x = unsqueeze_depth2d(x)

        # Pass through the checkboard-masked couplings
        for layer in reversed(self.in_couplings):
            x, ldj = layer.forward(x)
            log_det_jacobian += ldj

        return x, log_det_jacobian


class AutoregressiveLayer(torch.nn.Module):
    """Masked Autoregressive Flow autoregressive layer."""
    def __init__(self, in_features, depth, units, activation, reverse, sequential=True, rand_state=None):
        """
        Build an autoregressive layer as specified in Masked Autoregressive Flow paper.

        :param in_features: The number of input features.
        :param depth: The number of hidden layers of the conditioner.
        :param units: The number of units of each hidden layer of the conditioner.
        :param activation: The activation class used for inner layers of the conditioner.
        :param reverse: Whether to reverse the mask used in the autoregressive layer. Used only if sequential is True.
        :param sequential: Whether to use sequential degrees for inner layers masks.
        :param rand_state: The random state used to generate the masks degrees. Used only if sequential is False.
        """
        super(AutoregressiveLayer, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.depth = depth
        self.units = units
        self.activation = activation
        self.reverse = reverse
        self.sequential = sequential
        self.rand_state = rand_state
        self.layers = torch.nn.ModuleList()
        self.scale_activation = ScaledTanh()

        # Create the masks of the masked linear layers
        degrees = self._build_degrees_sequential() if sequential else self._build_degrees_random()
        masks = self._build_masks(degrees)

        # Preserve the input ordering
        self.ordering = degrees[0]

        # Initialize the conditioner neural network
        out_features = self.units
        for mask in masks[:-1]:
            self.layers.append(MaskedLinear(in_features, out_features, mask))
            self.layers.append(self.activation(inplace=True))
            in_features = out_features
        out_features = self.in_features * 2
        self.layers.append(MaskedLinear(in_features, out_features, np.tile(masks[-1], reps=(2, 1))))

    def inverse(self, x):
        """
        Evaluate the layer given some inputs (backward mode).

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Get the parameters and apply the affine transformation (backward mode)
        z = x
        for layer in self.layers:
            z = layer(z)
        t, s = torch.chunk(z, chunks=2, dim=1)
        s = self.scale_activation(s)
        u = (x - t) * torch.exp(-s)
        inv_log_det_jacobian = -torch.sum(s, dim=1, keepdim=True)
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
        # Get the parameters and apply the affine transformation (forward mode)
        for i in range(0, self.in_features):
            z = x
            for layer in self.layers:
                z = layer(z)
            t, s = torch.chunk(z, chunks=2, dim=1)
            s = self.scale_activation(s)
            idx = np.argwhere(self.ordering == i).item()
            x[:, idx] = u[:, idx] * torch.exp(s[:, idx]) + t[:, idx]
            log_det_jacobian[:, idx] = s[:, idx]
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
