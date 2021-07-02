import numpy as np
import torch
import torch.nn as nn

from deeprob.torch.utils import ScaledTanh
from deeprob.flows.utils import BatchNormLayer
from deeprob.flows.utils import squeeze_depth2d, unsqueeze_depth2d
from deeprob.flows.layers.densenet import DenseNetwork
from deeprob.flows.layers.resnet import ResidualNetwork


class CouplingLayer1d(nn.Module):
    """RealNVP 1D coupling layer."""
    def __init__(self, in_features, depth, units, affine=True, reverse=False):
        """
        Build a coupling layer as specified in RealNVP paper.

        :param in_features: The number of input features.
        :param depth: The number of hidden layers of the conditioner.
        :param units: The number of units of each hidden layer of the conditioner.
        :param reverse: Whether to reverse the mask used in the coupling layer. Useful for alternating masks.
        :param affine: Whether to use affine transformation. If False then use only translation (as in NICE).
        """
        super(CouplingLayer1d, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.depth = depth
        self.units = units
        self.affine = affine
        self.reverse = reverse
        self.layers = nn.ModuleList()
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
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(nn.ReLU(inplace=True))
            in_features = out_features
        out_features = self.in_features * 2 if self.affine else self.in_features
        self.layers.append(nn.Linear(in_features, out_features))

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

        if self.affine:
            # Apply the affine transformation (backward mode)
            t, s = torch.chunk(z, chunks=2, dim=1)
            s = self.scale_activation(s)
            t = self.inv_mask * t
            s = self.inv_mask * s
            u = (x - t) * torch.exp(-s)
            inv_log_det_jacobian = -torch.sum(s, dim=1, keepdim=True)
        else:
            # Apply the translation-only transformation (backward mode)
            t = self.inv_mask * z
            u = x - t
            inv_log_det_jacobian = 0.0

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

        if self.affine:
            # Apply the affine transformation (forward mode).
            t, s = torch.chunk(z, chunks=2, dim=1)
            s = self.scale_activation(s)
            t = self.inv_mask * t
            s = self.inv_mask * s
            x = u * torch.exp(s) + t
            log_det_jacobian = torch.sum(s, dim=1, keepdim=True)
        else:
            # Apply the translation-only transformation (forward mode)
            t = self.inv_mask * z
            x = u + t
            log_det_jacobian = 0.0

        return x, log_det_jacobian


class CouplingLayer2d(nn.Module):
    """RealNVP 2D coupling layer."""
    def __init__(self, in_features, network, n_blocks, channels, affine=True, channel_wise=False, reverse=False):
        """
        Build a ResNet-based coupling layer as specified in RealNVP paper.

        :param in_features: The size of the input.
        :param network: The network conditioner to use. It can be either 'resnet' or 'densenet'.
        :param n_blocks: The number of residual blocks or dense blocks.
        :param channels: The number of output channels of each convolutional layer.
        :param affine: Whether to use affine transformation. If False then use only translation (as in NICE).
        :param channel_wise: Whether to use channel-wise coupling mask.
                             Defaults to False, i.e. chessboard coupling mask.
        :param reverse: Whether to reverse the mask used in the coupling layer. Useful for alternating masks.
        """
        super(CouplingLayer2d, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.n_blocks = n_blocks
        self.channels = channels
        self.affine = affine
        self.channel_wise = channel_wise
        self.reverse = reverse

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
        out_channels = in_channels * 2 if self.affine else in_channels
        if network == 'resnet':
            self.network = ResidualNetwork(in_channels, self.channels, out_channels, self.n_blocks)
        elif network == 'densenet':
            self.network = DenseNetwork(in_channels, self.channels, out_channels, self.n_blocks)
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
            if self.reverse:
                mx, my = torch.chunk(x, chunks=2, dim=1)
            else:
                my, mx = torch.chunk(x, chunks=2, dim=1)

            # Get the parameters
            z = self.network(mx)

            if self.affine:
                # Apply the affine transformation (backward mode)
                t, s = torch.chunk(z, chunks=2, dim=1)
                s = self.scale_activation(s)
                my = (my - t) * torch.exp(-s)
                inv_log_det_jacobian = -torch.sum(s.view(batch_size, -1), dim=1, keepdim=True)
            else:
                # Apply the translation-only transformation (backward mode)
                my = my - z
                inv_log_det_jacobian = 0.0

            if self.reverse:
                u = torch.cat([mx, my], dim=1)
            else:
                u = torch.cat([my, mx], dim=1)
        else:
            # Get the parameters
            mx = self.mask * x
            z = self.network(mx)

            if self.affine:
                # Apply the affine transformation (backward mode)
                t, s = torch.chunk(z, chunks=2, dim=1)
                s = self.scale_activation(s)
                t = self.inv_mask * t
                s = self.inv_mask * s
                u = (x - t) * torch.exp(-s)
                inv_log_det_jacobian = -torch.sum(s.view(batch_size, -1), dim=1, keepdim=True)
            else:
                # Apply the translation-only transformation (backward mode)
                t = self.inv_mask * z
                u = x - t
                inv_log_det_jacobian = 0.0

        return u, inv_log_det_jacobian

    def forward(self, u):
        """
        Evaluate the layer given some inputs (forward mode).

        :param u: The inputs.
        :return: The tensor result of the layer.
        """
        batch_size = u.shape[0]

        if self.channel_wise:
            if self.reverse:
                mu, mv = torch.chunk(u, chunks=2, dim=1)
            else:
                mv, mu = torch.chunk(u, chunks=2, dim=1)

            # Get the parameters
            z = self.network(mu)

            if self.affine:
                # Apply the affine transformation (forward mode)
                t, s = torch.chunk(z, chunks=2, dim=1)
                s = self.scale_activation(s)
                mv = mv * torch.exp(s) + t
                log_det_jacobian = torch.sum(s.view(batch_size, -1), dim=1, keepdim=True)
            else:
                # Apply the translation-only transformation (forward mode)
                mv = mv + z
                log_det_jacobian = 0.0

            if self.reverse:
                x = torch.cat([mu, mv], dim=1)
            else:
                x = torch.cat([mv, mu], dim=1)
        else:
            # Get the parameters
            mu = self.mask * u
            z = self.network(mu)

            if self.affine:
                # Apply the affine transformation (forward mode)
                t, s = torch.chunk(z, chunks=2, dim=1)
                s = self.scale_activation(s)
                t = self.inv_mask * t
                s = self.inv_mask * s
                x = u * torch.exp(s) + t
                log_det_jacobian = torch.sum(s.view(batch_size, -1), dim=1, keepdim=True)
            else:
                # Apply the translation-only transformation (forward mode)
                t = self.inv_mask * z
                x = u + t
                log_det_jacobian = 0.0

        return x, log_det_jacobian


class CouplingBlock2d(nn.Module):
    """RealNVP 2D coupling block, consisting of checkboard/channelwise couplings and squeeze operation."""
    def __init__(self, in_features, network, n_blocks, channels, affine=True, last_block=False):
        """
        Build a RealNVP 2d coupling block.

        :param in_features: The size of the input.
        :param network: The network conditioner to use. It can be either 'resnet' or 'densenet'.
        :param n_blocks: The number of residual blocks or dense blocks.
        :param channels: The number of output channels of each convolutional layer.
        :param affine: Whether to use affine transformation. If False then use only translation (as in NICE).
        :param last_block: Whether it is the last block (i.e. no channelwise-masked couplings) or not.
        """
        super(CouplingBlock2d, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.network = network
        self.n_blocks = n_blocks
        self.channels = channels
        self.affine = affine
        self.last_block = last_block

        if self.last_block:
            # Build the input couplings (consisting of 4 checkboard-masked couplings)
            self.in_couplings = nn.ModuleList([
                CouplingLayer2d(
                    self.in_features, self.network, self.n_blocks, self.channels, self.affine,
                    channel_wise=False, reverse=False
                ),
                BatchNormLayer(self.in_features),
                CouplingLayer2d(
                    self.in_features, self.network, self.n_blocks, self.channels, self.affine,
                    channel_wise=False, reverse=True
                ),
                BatchNormLayer(self.in_features),
                CouplingLayer2d(
                    self.in_features, self.network, self.n_blocks, self.channels, self.affine,
                    channel_wise=False, reverse=False
                ),
                BatchNormLayer(self.in_features),
                CouplingLayer2d(
                    self.in_features, self.network, self.n_blocks, self.channels, self.affine,
                    channel_wise=False, reverse=True
                ),
                BatchNormLayer(self.in_features)
            ])
        else:
            # Build the input couplings (consisting of 3 checkboard-masked couplings)
            self.in_couplings = nn.ModuleList([
                CouplingLayer2d(
                    self.in_features, self.network, self.n_blocks, self.channels, self.affine,
                    channel_wise=False, reverse=False
                ),
                BatchNormLayer(self.in_features),
                CouplingLayer2d(
                    self.in_features, self.network, self.n_blocks, self.channels, self.affine,
                    channel_wise=False, reverse=True
                ),
                BatchNormLayer(self.in_features),
                CouplingLayer2d(
                    self.in_features, self.network, self.n_blocks, self.channels, self.affine,
                    channel_wise=False, reverse=False
                ),
                BatchNormLayer(self.in_features)
            ])

            # Compute the size of the input (after squeezing operation)
            squeezed_features = (self.in_channels * 4, self.in_height // 2, self.in_width // 2)

            # Build the output couplings (consisting of 3 channel-wise-masked couplings)
            self.out_couplings = nn.ModuleList([
                CouplingLayer2d(
                    squeezed_features, self.network, self.n_blocks, self.channels * 2, self.affine,
                    channel_wise=True, reverse=False
                ),
                BatchNormLayer(squeezed_features),
                CouplingLayer2d(
                    squeezed_features, self.network, self.n_blocks, self.channels * 2, self.affine,
                    channel_wise=True, reverse=True
                ),
                BatchNormLayer(squeezed_features),
                CouplingLayer2d(
                    squeezed_features, self.network, self.n_blocks, self.channels * 2, self.affine,
                    channel_wise=True, reverse=False
                ),
                BatchNormLayer(squeezed_features)
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
        :return: The outputs of checkboard coupling if last_block=True
                 and the outputs of squeeze-channelwise coupling-unsqueeze-checkboard coupling if last_block=False.
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

            # Unsqueeze the outputs
            x = unsqueeze_depth2d(x)

        return x, inv_log_det_jacobian

    def forward(self, x):
        """
        Evaluate the coupling block (forward mode).

        :param x: The inputs.
        :return: The outputs of checkboard coupling if last_block=True
                 and the outputs of checkboard coupling-squeeze-channelwise coupling-unsqueeze if last_block=False.
        """
        log_det_jacobian = 0.0

        if not self.last_block:
            # Squeeze the inputs
            x = squeeze_depth2d(x)

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
