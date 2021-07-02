import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeprob.flows.utils import BatchNormLayer
from deeprob.flows.layers.coupling import CouplingLayer1d, CouplingBlock2d
from deeprob.flows.models.abstract import AbstractNormalizingFlow


class RealNVP1d(AbstractNormalizingFlow):
    """Real Non-Volume-Preserving (RealNVP) 1D normalizing flow model."""
    def __init__(self,
                 in_features,
                 dequantize=False,
                 logit=None,
                 in_base=None,
                 n_flows=5,
                 depth=1,
                 units=128,
                 batch_norm=True,
                 affine=True
                 ):
        """
        Initialize a RealNVP.

        :param in_features: The number of input features.
        :param dequantize: Whether to apply the dequantization transformation.
        :param logit: The logit factor to use. Use None to disable the logit transformation.
        :param in_base: The input base distribution to use. If None, the standard Normal distribution is used.
        :param n_flows: The number of sequential coupling flows.
        :param depth: The number of hidden layers of flows conditioners.
        :param units: The number of hidden units per layer of flows conditioners.
        :param batch_norm: Whether to apply batch normalization after each coupling layer.
        :param affine: Whether to use affine transformation. If False then use only translation (as in NICE).
        """
        super(RealNVP1d, self).__init__(in_features, dequantize=dequantize, logit=logit, in_base=in_base)
        assert depth > 0
        assert units > 0
        self.n_flows = n_flows
        self.depth = depth
        self.units = units
        self.batch_norm = batch_norm
        self.affine = affine

        # Build the coupling layers
        reverse = False
        for _ in range(self.n_flows):
            self.layers.append(
                CouplingLayer1d(
                    self.in_features, self.depth, self.units,
                    affine=self.affine, reverse=reverse
                )
            )

            # Append batch normalization after each layer, if specified
            if self.batch_norm:
                self.layers.append(BatchNormLayer(self.in_features))

            # Invert the input ordering
            reverse = not reverse


class RealNVP2d(AbstractNormalizingFlow):
    """Real Non-Volume-Preserving (RealNVP) 2D normalizing flow model based on ResNets."""
    def __init__(self,
                 in_features,
                 dequantize=False,
                 logit=None,
                 in_base=None,
                 network='resnet',
                 n_flows=1,
                 n_blocks=2,
                 channels=16,
                 affine=True
                 ):
        """
        Initialize a RealNVP.

        :param in_features: The input size.
        :param dequantize: Whether to apply the dequantization transformation.
        :param logit: The logit factor to use. Use None to disable the logit transformation.
        :param in_base: The input base distribution to use. If None, the standard Normal distribution is used.
        :param network: The neural network conditioner to use. It can be either 'resnet' or 'densenet'.
        :param n_flows: The number of sequential multi-scale architectures.
        :param n_blocks: The number of residual blocks or dense blocks.
        :param channels: The number of output channels of each convolutional layer.
        :param affine: Whether to use affine transformation. If False then use only translation (as in NICE).
        """
        super(RealNVP2d, self).__init__(in_features, dequantize=dequantize, logit=logit, in_base=in_base)
        assert n_flows > 0
        assert n_blocks > 0
        assert channels > 0
        self.n_flows = n_flows
        self.network = network
        self.n_blocks = n_blocks
        self.channels = channels
        self.affine = affine
        self.perm_matrices = torch.nn.ParameterList()

        # Build the coupling blocks
        channels = self.channels
        in_features = self.in_features
        for _ in range(n_flows):
            coupling_block = CouplingBlock2d(
                in_features, self.network, self.n_blocks, channels,
                affine=self.affine, last_block=False
            )
            self.layers.append(coupling_block)

            # Initialize the order matrix for downscaling-upscaling
            self.perm_matrices.append(
                nn.Parameter(self.__build_permutation_matrix(in_features[0]), requires_grad=False)
            )

            # Halve the number of channels due to multi-scale architecture
            in_features = (in_features[0] * 2, in_features[1] // 2, in_features[2] // 2)

            # Double the number of channels
            channels *= 2

        # Add the last coupling block
        self.last_block = CouplingBlock2d(
            in_features, self.network, self.n_blocks, channels,
            affine=self.affine, last_block=True
        )

    @staticmethod
    def __build_permutation_matrix(channels):
        """
        Build the permutation matrix that defines (a non-trivial) variables ordering
        when downscaling or upscaling as in RealNVP.

        :param channels: The number of input channels.
        :return: The permutation matrix tensor.
        """
        weights = np.zeros([channels * 4, channels, 2, 2], dtype=np.float32)
        ordering = np.array([
            [[[1., 0.],
              [0., 0.]]],
            [[[0., 0.],
              [0., 1.]]],
            [[[0., 1.],
              [0., 0.]]],
            [[[0., 0.],
              [1., 0.]]]
        ], dtype=np.float32)
        for i in range(channels):
            weights[4*i:4*i+4, i:i+1] = ordering
        permutation = np.array([4 * i + j for j in [0, 1, 2, 3] for i in range(channels)])
        return torch.tensor(weights[permutation], dtype=torch.float32)

    def forward(self, x):
        """
        Compute the log-likelihood given complete evidence.

        :param x: The inputs tensor.
        :return: The output of the model.
        """
        # Preprocess the samples
        batch_size = x.shape[0]
        x, inv_log_det_jacobian = self.preprocess(x)

        # Apply the coupling block layers
        slices = []
        for layer, p in zip(self.layers, self.perm_matrices):
            x, ildj = layer.inverse(x)
            inv_log_det_jacobian += ildj

            # Downscale the results and split them in half (i.e. multi-scale architecture)
            x = F.conv2d(x, p, stride=2)
            x, z = torch.chunk(x, chunks=2, dim=1)
            slices.append(z)

        # Apply the last coupling block
        x, ildj = self.last_block.inverse(x)
        inv_log_det_jacobian += ildj

        # Re-concatenate all the chunks in reverse order and upscale the results
        for p, z in zip(reversed(self.perm_matrices), reversed(slices)):
            x = torch.cat([x, z], dim=1)
            x = F.conv_transpose2d(x, p, stride=2)

        # Compute the prior log-likelihood
        prior = torch.sum(
            self.in_base.log_prob(x).view(batch_size, -1),
            dim=1, keepdim=True
        )

        # Return the final log-likelihood
        return prior + inv_log_det_jacobian

    @torch.no_grad()
    def sample(self, n_samples):
        """
        Sample some values from the modeled distribution.

        :param n_samples: The number of samples.
        :return: The samples.
        """
        # Sample from the base distribution
        x = self.in_base.sample([n_samples])

        # Collect the chunks in and upscale the results
        slices = []
        for p in self.perm_matrices:
            # Downscale the results and split them in half (i.e. multi-scale architecture)
            x = F.conv2d(x, p, stride=2)
            x, z = torch.chunk(x, chunks=2, dim=1)
            slices.append(z)

        # Apply the last coupling block
        x, _ = self.last_block.forward(x)

        # Apply the normalizing flows in forward mode
        for layer, p, z in zip(reversed(self.layers), reversed(self.perm_matrices), reversed(slices)):
            x = torch.cat([x, z], dim=1)
            x = F.conv_transpose2d(x, p, stride=2)
            x, _ = layer.forward(x)

        # Apply reverse preprocessing transformation
        x, _ = self.unpreprocess(x)
        return x
