import torch
import numpy as np

from spnflow.torch.models.abstract import AbstractModel
from spnflow.torch.layers.flows import CouplingLayer1d, CouplingBlock2d
from spnflow.torch.layers.flows import AutoregressiveLayer
from spnflow.torch.layers.utils import BatchNormLayer
from spnflow.torch.utils import get_activation_class
from spnflow.torch.utils import squeeze_depth2d, unsqueeze_depth2d


class AbstractNormalizingFlow(AbstractModel):
    """Abstract Normalizing Flow model."""
    def __init__(self, in_features, dequantize=False, logit=None, in_base=None):
        """
        Initialize an abstract Normalizing Flow model.
        :param in_features: The input size.
        :param dequantize: Whether to apply the dequantization transformation.
        :param logit: The logit factor to use. Use None to disable the logit transformation.
        :param in_base: The input base distribution to use. If None, the standard Normal distribution is used.
        """
        super(AbstractNormalizingFlow, self).__init__(dequantize=dequantize, logit=logit)
        assert (type(in_features) == int and in_features > 0) or \
               (len(in_features) == 3 and all(map(lambda d: d > 0, in_features)))
        self.in_features = in_features

        # Build the base distribution, if necessary
        if in_base is None:
            self.in_base_loc = torch.nn.Parameter(torch.zeros(self.in_features, requires_grad=False))
            self.in_base_scale = torch.nn.Parameter(torch.ones(self.in_features, requires_grad=False))
            self.in_base = torch.distributions.Normal(self.in_base_loc, self.in_base_scale)
        else:
            self.in_base = in_base

        # Initialize the normalizing flow layers
        self.layers = torch.nn.ModuleList()

    def train(self, mode=True, base_mode=True):
        """
        Set the training mode.

        :param mode: The training mode for the flows layers.
        :param base_mode: The training mode for the in_base distribution.
        :return: Itself.
        """
        self.training = mode
        self.layers.train(mode)
        if isinstance(self.in_base, torch.nn.Module):
            self.in_base.train(base_mode)
        return self

    def eval(self):
        """
        Turn off the training mode for both the flows layers and the in_base distribution.

        :return: Itself.
        """
        return self.train(False, False)

    def forward(self, x):
        """
        Compute the log-likelihood given complete evidence.

        :param x: The inputs tensor.
        :return: The output of the model.
        """
        # Preprocess the samples
        batch_size = x.shape[0]
        x, inv_log_det_jacobian = self.preprocess(x)

        # Apply the normalizing flow layers
        for layer in self.layers:
            x, ildj = layer.inverse(x)
            inv_log_det_jacobian += ildj

        # Compute the prior log-likelihood
        prior = torch.sum(
            self.in_base.log_prob(x).view(batch_size, -1),
            dim=1, keepdim=True
        )

        # Return the final log-likelihood
        return prior + inv_log_det_jacobian

    @torch.no_grad()
    def mpe(self, x):
        raise NotImplementedError('Maximum at posteriori estimation is not implemented for Normalizing Flows')

    @torch.no_grad()
    def sample(self, n_samples):
        """
        Sample some values from the modeled distribution.

        :param n_samples: The number of samples.
        :return: The samples.
        """
        # Sample from the base distribution
        x = self.in_base.sample([n_samples])

        # Apply the normalizing flows in forward mode
        for layer in reversed(self.layers):
            x, _ = layer.forward(x)

        # Apply reverse preprocessing transformation
        x, _ = self.unpreprocess(x)
        return x


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
                 batch_norm=True
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
        """
        super(RealNVP1d, self).__init__(in_features, dequantize=dequantize, logit=logit, in_base=in_base)
        assert depth > 0
        assert units > 0
        self.n_flows = n_flows
        self.depth = depth
        self.units = units
        self.batch_norm = batch_norm

        # Build the coupling layers
        reverse = False
        for _ in range(self.n_flows):
            self.layers.append(
                CouplingLayer1d(self.in_features, self.depth, self.units, reverse=reverse)
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
                 channels=16
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
        """
        super(RealNVP2d, self).__init__(in_features, dequantize=dequantize, logit=logit, in_base=in_base)
        assert n_flows > 0
        assert n_blocks > 0
        assert channels > 0
        self.n_flows = n_flows
        self.network = network
        self.n_blocks = n_blocks
        self.channels = channels

        # Build the coupling blocks
        channels = self.channels
        in_features = self.in_features
        for _ in range(n_flows):
            coupling_block = CouplingBlock2d(in_features, self.network, self.n_blocks, channels, last_block=False)
            self.layers.append(coupling_block)

            # Halve the number of channels due to multi-scale architecture
            in_features = coupling_block.out_features
            in_features = (in_features[0] // 2, in_features[1], in_features[2])

            # Double the number of channels
            channels *= 2

        # Add the last coupling block
        self.last_block = CouplingBlock2d(in_features, self.network, self.n_blocks, channels, last_block=True)

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
        for layer in self.layers:
            x, ildj = layer.inverse(x)
            x, z = torch.chunk(x, chunks=2, dim=1)
            slices.append(z)  # Collect the half-chunks (used for multi-scale architecture)
            inv_log_det_jacobian += ildj

        # Apply the last coupling block
        x, ildj = self.last_block.inverse(x)
        inv_log_det_jacobian += ildj

        # Re-concatenate all the half-cunks in reverse order and unsqueeze the results
        for z in reversed(slices):
            x = torch.cat([x, z], dim=1)
            x = unsqueeze_depth2d(x)

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

        # Collect the half-cunks in and squeeze the results
        slices = []
        for _ in range(len(self.layers)):
            x = squeeze_depth2d(x)
            x, z = torch.chunk(x, chunks=2, dim=1)
            slices.append(z)

        # Apply the last coupling block
        x, _ = self.last_block.forward(x)

        # Apply the normalizing flows in forward mode
        for layer, z in zip(reversed(self.layers), reversed(slices)):
            x = torch.cat([x, z], dim=1)
            x, _ = layer.forward(x)

        # Apply reverse preprocessing transformation
        x, _ = self.unpreprocess(x)
        return x


class MAF(AbstractNormalizingFlow):
    """Masked Autoregressive Flow (MAF) normalizing flow model."""
    def __init__(self,
                 in_features,
                 dequantize=False,
                 logit=None,
                 in_base=None,
                 n_flows=5,
                 depth=1,
                 units=128,
                 batch_norm=True,
                 activation='relu',
                 sequential=True,
                 rand_state=None
                 ):
        """
        Initialize a MAF.

        :param in_features: The number of input features.
        :param dequantize: Whether to apply the dequantization transformation.
        :param logit: The logit factor to use. Use None to disable the logit transformation.
        :param in_base: The input base distribution to use. If None, the standard Normal distribution is used.
        :param n_flows: The number of sequential autoregressive layers.
        :param depth: The number of hidden layers of flows conditioners.
        :param units: The number of hidden units per layer of flows conditioners.
        :param batch_norm: Whether to apply batch normalization after each autoregressive layer.
        :param activation: The activation function name to use for the flows conditioners hidden layers.
        :param sequential: If True build masks degrees sequentially, otherwise randomly.
        :param rand_state: The random state used to generate the masks degrees. Used only if sequential is False.
        """
        super(MAF, self).__init__(in_features, dequantize=dequantize, logit=logit, in_base=in_base)
        assert n_flows > 0
        assert depth > 0
        assert units > 0
        self.n_flows = n_flows
        self.depth = depth
        self.units = units
        self.batch_norm = batch_norm
        self.activation = get_activation_class(activation)
        self.sequential = sequential
        self.rand_state = rand_state

        # If necessary, instantiate a random state
        if not self.sequential and self.rand_state is None:
            self.rand_state = np.random.RandomState(42)

        # Build the autoregressive layers
        reverse = False
        for _ in range(self.n_flows):
            self.layers.append(
                AutoregressiveLayer(
                    self.in_features, self.depth, self.units, self.activation,
                    reverse=reverse, sequential=self.sequential, rand_state=self.rand_state
                )
            )

            # Append batch normalization after each layer, if specified
            if self.batch_norm:
                self.layers.append(BatchNormLayer(self.in_features))

            # Invert the input ordering
            reverse = not reverse
