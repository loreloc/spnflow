import torch
import numpy as np

from spnflow.torch.models.abstract import AbstractModel
from spnflow.torch.layers.flows import CouplingLayer, AutoregressiveLayer, BatchNormLayer, LogitLayer
from spnflow.torch.utils import torch_get_activation


class AbstractNormalizingFlow(AbstractModel):
    """Abstract Normalizing Flow model."""
    def __init__(self, in_features, n_flows=5, logit=False, in_base=None):
        """
        Initialize an abstract Normalizing Flow model.

        :param in_features: The number of input features.
        :param n_flows: The number of sequential coupling flows.
        :param logit: Whether to apply logit transformation on the input layer.
        :param in_base: The input base distribution to use. If None, the standard Normal distribution is used.
        """
        super(AbstractNormalizingFlow, self).__init__()
        assert in_features > 0
        assert n_flows > 0
        self.in_features = in_features
        self.n_flows = n_flows
        self.logit = logit

        # Build the base distribution, if necessary
        if in_base is None:
            self.in_base_loc = torch.nn.Parameter(torch.zeros([self.in_features], requires_grad=False))
            self.in_base_scale = torch.nn.Parameter(torch.ones([self.in_features], requires_grad=False))
            self.in_base = torch.distributions.Normal(self.in_base_loc, self.in_base_scale)
        else:
            self.in_base = in_base

        # Initialize the normalizing flow layers
        # Moreover, append the logit transformation, if specified
        self.layers = torch.nn.ModuleList()
        if self.logit:
            self.layers.append(LogitLayer())

    def forward(self, x):
        """
        Compute the log-likelihood given complete evidence.

        :param x: The inputs tensor.
        :return: The output of the model.
        """
        inv_log_det_jacobian = 0.0
        for layer in self.layers:
            x, ildj = layer.inverse(x)
            inv_log_det_jacobian += ildj
        prior = torch.sum(self.in_base.log_prob(x), dim=1, keepdim=True)
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
        x = self.in_base.sample([n_samples])
        for layer in reversed(self.layers):
            x, ldj = layer.forward(x)
        return x


class RealNVP(AbstractNormalizingFlow):
    """Real Non-Volume-Preserving (RealNVP) normalizing flow model."""
    def __init__(self,
                 in_features,
                 n_flows=5,
                 depth=1,
                 units=128,
                 batch_norm=True,
                 activation='relu',
                 logit=False,
                 in_base=None,
                 ):
        """
        Initialize a RealNVP.

        :param in_features: The number of input features.
        :param n_flows: The number of sequential coupling flows.
        :param depth: The number of hidden layers of flows conditioners.
        :param units: The number of hidden units per layer of flows conditioners.
        :param batch_norm: Whether to apply batch normalization after each coupling layer.
        :param activation: The activation function name to use for the flows conditioners hidden layers.
        :param logit: Whether to apply logit transformation on the input layer.
        :param in_base: The input base distribution to use. If None, the standard Normal distribution is used.
        """
        super(RealNVP, self).__init__(in_features, n_flows, logit, in_base)
        assert depth > 0
        assert units > 0
        self.depth = depth
        self.units = units
        self.batch_norm = batch_norm
        self.activation = torch_get_activation(activation)

        # Build the coupling layers
        reverse = False
        for _ in range(self.n_flows):
            self.layers.append(
                CouplingLayer(self.in_features, self.depth, self.units, self.activation, reverse=reverse)
            )

            # Append batch normalization after each layer, if specified
            if self.batch_norm:
                self.layers.append(BatchNormLayer(self.in_features))

            # Invert the input ordering
            reverse = not reverse


class MAF(AbstractNormalizingFlow):
    """Masked Autoregressive Flow (MAF) normalizing flow model."""
    def __init__(self,
                 in_features,
                 n_flows=5,
                 depth=1,
                 units=128,
                 batch_norm=True,
                 activation='relu',
                 sequential=True,
                 logit=False,
                 in_base=None,
                 rand_state=None
                 ):
        """
        Initialize a MAF.

        :param in_features: The number of input features.
        :param n_flows: The number of sequential autoregressive layers.
        :param depth: The number of hidden layers of flows conditioners.
        :param units: The number of hidden units per layer of flows conditioners.
        :param batch_norm: Whether to apply batch normalization after each autoregressive layer.
        :param activation: The activation function name to use for the flows conditioners hidden layers.
        :param sequential: If True build masks degrees sequentially, otherwise randomly.
        :param logit: Whether to apply logit transformation on the input layer.
        :param in_base: The input base distribution to use. If None, the standard Normal distribution is used.
        :param rand_state: The random state used to generate the masks degrees. Used only if sequential is False.
        """
        super(MAF, self).__init__(in_features, n_flows, logit, in_base)
        assert depth > 0
        assert units > 0
        self.depth = depth
        self.units = units
        self.batch_norm = batch_norm
        self.activation = torch_get_activation(activation)
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
                    sequential=self.sequential, reverse=reverse, rand_state=self.rand_state
                )
            )

            # Append batch normalization after each layer, if specified
            if self.batch_norm:
                self.layers.append(BatchNormLayer(self.in_features))

            # Invert the input ordering
            reverse = not reverse
