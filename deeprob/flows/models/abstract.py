import abc
import torch
import torch.nn as nn
import torch.distributions as distributions

from deeprob.flows.utils import DequantizeLayer, LogitLayer


class AbstractNormalizingFlow(abc.ABC, nn.Module):
    """Abstract Normalizing Flow model."""
    def __init__(self, in_features, dequantize=False, logit=None, in_base=None):
        """
        Initialize an abstract Normalizing Flow model.
        :param in_features: The input size.
        :param dequantize: Whether to apply the dequantization transformation.
        :param logit: The logit factor to use. Use None to disable the logit transformation.
        :param in_base: The input base distribution to use. If None, the standard Normal distribution is used.
        """
        super(AbstractNormalizingFlow, self).__init__()
        assert (type(in_features) == int and in_features > 0) or \
               (len(in_features) == 3 and all(map(lambda d: d > 0, in_features)))
        self.in_features = in_features

        # Build the dequantization layer
        if dequantize:
            self.dequantize = DequantizeLayer()
        else:
            self.dequantize = None

        # Build the logit layer
        if logit is not None:
            assert 0.0 < logit < 1.0, "The logit factor must be in (0.0, 1.0)"
            self.logit = LogitLayer(alpha=logit)
        else:
            self.logit = None

        # Build the base distribution, if necessary
        if in_base is None:
            self.in_base_loc = nn.Parameter(torch.zeros(self.in_features, requires_grad=False))
            self.in_base_scale = nn.Parameter(torch.ones(self.in_features, requires_grad=False))
            self.in_base = distributions.Normal(self.in_base_loc, self.in_base_scale)
        else:
            self.in_base = in_base

        # Initialize the normalizing flow layers
        self.layers = nn.ModuleList()

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

    def preprocess(self, x):
        """
        Preprocess the data batch before feeding it to the probabilistic model (forward mode).

        :param x: The input data batch.
        :return: The preprocessed data batch and the inv-log-det-jacobian.
        """
        inv_log_det_jacobian = 0.0
        if self.dequantize is not None:
            x, ildj = self.dequantize.inverse(x)
            inv_log_det_jacobian += ildj
        if self.logit is not None:
            x, ildj = self.logit.inverse(x)
            inv_log_det_jacobian += ildj
        return x, inv_log_det_jacobian

    def unpreprocess(self, x):
        """
        Preprocess the data batch before feeding it to the probabilistic model (backward mode).

        :param x: The input data batch.
        :return: The unpreprocessed data batch and the log-det-jacobian.
        """
        log_det_jacobian = 0.0
        if self.logit is not None:
            x, ldj = self.logit.forward(x)
            log_det_jacobian += ldj
        if self.dequantize is not None:
            x, ldj = self.dequantize.forward(x)
            log_det_jacobian += ldj
        return x, log_det_jacobian

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

    def log_prob(self, x):
        return self.forward(x)

    @torch.no_grad()
    def mpe(self, x):
        raise NotImplementedError('Maximum at Posteriori Estimation is not implemented for Normalizing Flows')

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

    def apply_constraints(self):
        """Apply the constraints specified by the model."""
        pass
