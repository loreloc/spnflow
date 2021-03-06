import abc
import torch

from spnflow.torch.layers.utils import LogitLayer


class AbstractModel(abc.ABC, torch.nn.Module):
    """Abstract class for deep probabilistic models."""
    def __init__(self, logit=False):
        """
        Initialize the model.

        :param logit: Whether to apply logit transformation on the input layer.
        """
        super(AbstractModel, self).__init__()
        self.logit = LogitLayer() if logit else None

    def preprocess(self, x):
        """
        Preprocess the data batch before feeding it to the probabilistic model (forward mode).

        :param x: The input data batch.
        :return: The preprocessed data batch and the inv-log-det-jacobian.
        """
        inv_log_det_jacobian = 0.0
        if self.logit is not None:
            x, inv_log_det_jacobian = self.logit.inverse(x)
        return x, inv_log_det_jacobian

    def unpreprocess(self, x):
        """
        Preprocess the data batch before feeding it to the probabilistic model (backward mode).

        :param x: The input data batch.
        :return: The unpreprocessed data batch and the log-det-jacobian.
        """
        log_det_jacobian = 0.0
        if self.logit is not None:
            x, log_det_jacobian = self.logit.forward(x)
        return x, log_det_jacobian

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def mpe(self, x):
        pass

    @abc.abstractmethod
    def sample(self, n_samples):
        pass

    def log_prob(self, x):
        return super(AbstractModel, self).__call__(x)

    def apply_constraints(self):
        pass
