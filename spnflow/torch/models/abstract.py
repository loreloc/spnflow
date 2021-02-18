import abc
import torch


class AbstractModel(abc.ABC, torch.nn.Module):
    """Abstract class for deep probabilistic models."""
    def __init__(self):
        super(AbstractModel, self).__init__()

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
